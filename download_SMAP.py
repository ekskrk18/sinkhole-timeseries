import os
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import earthaccess


# =========================
# 사용자 설정
# =========================
EVENTS_CSV = "sinkhole_events.csv"  # 같은 폴더에 두면 됨
OUT_DIR = "SMAP_sinkhole_rootzone_30d"
SHORT_NAME = "SPL4SMGP"  # SMAP L4 Global 3-hourly 9km (rootzone 포함)
BBOX_HALF_SIZE_DEG = 0.15  # 점 주변 bounding box 크기(0.15deg ~ 약 15km 수준). 필요시 0.05~0.3에서 조정
KST = timezone(timedelta(hours=9))


# =========================
# 유틸
# =========================
def parse_event_time_kst(s: str) -> datetime:
    """
    CSV event_time을 KST datetime으로 파싱.
    예) "2024-09-21 8:45", "2024-08-29 11:26", "2022-08-03 6:40"
    """
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt_naive = datetime.strptime(s, fmt)
            return dt_naive.replace(tzinfo=KST)
        except ValueError:
            pass
    # 혹시 한 자리 시/분 등 예외가 있으면 pandas로 fallback
    dt_parsed = pd.to_datetime(s, errors="raise")
    if dt_parsed.tzinfo is None:
        return dt_parsed.to_pydatetime().replace(tzinfo=KST)
    return dt_parsed.to_pydatetime().astimezone(KST)


def kst_window_to_utc(event_time_kst: datetime, days_before=30, days_after=1):
    start_kst = event_time_kst - timedelta(days=days_before)
    end_kst = event_time_kst + timedelta(days=days_after)
    start_utc = start_kst.astimezone(timezone.utc)
    end_utc = end_kst.astimezone(timezone.utc)
    return start_kst, end_kst, start_utc, end_utc


def build_bbox(lat: float, lon: float, half_size_deg: float):
    """
    earthaccess는 (west, south, east, north) = (minLon, minLat, maxLon, maxLat)
    """
    west = lon - half_size_deg
    east = lon + half_size_deg
    south = lat - half_size_deg
    north = lat + half_size_deg
    return (west, south, east, north)


def extract_time_from_filename(fname: str):
    """
    SMAP L4 파일명에 포함된 timestamp 파싱(최대한 유연하게).
    예: ..._20240829T030000_... 또는 ..._2024-08-29T03:00:00...
    실패하면 None 반환.
    """
    base = os.path.basename(fname)

    # 1) 20240829T030000
    m = re.search(r"(\d{8})T(\d{6})", base)
    if m:
        ymd, hms = m.group(1), m.group(2)
        dt_utc = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt_utc

    # 2) 2024-08-29T03:00:00
    m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})", base)
    if m:
        dt_utc = datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt_utc

    return None


def list_all_datasets(h5obj, prefix=""):
    """
    HDF5 내부 dataset 경로를 전부 모아 반환
    """
    paths = []
    for key in h5obj.keys():
        item = h5obj[key]
        p = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            paths.append(p)
        elif isinstance(item, h5py.Group):
            paths.extend(list_all_datasets(item, prefix=p))
    return paths


def find_dataset_path(all_paths, candidates):
    """
    all_paths 중에서 candidates(우선순위 리스트)와 매칭되는 dataset 경로 찾기
    """
    # exact match 우선
    for c in candidates:
        for p in all_paths:
            if p.endswith("/" + c):
                return p

    # contains match(후보 단어 포함)
    for c in candidates:
        for p in all_paths:
            if c in p:
                return p

    return None


def read_rootzone_nearest(h5_path: str, target_lat: float, target_lon: float):
    """
    SMAP L4 HDF5에서 target_lat/lon에 가장 가까운 격자의 sm_rootzone 값 추출
    반환: (sm_rootzone_value, nearest_lat, nearest_lon)
    """
    with h5py.File(h5_path, "r") as f:
        all_paths = list_all_datasets(f)

        # root zone soil moisture 후보 이름들(제품/버전에 따라 약간씩 다를 수 있어 후보를 넓게 둠)
        sm_candidates = [
            "sm_rootzone", "SM_rootzone", "soil_moisture_rootzone", "rootzone_soil_moisture"
        ]
        lat_candidates = ["cell_lat", "latitude", "lat"]
        lon_candidates = ["cell_lon", "longitude", "lon"]

        sm_path = find_dataset_path(all_paths, sm_candidates)
        lat_path = find_dataset_path(all_paths, lat_candidates)
        lon_path = find_dataset_path(all_paths, lon_candidates)

        if sm_path is None or lat_path is None or lon_path is None:
            raise RuntimeError(
                "HDF5 내부에서 필요한 dataset을 찾지 못했습니다.\n"
                f"  sm_path={sm_path}\n  lat_path={lat_path}\n  lon_path={lon_path}\n"
                "제품 버전/구조가 다른 경우 후보 리스트를 늘려야 합니다."
            )

        sm = f[sm_path][...]
        lat = f[lat_path][...]
        lon = f[lon_path][...]

        # lat/lon이 1D 또는 2D일 수 있어 둘 다 처리
        if lat.ndim == 1 and lon.ndim == 1 and sm.ndim == 2:
            # meshgrid 형태로 계산
            # (주의) 메모리 이슈가 있으면 이 블록을 더 최적화해야 하지만 SMAP 격자는 보통 감당 가능
            lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        else:
            lat2d, lon2d = lat, lon

        # 거리 계산(간단히 유클리드 근사; 한국 범위면 충분)
        dist2 = (lat2d - target_lat) ** 2 + (lon2d - target_lon) ** 2
        idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)

        val = sm[idx]
        near_lat = float(lat2d[idx])
        near_lon = float(lon2d[idx])

        # fill value 처리(있으면)
        if np.issubdtype(type(val), np.floating):
            if np.isnan(val):
                return np.nan, near_lat, near_lon

        return float(val), near_lat, near_lon


# =========================
# 메인
# =========================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # events 로드
    df = pd.read_csv(EVENTS_CSV)
    required = {"id", "event_time", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    # Earthdata 로그인
    earthaccess.login(persist=True)

    all_rows = []

    for _, row in df.iterrows():
        event_id = row["id"]
        event_time_kst = parse_event_time_kst(row["event_time"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        location = row.get("location", "")

        start_kst, end_kst, start_utc, end_utc = kst_window_to_utc(event_time_kst, 30, 1)
        bbox = build_bbox(lat, lon, BBOX_HALF_SIZE_DEG)

        print(f"\n[Event {event_id}] {location}")
        print(f"  event_time (KST): {event_time_kst.isoformat()}")
        print(f"  window (KST): {start_kst.isoformat()} ~ {end_kst.isoformat()}")
        print(f"  window (UTC): {start_utc.isoformat()} ~ {end_utc.isoformat()}")
        print(f"  bbox: {bbox}")

        # 그라뉼 검색
        granules = earthaccess.search_data(
            short_name=SHORT_NAME,
            temporal=(start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")),
            bounding_box=bbox,
        )

        if not granules:
            print("  -> 검색 결과 0개 (bbox/기간/short_name 확인 필요)")
            continue

        # 다운로드
        ev_dir = out_dir / f"event_{event_id}"
        gran_dir = ev_dir / "granules"
        ev_dir.mkdir(parents=True, exist_ok=True)
        gran_dir.mkdir(parents=True, exist_ok=True)

        downloaded = earthaccess.download(granules, local_path=str(gran_dir))
        downloaded = [p for p in downloaded if p]  # None 제거

        if not downloaded:
            print("  -> 다운로드 실패(권한/로그인/네트워크 확인)")
            continue

        # 각 파일에서 rootzone 값 추출
        event_rows = []
        for fpath in tqdm(downloaded, desc=f"Extract Event {event_id}", leave=False):
            fpath = str(fpath)
            obs_time_utc = extract_time_from_filename(fpath)
            obs_time_kst = obs_time_utc.astimezone(KST) if obs_time_utc else None

            try:
                sm_val, near_lat, near_lon = read_rootzone_nearest(fpath, lat, lon)
            except Exception as e:
                sm_val, near_lat, near_lon = np.nan, np.nan, np.nan
                # 필요하면 아래 print 활성화
                # print(f"    읽기 실패: {os.path.basename(fpath)} -> {e}")

            event_rows.append({
                "event_id": event_id,
                "location": location,
                "event_time_kst": event_time_kst.isoformat(),
                "target_lat": lat,
                "target_lon": lon,
                "nearest_lat": near_lat,
                "nearest_lon": near_lon,
                "obs_time_utc": obs_time_utc.isoformat() if obs_time_utc else "",
                "obs_time_kst": obs_time_kst.isoformat() if obs_time_kst else "",
                "sm_rootzone": sm_val,
                "file": os.path.basename(fpath),
                "file_path": fpath,
            })

        # 정렬 + 저장
        ev_df = pd.DataFrame(event_rows)
        if "obs_time_utc" in ev_df.columns:
            ev_df = ev_df.sort_values("obs_time_utc")

        ev_csv = ev_dir / f"event_{event_id}_rootzone_30d_to_1d.csv"
        ev_df.to_csv(ev_csv, index=False, encoding="utf-8-sig")
        print(f"  -> saved: {ev_csv}")

        all_rows.append(ev_df)

    # 전체 합본 저장
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_csv = out_dir / "ALL_events_rootzone_30d_to_1d.csv"
        all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

        # Excel(이벤트별 시트)
        xlsx_path = out_dir / "ALL_events_rootzone_30d_to_1d.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="ALL", index=False)
            for eid, g in all_df.groupby("event_id"):
                sheet = f"event_{eid}"
                g.to_excel(writer, sheet_name=sheet[:31], index=False)  # 엑셀 시트명 31자 제한
        print(f"\n[Done] merged saved: {all_csv}")
        print(f"[Done] excel saved:  {xlsx_path}")
    else:
        print("\n[Done] 저장할 결과가 없습니다(검색/다운로드 결과 0).")


if __name__ == "__main__":
    main()
