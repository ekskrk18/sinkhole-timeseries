# -*- coding: utf-8 -*-
import os
import time
import requests
import pandas as pd
from datetime import timedelta
from io import StringIO

# =========================================================
# 사용자 설정
# =========================================================
AUTH_KEY = "3H0HXiOxQva9B14jsVL2kA"

BASE_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_nc_pt_api"

EVENTS_CSV = r"E:\20260124\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python\sinkhole_events.csv"
OUT_DIR    = r"E:\20260124\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python\kma_rn60m"

DAYS_BEFORE = 30
DAYS_AFTER  = 1

ITV_MIN = 60                 # 60분 간격
CHUNK_DAYS = 7               # 7일 단위로 쪼개기
SLEEP_SEC = 0.25
TIMEOUT_SEC = 60

# =========================================================
# 유틸
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_tm(dt: pd.Timestamp) -> str:
    """KMA tm1/tm2는 보통 YYYYMMDDHHMM (12자리)"""
    return dt.strftime("%Y%m%d%H%M")

def fetch_kma_text(obs: str, lat: float, lon: float, tm1: str, tm2: str, itv_min: int) -> str:
    params = {
        "obs": obs,               # rn_60m
        "tm1": tm1,
        "tm2": tm2,
        "itv": str(itv_min),
        "lon": f"{lon:.4f}",
        "lat": f"{lat:.4f}",
        "authKey": AUTH_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.text

def _looks_like_html(text: str) -> bool:
    t = (text or "").lstrip().lower()
    return t.startswith("<!doctype") or t.startswith("<html")

def _strip_comment_lines(text: str) -> str:
    # KMA typ01 응답에 #,* 등 주석라인이 섞일 수 있어서 제거
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("*"):
            continue
        out.append(ln)
    return "\n".join(out)

def parse_kma_table(text: str) -> pd.DataFrame:
    """
    KMA typ01 응답은 케이스가 섞여 나옵니다.
    - (A) 콤마(,)로 구분된 CSV 형태: 202407301125,rn_60m,lon,lat,value,flag
    - (B) 공백 구분 텍스트 테이블 형태

    둘 다 자동 감지해서 파싱합니다.
    """
    if not text or not text.strip():
        raise RuntimeError("빈 응답입니다(요청 실패/제한 가능).")

    if _looks_like_html(text):
        raise RuntimeError("응답이 HTML입니다(인증키/요청 오류 가능). raw 텍스트 확인 필요.")

    cleaned = _strip_comment_lines(text)

    # 첫 '데이터로 보이는' 라인을 찾아 형식을 판단
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("주석 제거 후 데이터가 없습니다.")

    # 데이터 라인 후보: 첫 토큰이 숫자로 시작
    first_data = None
    for ln in lines:
        tok0 = ln.split(",")[0].split()[0]
        if tok0[:1].isdigit():
            first_data = ln
            break

    if first_data is None:
        # 정말 이상한 응답이면 전체 첫줄로 판단
        first_data = lines[0]

    # -----------------------------------------------------
    # (A) CSV 콤마 구분 형태
    # -----------------------------------------------------
    if "," in first_data:
        # pandas로 유연하게 읽기: 컬럼 수가 가변이어도 처리
        df = pd.read_csv(
            StringIO(cleaned),
            header=None,
            sep=",",
            engine="python",
            comment=None
        )

        # 최소 5열: tm, obs, lon, lat, value (그 외 flag 등 추가 가능)
        if df.shape[1] < 5:
            raise RuntimeError(f"파싱 실패(CSV): 컬럼 수({df.shape[1]})가 너무 적습니다. 예시 라인: {first_data}")

        # 컬럼명 부여
        base_cols = ["tm", "obs", "lon", "lat", "value"]
        extra_cols = [f"col_{i}" for i in range(6, df.shape[1] + 1)]
        df.columns = base_cols + extra_cols

        # 타입 정리
        df["tm"] = df["tm"].astype(str).str.strip()
        df["obs_time"] = pd.to_datetime(df["tm"], format="%Y%m%d%H%M", errors="coerce")
        if df["obs_time"].isna().all():
            df["obs_time"] = pd.to_datetime(df["tm"], format="%Y%m%d%H", errors="coerce")

        for c in ["lon", "lat", "value"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # value가 들어있는 열만 남기고(필요시 나머지 유지 가능)
        return df

    # -----------------------------------------------------
    # (B) 공백 구분 테이블 형태 (기존 로직 개선)
    # -----------------------------------------------------
    data_lines = []
    for ln in lines:
        if ln.startswith("#") or ln.startswith("*"):
            continue
        tok0 = ln.split()[0]
        if tok0[:1].isdigit():
            data_lines.append(ln)

    if not data_lines:
        raise RuntimeError(f"파싱 실패(공백): 데이터 라인을 찾지 못했습니다. 첫 줄: {lines[0]}")

    rows = [ln.split() for ln in data_lines]
    min_cols = min(len(r) for r in rows) if rows else 0
    if min_cols < 2:
        raise RuntimeError(f"파싱 실패(공백): 데이터 컬럼 수가 2 미만입니다. 예시 라인: {data_lines[0]}")

    max_cols = max(len(r) for r in rows)
    colnames = ["tm", "value"] + [f"col_{i}" for i in range(3, max_cols + 1)]
    padded = [r + [None] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(padded, columns=colnames)

    df["tm"] = df["tm"].astype(str)
    df["obs_time"] = pd.to_datetime(df["tm"], format="%Y%m%d%H%M", errors="coerce")
    if df["obs_time"].isna().all():
        df["obs_time"] = pd.to_datetime(df["tm"], format="%Y%m%d%H", errors="coerce")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def download_rn60m_for_event(event_id: str, event_time: pd.Timestamp, lat: float, lon: float) -> pd.DataFrame:
    start = event_time - timedelta(days=DAYS_BEFORE)
    end   = event_time + timedelta(days=DAYS_AFTER)

    def expected_rows(t1: pd.Timestamp, t2: pd.Timestamp, itv_min: int) -> int:
        # inclusive/exclusive 애매함 방지로 +1 정도 여유
        minutes = max(0, int((t2 - t1).total_seconds() // 60))
        return (minutes // itv_min) + 1

    def fetch_window(t1: pd.Timestamp, t2: pd.Timestamp) -> pd.DataFrame:
        tm1 = to_tm(t1)
        tm2 = to_tm(t2)
        text = fetch_kma_text("rn_60m", lat, lon, tm1, tm2, ITV_MIN)
        df = parse_kma_table(text)

        # obs_time 정리
        if not df.empty and "obs_time" in df.columns:
            df = df.dropna(subset=["obs_time"]).sort_values("obs_time")
        return df

    # ------------------------------------------------------------------
    # 핵심: 1일 단위로 요청 (KMA가 장기간 요청을 잘라먹는 현상 회피)
    # ------------------------------------------------------------------
    cur = start
    parts = []

    while cur < end:
        nxt = min(cur + timedelta(days=1), end)  # ✅ 하루 단위
        df_part = fetch_window(cur, nxt)

        # 혹시 서버가 더 잘라먹으면(예: 몇 줄만 주는 경우) -> 6시간 단위로 재시도
        exp = expected_rows(cur, nxt, ITV_MIN)
        if len(df_part) < max(5, int(exp * 0.6)):  # 기대치의 60% 미만이면 "잘림" 의심
            sub_cur = cur
            sub_parts = []
            while sub_cur < nxt:
                sub_nxt = min(sub_cur + timedelta(hours=6), nxt)
                sub_df = fetch_window(sub_cur, sub_nxt)
                sub_parts.append(sub_df)
                time.sleep(SLEEP_SEC)
                sub_cur = sub_nxt
            df_part = pd.concat(sub_parts, ignore_index=True) if sub_parts else df_part

        parts.append(df_part)
        time.sleep(SLEEP_SEC)
        cur = nxt

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # 중복 제거/정렬
    if not df.empty and "obs_time" in df.columns:
        df = df.dropna(subset=["obs_time"]).sort_values("obs_time").drop_duplicates(subset=["obs_time"], keep="last")

    if df.empty:
        # 완전 실패면 그대로 반환 (상위에서 error 처리)
        df = pd.DataFrame(columns=["tm","obs","lon","lat","value","obs_time"])

    # 메타 추가
    df.insert(0, "event_id", event_id)
    df.insert(1, "event_time", event_time)
    df.insert(2, "event_latitude", lat)
    df.insert(3, "event_longitude", lon)

    # value -> rn_60m 통일
    if "value" in df.columns:
        df.rename(columns={"value": "rn_60m"}, inplace=True)
    else:
        # 공백테이블 케이스 대비(혹시 value가 이미 rn_60m로 들어온다면)
        if "rn_60m" not in df.columns:
            df["rn_60m"] = pd.NA

    # 요청창 기록(검증용)
    df["request_start"] = start
    df["request_end"] = end

    return df


# =========================================================
# 메인
# =========================================================
def main():
    ensure_dir(OUT_DIR)
    raw_dir = os.path.join(OUT_DIR, "_raw_failed")
    ensure_dir(raw_dir)

    ev = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    ev = ev.loc[:, ~ev.columns.astype(str).str.startswith("Unnamed")].copy()

    # event_time 파싱
    ev["event_time"] = (
        ev["event_time"].astype(str)
          .str.replace("KST", "", regex=False)
          .str.replace(".", "-", regex=False)
          .str.replace("/", "-", regex=False)
          .str.strip()
    )
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")

    # id / latitude / longitude 컬럼명이 다를 수도 있으니 안전 처리
    # (기존 파일이 id, latitude, longitude 라는 전제)
    needed = ["id", "event_time", "latitude", "longitude"]
    missing = [c for c in needed if c not in ev.columns]
    if missing:
        raise RuntimeError(f"sinkhole_events.csv에 필요한 컬럼이 없습니다: {missing} / 현재 컬럼: {list(ev.columns)}")

    ev = ev.dropna(subset=needed).copy()

    summary = []

    for _, row in ev.iterrows():
        event_id = str(int(row["id"])) if pd.notna(row["id"]) else "NA"
        event_time = row["event_time"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        print(f"[Event {event_id}] {event_time} lat={lat} lon={lon}")

        try:
            df = download_rn60m_for_event(event_id, event_time, lat, lon)

            out_path = os.path.join(OUT_DIR, f"event_{event_id}_rn60m.csv")
            df.to_csv(out_path, index=False, encoding="utf-8-sig")

            summary.append({
                "event_id": event_id,
                "event_time": str(event_time),
                "lat": lat,
                "lon": lon,
                "n_rows": int(len(df)),
                "output_csv": out_path,
                "error": ""
            })

        except Exception as e:
            # 실패 시 raw 응답 저장(진단용)
            raw_path = ""
            try:
                tm1 = to_tm(event_time - timedelta(hours=3))
                tm2 = to_tm(event_time)
                raw = fetch_kma_text("rn_60m", lat, lon, tm1, tm2, ITV_MIN)
                raw_path = os.path.join(raw_dir, f"event_{event_id}_raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(raw)
            except Exception:
                pass

            summary.append({
                "event_id": event_id,
                "event_time": str(event_time),
                "lat": lat,
                "lon": lon,
                "n_rows": 0,
                "output_csv": "",
                "error": f"{e} | raw_saved={raw_path}"
            })

        time.sleep(SLEEP_SEC)

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUT_DIR, "rn60m_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Done. Summary saved: {summary_path}")

if __name__ == "__main__":
    main()

