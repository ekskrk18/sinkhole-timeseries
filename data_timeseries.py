# -*- coding: utf-8 -*-
"""
Event별 시계열 플롯 생성
- Groundwater: lev(=lev) + ec (1개 figure, twin axis)
- Rain + SMAP: rn_60m (bar) + sm_rootzone (line, twin axis)
- 사고시점: 빨간색 dotted line
- 기간: 사고시점 기준 -30일 ~ +1일

[핵심 수정]
1) 지하수 daily 날짜(YYYYMMDD 정수/문자)를 올바르게 datetime으로 파싱
2) 시간컬럼 자동탐지 시 event_time/request_* 제외
3) tz-aware / tz-naive 혼합 비교 오류 방지(모두 tz 제거)
4) x축을 start~end로 강제 고정
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# =========================
# 사용자 설정
# =========================
EVENTS_CSV = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python\sinkhole_events.csv"

GW_DIR   = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python\gims_groundwater_daily"
SMAP_DIR = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python\SMAP_sinkhole_rootzone_30d"
RN_DIR   = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python\kma_rn60m"

OUT_DIR  = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python\event_timeseries_plots"

DAYS_BEFORE = 30
DAYS_AFTER  = 1

# =========================
# 유틸
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_csv_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="cp949")

def parse_event_time_series(s: pd.Series) -> pd.Series:
    ss = (s.astype(str)
            .str.replace("KST", "", regex=False)
            .str.replace(".", "-", regex=False)
            .str.replace("/", "-", regex=False)
            .str.strip())
    return pd.to_datetime(ss, errors="coerce")

def first_existing(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return hits[0]
    return None

# =========================
# 시간 컬럼 탐지/정리 (핵심)
# =========================
def find_datetime_col(df: pd.DataFrame) -> str:
    """
    - event_time/request_* 같은 메타 컬럼 제외
    - obs_time/time/datetime/date/tm/ymd 우선
    """
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}

    exclude_keywords = ["event_time", "event_", "request_", "start", "end"]
    candidate_cols = []
    for c in cols:
        lc = lower[c]
        if any(k in lc for k in exclude_keywords):
            continue
        candidate_cols.append(c)

    priority_exact = [
        "obs_time",
        "time_kst", "time_kr", "time_seoul",
        "time_utc",
        "datetime", "timestamp",
        "date",
        "tm",
        "ymd",
    ]
    for p in priority_exact:
        for c in candidate_cols:
            if lower[c] == p:
                return c

    priority_contains = ["obs_time", "time", "datetime", "timestamp", "date", "tm", "ymd"]
    for key in priority_contains:
        for c in candidate_cols:
            if key in lower[c]:
                return c

    raise RuntimeError(f"시간 컬럼을 찾지 못했습니다. 컬럼 목록: {cols}")

def _smart_parse_datetime(series: pd.Series, assume_local_tz: str = "Asia/Seoul") -> pd.Series:
    """
    숫자/문자 형태의 날짜를 똑똑하게 파싱하고,
    tz-aware(예: +00:00, Z)면 KST로 변환 후 tz 제거하여 tz-naive로 통일합니다.

    - YYYYMMDD (8), YYYYMMDDHH (10), YYYYMMDDHHMM (12), YYYYMMDDHHMMSS (14) 지원
    - ISO8601(+00:00, Z 등)이면 utc=True로 파싱 → Asia/Seoul 변환 → tz 제거
    """
    s_str = series.astype("string").str.strip()
    s_str = s_str.str.replace(r"\.0$", "", regex=True)

    has_tz = s_str.str.contains(r"(Z|[+\-]\d{2}:?\d{2})", case=False, regex=True).fillna(False)

    dt = pd.Series([pd.NaT] * len(s_str), index=s_str.index, dtype="datetime64[ns]")

    # tz-aware → UTC parse → KST convert → tz remove
    if has_tz.any():
        dt_tz = pd.to_datetime(s_str.loc[has_tz], errors="coerce", utc=True)
        dt_tz = dt_tz.dt.tz_convert(assume_local_tz)
        dt.loc[has_tz] = dt_tz.dt.tz_localize(None)

    # rest → digits-first parse
    rest = s_str.loc[~has_tz]
    digits = rest.str.replace(r"[^0-9]", "", regex=True)

    def parse_by_len(n, fmt):
        mask = digits.str.len() == n
        out = pd.Series([pd.NaT] * len(rest), index=rest.index)
        if mask.any():
            out.loc[mask] = pd.to_datetime(digits.loc[mask], format=fmt, errors="coerce")
        return out

    dt8  = parse_by_len(8,  "%Y%m%d")
    dt10 = parse_by_len(10, "%Y%m%d%H")
    dt12 = parse_by_len(12, "%Y%m%d%H%M")
    dt14 = parse_by_len(14, "%Y%m%d%H%M%S")

    dt_rest = dt14.combine_first(dt12).combine_first(dt10).combine_first(dt8)

    need_fallback = dt_rest.isna()
    if need_fallback.any():
        dt_rest.loc[need_fallback] = pd.to_datetime(rest.loc[need_fallback], errors="coerce")

    dt.loc[~has_tz] = dt_rest
    dt = pd.to_datetime(dt, errors="coerce")
    return dt

def clip_window(df: pd.DataFrame, tcol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out[tcol] = _smart_parse_datetime(out[tcol], assume_local_tz="Asia/Seoul")

    start2 = pd.to_datetime(start, errors="coerce")
    end2   = pd.to_datetime(end, errors="coerce")

    if getattr(start2, "tzinfo", None) is not None:
        start2 = start2.tz_localize(None)
    if getattr(end2, "tzinfo", None) is not None:
        end2 = end2.tz_localize(None)

    out = out.dropna(subset=[tcol])
    out = out[(out[tcol] >= start2) & (out[tcol] <= end2)].copy()
    out = out.sort_values(tcol)
    return out

# =========================
# 파일 찾기
# =========================
def find_groundwater_file(event_id: int) -> str | None:
    patterns = [
        os.path.join(GW_DIR, f"event_{event_id}_gennum_*_daily*.csv"),
        os.path.join(GW_DIR, f"event_{event_id}_gennum_*_daily*.txt"),
        os.path.join(GW_DIR, f"event_{event_id}_gennum_*_daily*.*"),
    ]
    return first_existing(patterns)

def find_smap_file(event_id: int) -> str | None:
    patterns = [
        os.path.join(SMAP_DIR, f"event_{event_id}", f"event_{event_id}_rootzone_30d_to_1d*.csv"),
        os.path.join(SMAP_DIR, f"event_{event_id}", f"*rootzone*30d*to*1d*.csv"),
        os.path.join(SMAP_DIR, f"event_{event_id}", "*.csv"),
    ]
    return first_existing(patterns)

def find_rain_file(event_id: int) -> str | None:
    patterns = [
        os.path.join(RN_DIR, f"event_{event_id}_rn60m*.csv"),
        os.path.join(RN_DIR, f"event_{event_id}*rn60m*.csv"),
    ]
    return first_existing(patterns)

# =========================
# 플롯 (변경: GW는 lev+ec 한 그림)
# =========================
def plot_groundwater(event_id: int, event_time: pd.Timestamp, df_gw: pd.DataFrame, out_png: str):
    start = event_time - timedelta(days=DAYS_BEFORE)
    end   = event_time + timedelta(days=DAYS_AFTER)

    tcol = find_datetime_col(df_gw)
    df = clip_window(df_gw, tcol, start, end)

    # ---- 컬럼명 정리: lev(우선) / lev(대체) -> lev로 통일, ec는 그대로 ----
    # lev/lev 통일
    if "lev" not in df.columns:
        # 대소문자/다른 표기 대응
        for cc in df.columns:
            if str(cc).lower() == "lev":
                df.rename(columns={cc: "lev"}, inplace=True)
                break

    if "lev" not in df.columns and "lev" in df.columns:
        df.rename(columns={"lev": "lev"}, inplace=True)
    elif "lev" not in df.columns:
        for cc in df.columns:
            if str(cc).lower() == "lev":
                df.rename(columns={cc: "lev"}, inplace=True)
                break

    # ec 통일
    if "ec" not in df.columns:
        for cc in df.columns:
            if str(cc).lower() == "ec":
                df.rename(columns={cc: "ec"}, inplace=True)
                break

    # 숫자 변환
    if "lev" in df.columns:
        df["lev"] = pd.to_numeric(df["lev"], errors="coerce")
    if "ec" in df.columns:
        df["ec"] = pd.to_numeric(df["ec"], errors="coerce")

    # 디버그
    print(
        f"[GW] event {event_id} time_col={tcol} rows_before={len(df_gw)} rows_after_clip={len(df)} "
        f"non-null(lev/ec)=("
        f"{df['lev'].notna().sum() if 'lev' in df.columns else 'NA'}, "
        f"{df['ec'].notna().sum() if 'ec' in df.columns else 'NA'})"
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # lev(좌측)
    if "lev" in df.columns and df["lev"].notna().any():
        ax1.plot(df[tcol], df["lev"], linestyle="-", label="Groundwater level (lev)")
        ax1.set_ylabel("Groundwater level (lev)")
    else:
        ax1.text(0.01, 0.85, "lev: no data", transform=ax1.transAxes)

    # ec(우측)
    ax2 = ax1.twinx()
    if "ec" in df.columns and df["ec"].notna().any():
        ax2.plot(df[tcol], df["ec"], linestyle="--", label="EC (ec)")
        ax2.set_ylabel("Electrical Conductivity (ec)")
    else:
        ax1.text(0.01, 0.70, "ec: no data", transform=ax1.transAxes)

    # 사고시점 + x축 고정
    ax1.axvline(event_time, color="red", linestyle=":", linewidth=2)
    ax1.set_xlim(start, end)
    ax1.grid(True, alpha=0.3)

    # 범례 합치기
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if (h1 + h2):
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    ax1.set_title(f"Event {event_id} Groundwater (lev + ec) | {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# =========================
# 플롯 (그대로 유지: Rain + SMAP)
# =========================
def plot_rain_smap(event_id: int, event_time: pd.Timestamp, df_rn: pd.DataFrame, df_smap: pd.DataFrame, out_png: str):
    start = event_time - timedelta(days=DAYS_BEFORE)
    end   = event_time + timedelta(days=DAYS_AFTER)

    # rain
    tcol_r = find_datetime_col(df_rn)
    rn = clip_window(df_rn, tcol_r, start, end)
    if "rn_60m" not in rn.columns:
        for cc in rn.columns:
            if str(cc).lower() == "rn_60m":
                rn.rename(columns={cc: "rn_60m"}, inplace=True)
                break
    if "rn_60m" in rn.columns:
        rn["rn_60m"] = pd.to_numeric(rn["rn_60m"], errors="coerce")

    # smap
    tcol_s = find_datetime_col(df_smap)
    sm = clip_window(df_smap, tcol_s, start, end)
    if "sm_rootzone" not in sm.columns:
        for cc in sm.columns:
            if str(cc).lower() == "sm_rootzone":
                sm.rename(columns={cc: "sm_rootzone"}, inplace=True)
                break
    if "sm_rootzone" in sm.columns:
        sm["sm_rootzone"] = pd.to_numeric(sm["sm_rootzone"], errors="coerce")

    print(f"[RN] event {event_id} time_col={tcol_r} rows_after_clip={len(rn)}")
    print(f"[SM] event {event_id} time_col={tcol_s} rows_after_clip={len(sm)} "
          f"non-null(sm_rootzone)={sm['sm_rootzone'].notna().sum() if 'sm_rootzone' in sm.columns else 'NA'}")

    fig, ax1 = plt.subplots(figsize=(12, 5))

    if "rn_60m" in rn.columns and rn["rn_60m"].notna().any():
        ax1.bar(rn[tcol_r], rn["rn_60m"], width=0.03, label="rn_60m")
        ax1.set_ylabel("Rain (rn_60m)")
    else:
        ax1.text(0.01, 0.85, "rn_60m: no data", transform=ax1.transAxes)

    ax2 = ax1.twinx()
    if "sm_rootzone" in sm.columns and sm["sm_rootzone"].notna().any():
        ax2.plot(sm[tcol_s], sm["sm_rootzone"], label="sm_rootzone", linestyle="-")
        ax2.set_ylabel("SMAP rootzone (sm_rootzone)")
    else:
        ax2.text(0.01, 0.70, "sm_rootzone: no data", transform=ax1.transAxes)

    ax1.axvline(event_time, color="red", linestyle=":", linewidth=2)
    ax1.set_xlim(start, end)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if (h1 + h2):
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Event {event_id} Rain (rn_60m) + SMAP (sm_rootzone) | {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# =========================
# 메인
# =========================
def main():
    ensure_dir(OUT_DIR)

    ev = load_csv_any(EVENTS_CSV)
    ev = ev.loc[:, ~ev.columns.astype(str).str.startswith("Unnamed")].copy()

    needed = ["id", "event_time", "latitude", "longitude"]
    miss = [c for c in needed if c not in ev.columns]
    if miss:
        raise RuntimeError(f"sinkhole_events.csv에 필요한 컬럼이 없습니다: {miss} / 현재: {list(ev.columns)}")

    ev["event_time"] = parse_event_time_series(ev["event_time"])
    ev = ev.dropna(subset=needed).copy()

    for _, row in ev.iterrows():
        event_id = int(row["id"])
        event_time = row["event_time"]

        print(f"\n[Event {event_id}] {event_time}")

        gw_path = find_groundwater_file(event_id)
        sm_path = find_smap_file(event_id)
        rn_path = find_rain_file(event_id)

        if gw_path:
            print(f"  - GW file: {gw_path}")
            df_gw = load_csv_any(gw_path)
            out1 = os.path.join(OUT_DIR, f"event_{event_id:02d}_groundwater_lev_ec.png")
            plot_groundwater(event_id, event_time, df_gw, out1)
            print(f"  ✓ saved: {out1}")
        else:
            print(f"  - Groundwater file not found for event {event_id}")

        if rn_path and sm_path:
            print(f"  - Rain file: {rn_path}")
            print(f"  - SMAP file: {sm_path}")
            df_rn = load_csv_any(rn_path)
            df_sm = load_csv_any(sm_path)
            out2 = os.path.join(OUT_DIR, f"event_{event_id:02d}_rain_rn60m_and_smap_rootzone.png")
            plot_rain_smap(event_id, event_time, df_rn, df_sm, out2)
            print(f"  ✓ saved: {out2}")
        else:
            if not rn_path:
                print(f"  - Rain file not found for event {event_id}")
            if not sm_path:
                print(f"  - SMAP file not found for event {event_id}")

    print("\nDone.")

if __name__ == "__main__":
    main()
