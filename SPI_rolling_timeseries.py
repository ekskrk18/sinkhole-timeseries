# -*- coding: utf-8 -*-
r"""
SPI time-series plots for Event 1–8 (overlay L = 1, 3, 7, 10, 14)

Fixes:
- Avoid empty plots by NOT using inner-merge across GW/SMAP/RAIN.
- Use GW daily date as backbone, left-join SMAP & RAIN.
- Auto-select best day-shift (-1,0,+1) for SMAP and RAIN to maximize overlap.
- Force x-axis to [event_date-30d, event_date+1d].
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) PATH SETTINGS
# =========================
BASE_DIR  = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python"
GW_DIR    = os.path.join(BASE_DIR, "gims_groundwater_daily")
RAIN_DIR  = os.path.join(BASE_DIR, "kma_rn60m")
SMAP_DIR  = os.path.join(BASE_DIR, "SMAP_sinkhole_rootzone_30d")
OUT_DIR   = os.path.join(BASE_DIR, "SPI_plots_by_event")
EVENTS_CSV = os.path.join(BASE_DIR, "sinkhole_events.csv")

L_LIST = (1, 3, 7, 10, 14)
EVENT_ID_LIST = [1, 2, 3, 4, 5, 6, 7, 8]  # 필요하면 None으로


# =========================
# 1) Robust z-score
# =========================
def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    return (x - med) / mad


# =========================
# 2) SPI core
# =========================
def compute_spi(df: pd.DataFrame, L: int,
                w_DI=0.25, w_RI=0.20, w_CI=0.25, w_SI=0.20, w_HI=0.10,
                alpha_rain=0.01) -> pd.DataFrame:
    d = df.copy()

    # 변화량(shift 기반)
    d["dlev"] = d["lev"] - d["lev"].shift(L)
    d["dec"]  = d["ec"]  - d["ec"].shift(L)
    d["dsm"]  = d["sm_rootzone"] - d["sm_rootzone"].shift(L)

    # L일 누적 강우
    d["rain_L"] = d["rn_60m"].rolling(L, min_periods=L).sum()

    # "비가 거의 안 온 조건" (사분위 25% 이하를 무강우로 간주)
    rain_thr = d["rain_L"].quantile(0.25)
    d["NR"] = (d["rain_L"] <= rain_thr).astype(int)

    # 지표 구성 (당신이 말한 물리적 해석을 반영한 형태)
    d["DI"] = np.maximum(0.0, -d["dlev"])                 # 급격한 수위 저하(굴착/배수 등)
    d["RI"] = np.maximum(0.0,  d["dlev"]) * d["NR"]       # 무강우 상태의 수위 상승(유입/관로 누수 등)
    d["CI"] = np.maximum(0.0,  d["dec"])  * d["NR"]       # 무강우 상태의 EC 증가(오수/하수 누수 등)
    d["SI"] = np.maximum(0.0,  d["dsm"]) + alpha_rain * d["rain_L"]  # 토양수분 증가(강우+침투)
    d["HI"] = np.maximum(0.0,  d["dsm"]) * d["NR"]        # 무강우 상태의 토양수분 증가(상수관/지하수 유입 등)

    # Robust 표준화
    for col in ["DI", "RI", "CI", "SI", "HI"]:
        d[col + "_z"] = robust_z(d[col].to_numpy())

    # 가중합 SPI
    d["SPI"] = (
        w_DI * d["DI_z"] +
        w_RI * d["RI_z"] +
        w_CI * d["CI_z"] +
        w_SI * d["SI_z"] +
        w_HI * d["HI_z"]
    )
    return d


# =========================
# 3) Helpers
# =========================
def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file:\n  {path}")
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def to_naive_date(dt_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_series, errors="coerce")
    # tz-aware면 tz 제거
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    return s.dt.floor("D")


def load_events_table(path: str) -> pd.DataFrame:
    ev = _normalize_columns(_read_csv(path))

    if "event_id" not in ev.columns:
        for alt in ["id", "event", "case", "no"]:
            if alt in ev.columns:
                ev = ev.rename(columns={alt: "event_id"})
                break

    if "event_id" not in ev.columns:
        raise ValueError(f"[EVENTS] event_id not found. Columns: {ev.columns.tolist()}")
    if "gennum" not in ev.columns:
        raise ValueError(f"[EVENTS] gennum not found. Columns: {ev.columns.tolist()}")

    ev["event_id"] = pd.to_numeric(ev["event_id"], errors="coerce").astype("Int64")
    ev["gennum"]   = pd.to_numeric(ev["gennum"], errors="coerce").astype("Int64")

    # event time(가능하면 사용)
    time_candidates = ["event_time_kst", "event_time", "event_time_utc", "time_kst", "time"]
    time_col = next((c for c in time_candidates if c in ev.columns), None)
    ev["event_time_dt"] = pd.to_datetime(ev[time_col], errors="coerce") if time_col else pd.NaT

    return ev


# =========================
# 4) Load GW / SMAP / RAIN to daily
# =========================
def load_gw_daily(gw_path: str) -> pd.DataFrame:
    gw = _normalize_columns(_read_csv(gw_path))

    if "ymd" not in gw.columns:
        raise ValueError(f"[GW] 'ymd' not found. Columns: {gw.columns.tolist()}")

    dt = pd.to_datetime(gw["ymd"].astype(str), format="%Y%m%d", errors="coerce")
    gw["date"] = to_naive_date(dt)

    if "lev" not in gw.columns and "elev" in gw.columns:
        gw["lev"] = gw["elev"]

    for c in ["lev", "ec"]:
        if c not in gw.columns:
            raise ValueError(f"[GW] Missing '{c}'. Columns: {gw.columns.tolist()}")
        gw[c] = pd.to_numeric(gw[c], errors="coerce")

    gw = gw[["date", "lev", "ec"]].groupby("date", as_index=False).mean(numeric_only=True)
    return gw


def load_smap_daily(sm_path: str) -> pd.DataFrame:
    sm = _normalize_columns(_read_csv(sm_path))

    if "obs_time_kst" in sm.columns:
        tcol = "obs_time_kst"
    elif "obs_time_utc" in sm.columns:
        tcol = "obs_time_utc"
    else:
        raise ValueError(f"[SMAP] No usable time column. Columns: {sm.columns.tolist()}")

    dt = pd.to_datetime(sm[tcol], errors="coerce")
    sm["date"] = to_naive_date(dt)

    if "sm_rootzone" not in sm.columns:
        raise ValueError(f"[SMAP] 'sm_rootzone' not found. Columns: {sm.columns.tolist()}")

    sm["sm_rootzone"] = pd.to_numeric(sm["sm_rootzone"], errors="coerce")
    sm = sm[["date", "sm_rootzone"]].groupby("date", as_index=False).mean()
    return sm


def load_rain_daily(rain_path: str) -> pd.DataFrame:
    rn = _normalize_columns(_read_csv(rain_path))

    if "rn_60m" not in rn.columns:
        for alt in ["rain", "rn", "precip", "prcp"]:
            if alt in rn.columns:
                rn = rn.rename(columns={alt: "rn_60m"})
                break
    if "rn_60m" not in rn.columns:
        raise ValueError(f"[RAIN] rn_60m not found. Columns: {rn.columns.tolist()}")

    rn["rn_60m"] = pd.to_numeric(rn["rn_60m"], errors="coerce")

    # 시간 컬럼 찾기
    if "date" in rn.columns:
        dt = pd.to_datetime(rn["date"], errors="coerce")
    else:
        dt = None
        for alt in ["datetime", "time", "tm", "timestamp", "obs_time", "obs_time_kst", "obs_time_utc"]:
            if alt in rn.columns:
                dt = pd.to_datetime(rn[alt], errors="coerce")
                break
        if dt is None:
            raise ValueError(f"[RAIN] No usable time column. Columns: {rn.columns.tolist()}")

    rn["date"] = to_naive_date(dt)
    rn = rn[["date", "rn_60m"]].groupby("date", as_index=False).sum()
    return rn


# =========================
# 5) Best-shift join (prevents empty merges)
# =========================
def choose_best_shift(base_dates: pd.Series, df: pd.DataFrame, date_col="date", shifts=(-1, 0, 1)):
    """
    df[date]를 -1/0/+1일 shift 해보면서 base_dates와 겹치는 날짜 수가 최대가 되는 shift 선택.
    """
    base_set = set(pd.to_datetime(base_dates).tolist())
    best_shift = 0
    best_overlap = -1
    best_df = df.copy()

    for s in shifts:
        tmp = df.copy()
        tmp[date_col] = tmp[date_col] + pd.Timedelta(days=s)
        overlap = len(set(tmp[date_col].tolist()) & base_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_shift = s
            best_df = tmp

    return best_df, best_shift, best_overlap


def build_backbone_window(event_date: pd.Timestamp) -> pd.DataFrame:
    start = (event_date - pd.Timedelta(days=30)).normalize()
    end   = (event_date + pd.Timedelta(days=1)).normalize()
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates})


def infer_event_date_from_gw(gw: pd.DataFrame) -> pd.Timestamp:
    # 보통 데이터가 30d~+1d로 잘려있으면 마지막날-1이 event day
    return (gw["date"].max() - pd.Timedelta(days=1)).normalize()


def load_event_merged_windowed(event_id: int, gennum: int, event_time_dt) -> tuple[pd.DataFrame, pd.Timestamp]:
    # file paths
    gw_filename   = f"event_{event_id}_gennum_{int(gennum)}_daily.csv"
    rain_filename = f"event_{event_id}_rn60m.csv"
    sm_filename   = f"event_{event_id}_rootzone_30d_to_1d.csv"

    gw_path   = os.path.join(GW_DIR, gw_filename)
    rain_path = os.path.join(RAIN_DIR, rain_filename)
    sm_path   = os.path.join(SMAP_DIR, f"event_{event_id}", sm_filename)

    gw = load_gw_daily(gw_path)
    sm = load_smap_daily(sm_path)
    rn = load_rain_daily(rain_path)

    # event date 결정
    if pd.notna(event_time_dt):
        event_date = to_naive_date(pd.Series([event_time_dt])).iloc[0]
    else:
        event_date = infer_event_date_from_gw(gw)

    # backbone window 생성(30d~+1d)
    backbone = build_backbone_window(event_date)

    # shift 자동 선택(겹침 최대화)
    sm2, sm_shift, sm_ov = choose_best_shift(backbone["date"], sm, shifts=(-1, 0, 1))
    rn2, rn_shift, rn_ov = choose_best_shift(backbone["date"], rn, shifts=(-1, 0, 1))

    # backbone + left join (절대 inner merge 안 함)
    df = backbone.merge(gw, on="date", how="left").merge(sm2, on="date", how="left").merge(rn2, on="date", how="left")

    # 강우는 결측이면 0이 합리적(비가 안 온 경우 포함)
    df["rn_60m"] = df["rn_60m"].fillna(0.0)

    # sm_rootzone, lev, ec는 결측이 있으면 SPI 계산이 깨지므로
    # - 너무 많이 비면 plot이 빈 선이 되니, 최소한의 보간/전방채움 선택:
    #   (일단 선이 보이게 하려면 fill이 필요. 연구용으로는 이후 민감도 분석 권장)
    for col in ["sm_rootzone", "lev", "ec"]:
        df[col] = df[col].interpolate(limit_direction="both")

    meta = {
        "event_date": event_date,
        "sm_shift": sm_shift, "sm_overlap": sm_ov,
        "rn_shift": rn_shift, "rn_overlap": rn_ov,
    }
    return df, meta


# =========================
# 6) Plot
# =========================
def plot_spi_event(event_id: int, gennum: int, event_time_dt, debug: bool = True) -> None:
    df, meta = load_event_merged_windowed(event_id, gennum, event_time_dt)
    event_date = meta["event_date"]

    if debug:
        print(f"\n[Event {event_id}] gennum={gennum} | window={df['date'].min().date()}~{df['date'].max().date()} | "
              f"SMAP shift={meta['sm_shift']}d (overlap={meta['sm_overlap']}) | "
              f"RAIN shift={meta['rn_shift']}d (overlap={meta['rn_overlap']}) | "
              f"n={len(df)}")

    plt.figure(figsize=(13, 5))
    any_plotted = False

    for L in L_LIST:
        d_spi = compute_spi(df, L)

        # SPI 계산 후 NaN만 남을 수도 있으니 dropna해서 실제 plot
        plot_df = d_spi.dropna(subset=["SPI"])
        if len(plot_df) > 0:
            plt.plot(plot_df["date"], plot_df["SPI"], label=f"L={L}d")
            any_plotted = True

    plt.axvline(event_date, color="red", linestyle=":", linewidth=2, label="Event time")

    # x축을 반드시 30d~+1d로 고정
    xmin = (event_date - pd.Timedelta(days=30)).normalize()
    xmax = (event_date + pd.Timedelta(days=1)).normalize()
    plt.xlim(xmin, xmax)

    plt.title(f"Event {event_id} – SPI time series (L={list(L_LIST)})")
    plt.xlabel("Date")
    plt.ylabel("SPI")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"event_{event_id:02d}_SPI_overlay.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    if not any_plotted:
        print(f"[WARN] Event {event_id}: SPI가 전부 NaN이라 선이 안 그려졌습니다. (데이터 결측/상수 여부 점검 필요)")
    print(f"[Saved] {out_path}")


# =========================
# 7) Main
# =========================
def main():
    print("BASE_DIR  :", BASE_DIR)
    print("EVENTS_CSV:", EVENTS_CSV)
    print("GW_DIR    :", GW_DIR)
    print("RAIN_DIR  :", RAIN_DIR)
    print("SMAP_DIR  :", SMAP_DIR)
    print("OUT_DIR   :", OUT_DIR)

    events = load_events_table(EVENTS_CSV)
    if EVENT_ID_LIST is not None:
        events = events[events["event_id"].isin(EVENT_ID_LIST)].copy()

    events = events.dropna(subset=["event_id", "gennum"]).sort_values("event_id")

    for _, row in events.iterrows():
        eid = int(row["event_id"])
        gennum = int(row["gennum"])
        etime = row.get("event_time_dt", pd.NaT)
        try:
            plot_spi_event(eid, gennum, etime, debug=True)
        except Exception as ex:
            print(f"\n[FAILED] Event {eid} (gennum={gennum}):\n{ex}\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
