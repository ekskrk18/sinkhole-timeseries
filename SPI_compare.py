# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
BASE_DIR   = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python"
GW_DIR     = os.path.join(BASE_DIR, "gims_groundwater_daily")
RAIN_DIR   = os.path.join(BASE_DIR, "kma_rn60m")
SMAP_DIR   = os.path.join(BASE_DIR, "SMAP_sinkhole_rootzone_30d")
EVENTS_CSV = os.path.join(BASE_DIR, "sinkhole_events.csv")

OUT_DIR    = os.path.join(BASE_DIR, "SPI_L_evaluation_plots")
os.makedirs(OUT_DIR, exist_ok=True)

L_LIST = [1, 3, 7, 10, 14]
EVENT_ID_LIST = [1,2,3,4,5,6,7,8]  # 필요시 수정

# 평가 창(논문에서 명시)
EVAL_PRE_DAYS = 30   # 사고 전 30일
EVAL_POST_DAYS = 1   # 사고 후 1일 (그래프 범위용)
PEAK_SEARCH_WINDOW_DAYS = 14  # "사고 직전" peak는 -14~0에서 찾기 추천


# =========================
# Utilities
# =========================
def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file:\n  {path}")
    return pd.read_csv(path)

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def to_naive_date(dt_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_series, errors="coerce")
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    return s.dt.floor("D")

def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    return (x - med) / mad

def choose_best_shift(base_dates: pd.Series, df: pd.DataFrame, date_col="date", shifts=(-1, 0, 1)):
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


# =========================
# Load events
# =========================
def load_events_table(path: str) -> pd.DataFrame:
    ev = _norm_cols(_read_csv(path))

    # event_id / gennum
    if "event_id" not in ev.columns:
        for alt in ["id", "event", "case", "no"]:
            if alt in ev.columns:
                ev = ev.rename(columns={alt: "event_id"})
                break
    if "event_id" not in ev.columns or "gennum" not in ev.columns:
        raise ValueError(f"[EVENTS] need event_id & gennum. Columns={ev.columns.tolist()}")

    # event_time column 후보
    time_candidates = ["event_time_kst", "event_time", "event_time_utc", "time_kst", "time"]
    time_col = next((c for c in time_candidates if c in ev.columns), None)
    ev["event_time_dt"] = pd.to_datetime(ev[time_col], errors="coerce") if time_col else pd.NaT

    ev["event_id"] = pd.to_numeric(ev["event_id"], errors="coerce").astype("Int64")
    ev["gennum"]   = pd.to_numeric(ev["gennum"], errors="coerce").astype("Int64")

    ev = ev.dropna(subset=["event_id","gennum"]).sort_values("event_id")
    return ev


# =========================
# Load daily series
# =========================
def load_gw_daily(gw_path: str) -> pd.DataFrame:
    gw = _norm_cols(_read_csv(gw_path))
    if "ymd" not in gw.columns:
        raise ValueError(f"[GW] ymd missing. Columns={gw.columns.tolist()}")
    dt = pd.to_datetime(gw["ymd"].astype(str), format="%Y%m%d", errors="coerce")
    gw["date"] = to_naive_date(dt)

    if "lev" not in gw.columns and "elev" in gw.columns:
        gw["lev"] = gw["elev"]

    for c in ["lev","ec"]:
        if c not in gw.columns:
            raise ValueError(f"[GW] {c} missing. Columns={gw.columns.tolist()}")
        gw[c] = pd.to_numeric(gw[c], errors="coerce")

    gw = gw[["date","lev","ec"]].groupby("date", as_index=False).mean(numeric_only=True)
    return gw

def load_smap_daily(sm_path: str) -> pd.DataFrame:
    sm = _norm_cols(_read_csv(sm_path))
    if "obs_time_kst" in sm.columns:
        tcol = "obs_time_kst"
    elif "obs_time_utc" in sm.columns:
        tcol = "obs_time_utc"
    else:
        raise ValueError(f"[SMAP] no obs_time. Columns={sm.columns.tolist()}")

    dt = pd.to_datetime(sm[tcol], errors="coerce")
    sm["date"] = to_naive_date(dt)

    if "sm_rootzone" not in sm.columns:
        raise ValueError(f"[SMAP] sm_rootzone missing. Columns={sm.columns.tolist()}")
    sm["sm_rootzone"] = pd.to_numeric(sm["sm_rootzone"], errors="coerce")

    sm = sm[["date","sm_rootzone"]].groupby("date", as_index=False).mean()
    return sm

def load_rain_daily(rain_path: str) -> pd.DataFrame:
    rn = _norm_cols(_read_csv(rain_path))
    if "rn_60m" not in rn.columns:
        for alt in ["rain","rn","precip","prcp"]:
            if alt in rn.columns:
                rn = rn.rename(columns={alt:"rn_60m"})
                break
    if "rn_60m" not in rn.columns:
        raise ValueError(f"[RAIN] rn_60m missing. Columns={rn.columns.tolist()}")

    rn["rn_60m"] = pd.to_numeric(rn["rn_60m"], errors="coerce")

    if "date" in rn.columns:
        dt = pd.to_datetime(rn["date"], errors="coerce")
    else:
        dt = None
        for alt in ["datetime","time","tm","timestamp","obs_time","obs_time_kst","obs_time_utc"]:
            if alt in rn.columns:
                dt = pd.to_datetime(rn[alt], errors="coerce")
                break
        if dt is None:
            raise ValueError(f"[RAIN] no time column. Columns={rn.columns.tolist()}")

    rn["date"] = to_naive_date(dt)
    rn = rn[["date","rn_60m"]].groupby("date", as_index=False).sum()
    return rn


def build_window(event_date: pd.Timestamp) -> pd.DataFrame:
    start = (event_date - pd.Timedelta(days=EVAL_PRE_DAYS)).normalize()
    end   = (event_date + pd.Timedelta(days=EVAL_POST_DAYS)).normalize()
    return pd.DataFrame({"date": pd.date_range(start, end, freq="D")})

def infer_event_date_from_gw(gw: pd.DataFrame) -> pd.Timestamp:
    # 보통 자료가 -30~+1이면 마지막날-1이 event day
    return (gw["date"].max() - pd.Timedelta(days=1)).normalize()

def load_event_merged(event_id: int, gennum: int, event_time_dt) -> tuple[pd.DataFrame, pd.Timestamp]:
    gw_path   = os.path.join(GW_DIR,   f"event_{event_id}_gennum_{int(gennum)}_daily.csv")
    rain_path = os.path.join(RAIN_DIR, f"event_{event_id}_rn60m.csv")
    sm_path   = os.path.join(SMAP_DIR, f"event_{event_id}", f"event_{event_id}_rootzone_30d_to_1d.csv")

    gw = load_gw_daily(gw_path)
    sm = load_smap_daily(sm_path)
    rn = load_rain_daily(rain_path)

    if pd.notna(event_time_dt):
        event_date = to_naive_date(pd.Series([event_time_dt])).iloc[0]
    else:
        event_date = infer_event_date_from_gw(gw)

    backbone = build_window(event_date)

    sm2, _, _ = choose_best_shift(backbone["date"], sm, shifts=(-1,0,1))
    rn2, _, _ = choose_best_shift(backbone["date"], rn, shifts=(-1,0,1))

    df = backbone.merge(gw, on="date", how="left").merge(sm2, on="date", how="left").merge(rn2, on="date", how="left")

    df["rn_60m"] = df["rn_60m"].fillna(0.0)
    for col in ["lev","ec","sm_rootzone"]:
        df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    return df, event_date


# =========================
# SPI definition (same as before)
# =========================
def compute_spi(df: pd.DataFrame, L: int,
                w_DI=0.25, w_RI=0.20, w_CI=0.25, w_SI=0.20, w_HI=0.10,
                alpha_rain=0.01) -> pd.DataFrame:
    d = df.copy()
    d["dlev"] = d["lev"] - d["lev"].shift(L)
    d["dec"]  = d["ec"]  - d["ec"].shift(L)
    d["dsm"]  = d["sm_rootzone"] - d["sm_rootzone"].shift(L)
    d["rain_L"] = d["rn_60m"].rolling(L, min_periods=L).sum()

    rain_thr = d["rain_L"].quantile(0.25)
    d["NR"] = (d["rain_L"] <= rain_thr).astype(int)

    d["DI"] = np.maximum(0.0, -d["dlev"])
    d["RI"] = np.maximum(0.0,  d["dlev"]) * d["NR"]
    d["CI"] = np.maximum(0.0,  d["dec"])  * d["NR"]
    d["SI"] = np.maximum(0.0,  d["dsm"]) + alpha_rain * d["rain_L"]
    d["HI"] = np.maximum(0.0,  d["dsm"]) * d["NR"]

    for col in ["DI","RI","CI","SI","HI"]:
        d[col+"_z"] = robust_z(d[col].to_numpy())

    d["SPI"] = (w_DI*d["DI_z"] + w_RI*d["RI_z"] + w_CI*d["CI_z"] + w_SI*d["SI_z"] + w_HI*d["HI_z"])
    return d


# =========================
# Metrics for comparing L
# =========================
def summarize_L_per_event(d_spi: pd.DataFrame, event_date: pd.Timestamp) -> dict:
    """
    사고 전(-30~0) / 직전(-14~0) 등에서:
    - maxSPI_pre14: 직전 14일 max SPI
    - maxSPI_pre30: 전체 30일 max SPI
    - lead_peak14: 직전 14일 peak가 사고 몇 일 전인지 (0=사고 당일)
    - spi_event: 사고일 SPI
    """
    d = d_spi.copy()
    d["relday"] = (d["date"] - event_date).dt.days  # event=0, pre=-1,-2...

    pre30 = d[(d["relday"] >= -EVAL_PRE_DAYS) & (d["relday"] <= 0)].dropna(subset=["SPI"])
    pre14 = d[(d["relday"] >= -PEAK_SEARCH_WINDOW_DAYS) & (d["relday"] <= 0)].dropna(subset=["SPI"])

    out = {"maxSPI_pre14": np.nan, "maxSPI_pre30": np.nan, "lead_peak14": np.nan, "spi_event": np.nan}
    if len(pre30) == 0:
        return out

    out["maxSPI_pre30"] = float(pre30["SPI"].max())

    if len(pre14) > 0:
        idx = pre14["SPI"].idxmax()
        peak_day = pre14.loc[idx, "date"]
        out["maxSPI_pre14"] = float(pre14.loc[idx, "SPI"])
        out["lead_peak14"] = int((event_date - peak_day).days)  # 0~14

    spi_event = pre30[pre30["relday"] == 0]["SPI"]
    if len(spi_event) > 0:
        out["spi_event"] = float(spi_event.iloc[0])

    return out


# =========================
# Plotting: Heatmap + Boxplots + Winner bar
# =========================
def plot_heatmap(matrix: np.ndarray, row_labels, col_labels, title, out_png, vmin=None, vmax=None):
    plt.figure(figsize=(9, 4.5))
    im = plt.imshow(matrix, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.xlabel("L (days)")
    plt.ylabel("Event")

    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=12)

    # cell text
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[Saved]", out_png)

def plot_boxplot(data_by_L: dict, title, ylabel, out_png):
    Ls = list(data_by_L.keys())
    data = [data_by_L[L] for L in Ls]

    plt.figure(figsize=(7.5, 4.5))
    plt.boxplot(data, labels=[str(L) for L in Ls], showfliers=True)
    plt.title(title)
    plt.xlabel("L (days)")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[Saved]", out_png)

def plot_winner_counts(winners: list, title, out_png):
    # winners: list of best L per event
    uniq = sorted(list(set(winners)))
    counts = [winners.count(u) for u in uniq]

    plt.figure(figsize=(7.0, 4.2))
    plt.bar([str(u) for u in uniq], counts)
    plt.title(title)
    plt.xlabel("Best L (days)")
    plt.ylabel("Count (events)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[Saved]", out_png)


# =========================
# Main evaluation
# =========================
def main():
    events = load_events_table(EVENTS_CSV)
    events = events[events["event_id"].isin(EVENT_ID_LIST)].copy()

    # 저장용: event x L metrics
    rows = []
    for _, r in events.iterrows():
        eid = int(r["event_id"])
        gennum = int(r["gennum"])
        etime = r.get("event_time_dt", pd.NaT)

        df, event_date = load_event_merged(eid, gennum, etime)

        for L in L_LIST:
            d_spi = compute_spi(df, L)
            m = summarize_L_per_event(d_spi, event_date)
            rows.append({
                "event_id": eid,
                "L": L,
                **m
            })

    res = pd.DataFrame(rows).sort_values(["event_id","L"])
    res_path = os.path.join(OUT_DIR, "SPI_L_metrics_by_event.csv")
    res.to_csv(res_path, index=False, encoding="utf-8-sig")
    print("[Saved]", res_path)

    # ---- 1) Heatmap: maxSPI_pre14
    event_labels = [f"Event {e}" for e in sorted(res["event_id"].unique())]
    L_labels = [str(L) for L in L_LIST]

    def make_matrix(metric: str):
        mat = np.full((len(event_labels), len(L_LIST)), np.nan)
        for i, eid in enumerate(sorted(res["event_id"].unique())):
            sub = res[res["event_id"] == eid].set_index("L")
            for j, L in enumerate(L_LIST):
                mat[i, j] = sub.loc[L, metric] if L in sub.index else np.nan
        return mat

    mat_max14 = make_matrix("maxSPI_pre14")
    plot_heatmap(
        mat_max14, event_labels, L_labels,
        title=f"Max SPI within {-PEAK_SEARCH_WINDOW_DAYS}~0 days (higher = stronger pre-signal)",
        out_png=os.path.join(OUT_DIR, "heatmap_maxSPI_pre14.png")
    )

    # ---- 2) Heatmap: lead time of peak (smaller is closer to event)
    mat_lead = make_matrix("lead_peak14")
    plot_heatmap(
        mat_lead, event_labels, L_labels,
        title=f"Lead time of peak SPI (days before event; smaller = closer)",
        out_png=os.path.join(OUT_DIR, "heatmap_leadtime_peak_pre14.png"),
        vmin=0, vmax=PEAK_SEARCH_WINDOW_DAYS
    )

    # ---- 3) Boxplot: maxSPI_pre14 by L
    data_max14 = {L: res[res["L"]==L]["maxSPI_pre14"].dropna().tolist() for L in L_LIST}
    plot_boxplot(
        data_max14,
        title=f"Distribution of max SPI within {-PEAK_SEARCH_WINDOW_DAYS}~0 days (across events)",
        ylabel="maxSPI_pre14",
        out_png=os.path.join(OUT_DIR, "boxplot_maxSPI_pre14.png")
    )

    # ---- 4) Boxplot: lead time by L
    data_lead = {L: res[res["L"]==L]["lead_peak14"].dropna().tolist() for L in L_LIST}
    plot_boxplot(
        data_lead,
        title=f"Distribution of peak lead time within {-PEAK_SEARCH_WINDOW_DAYS}~0 days (across events)",
        ylabel="lead_peak14 (days)",
        out_png=os.path.join(OUT_DIR, "boxplot_leadtime_peak_pre14.png")
    )

    # ---- 5) Winner counts (best L)
    # 기준: maxSPI_pre14가 가장 큰 L을 "best"로 (동률이면 lead time 더 작은 쪽 우선)
    winners = []
    for eid in sorted(res["event_id"].unique()):
        sub = res[res["event_id"] == eid].copy()
        sub = sub.dropna(subset=["maxSPI_pre14"])
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["maxSPI_pre14","lead_peak14"], ascending=[False, True])
        winners.append(int(sub.iloc[0]["L"]))

    if len(winners) > 0:
        plot_winner_counts(
            winners,
            title="Best L frequency across events (criterion: highest maxSPI_pre14, tie-break: smaller lead time)",
            out_png=os.path.join(OUT_DIR, "winner_L_counts.png")
        )

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
