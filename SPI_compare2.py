# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Settings (필요시 경로만 수정)
# =========================
BASE_DIR = r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\22th Sinkhole (Ensemble)\python"
OUT_DIR  = os.path.join(BASE_DIR, "SPI_L_evaluation_plots")  # 이전 평가 코드 결과 폴더
METRICS_CSV = os.path.join(OUT_DIR, "SPI_L_metrics_by_event.csv")

# best L 기준 지표(이전 코드에서 만든 컬럼들)
SCORE_COL = "maxSPI_pre14"     # 최고점수 기준
LEAD_COL  = "lead_peak14"      # best L에서 lead time

# 동점 처리: 점수는 큰 값 우선, lead time은 작은 값 우선
# (즉, peak가 사고에 더 가까운 L을 선호)
TIEBREAK_ASC_FOR_LEAD = True

# =========================
# Helpers
# =========================
def pick_best_L_per_event(res: pd.DataFrame) -> pd.DataFrame:
    """event_id 별 best L과 그때의 score/lead를 반환"""
    need_cols = {"event_id", "L", SCORE_COL, LEAD_COL}
    miss = need_cols - set(res.columns)
    if miss:
        raise ValueError(f"Missing columns in metrics csv: {sorted(list(miss))}\n"
                         f"Columns={res.columns.tolist()}")

    out_rows = []
    for eid, sub in res.groupby("event_id"):
        sub = sub.copy()
        sub = sub.dropna(subset=[SCORE_COL])  # score 없으면 평가 불가
        if len(sub) == 0:
            continue

        # lead가 NaN인 경우 tie-break가 애매하므로 큰 값(불리)로 처리
        lead_fill = 1e9
        sub["_lead_for_sort"] = sub[LEAD_COL].copy()
        sub["_lead_for_sort"] = sub["_lead_for_sort"].fillna(lead_fill)

        # 정렬: score 내림차순, lead 오름차순(작을수록 사고에 가까움)
        sub = sub.sort_values(
            by=[SCORE_COL, "_lead_for_sort"],
            ascending=[False, True if TIEBREAK_ASC_FOR_LEAD else False]
        )
        best = sub.iloc[0]

        out_rows.append({
            "event_id": int(eid),
            "best_L": int(best["L"]),
            "best_score": float(best[SCORE_COL]) if pd.notna(best[SCORE_COL]) else np.nan,
            "best_lead": float(best[LEAD_COL]) if pd.notna(best[LEAD_COL]) else np.nan,
        })

    out = pd.DataFrame(out_rows).sort_values("event_id").reset_index(drop=True)
    return out


def plot_bestL_only(best_df: pd.DataFrame, out_png: str):
    """Event별 best L만 표시 (막대)"""
    x = np.arange(len(best_df))
    labels = [f"Event {i}" for i in best_df["event_id"]]
    y = best_df["best_L"].to_numpy()

    plt.figure(figsize=(10, 4.5))
    plt.bar(x, y)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Best L (days)")
    plt.title(f"Best L per event (criterion: {SCORE_COL}, tie-break: smaller {LEAD_COL})")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[Saved]", out_png)


def plot_bestL_with_lead(best_df: pd.DataFrame, out_png: str):
    """
    Event별 best L(막대) + best lead time(점, 2축)
    lead time = 사고 몇 일 전 peak인지 (작을수록 사고에 가까움)
    """
    x = np.arange(len(best_df))
    labels = [f"Event {i}" for i in best_df["event_id"]]
    bestL = best_df["best_L"].to_numpy()
    lead  = best_df["best_lead"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(11, 4.8))

    # 막대: best L
    ax1.bar(x, bestL, alpha=0.9, label="Best L (days)")
    ax1.set_ylabel("Best L (days)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, axis="y")

    # 점: lead time (2축)
    ax2 = ax1.twinx()
    ax2.plot(x, lead, marker="o", linestyle="--", label="Lead time (days)")
    ax2.set_ylabel("Lead time of peak (days before event)")
    # lead time은 작을수록 “가까운 경보”라서, 원하면 축을 뒤집을 수도 있음:
    # ax2.invert_yaxis()

    # 범례 합치기
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.title(f"Per-event Best L and its lead time (score={SCORE_COL})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[Saved]", out_png)


# =========================
# Main
# =========================
def main():
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(
            f"Metrics file not found:\n  {METRICS_CSV}\n"
            f"먼저 SPI_L_metrics_by_event.csv를 생성하는 평가 코드를 실행해 주세요."
        )

    res = pd.read_csv(METRICS_CSV)
    best_df = pick_best_L_per_event(res)

    # best 목록 저장(표로도 남기기)
    best_csv = os.path.join(OUT_DIR, "bestL_and_leadtime_by_event.csv")
    best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")
    print("[Saved]", best_csv)

    # Plot 1: best L only
    plot_bestL_only(best_df, os.path.join(OUT_DIR, "bestL_per_event.png"))

    # Plot 2: best L + lead time
    plot_bestL_with_lead(best_df, os.path.join(OUT_DIR, "bestL_and_leadtime_per_event.png"))

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
