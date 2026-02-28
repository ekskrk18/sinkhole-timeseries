# -*- coding: utf-8 -*-
import os
import time
import json
import requests
import pandas as pd
from datetime import timedelta

# =========================================================
# 1) 사용자 설정
# =========================================================
API_KEY = "d6644697537c53a5bf3d4fb4dfc42a22deb06e7c7ff9729797469525c8aea"

BASE_URL_DAILY = "https://www.gims.go.kr/api/data/observationStationService/getGroundwaterMonitoringNetwork"

EVENTS_CSV = r"E:\20260124\00 KONKUK\02 Papers\01 SCIE\[22th] Sinkhole (Ensemble)\python\sinkhole_events.csv"
OUT_DIR    = r"E:\20260124\00 KONKUK\02 Papers\01 SCIE\[22th] Sinkhole (Ensemble)\python\gims_groundwater_daily"

DAYS_BEFORE = 30
DAYS_AFTER  = 1

SLEEP_SEC   = 0.25
TIMEOUT_SEC = 60

# =========================================================
# 2) 유틸
# =========================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

def extract_result(obj: dict):
    """
    응답의 resultCode/메시지 위치가 다를 수 있어 최대한 넓게 탐색
    """
    if not isinstance(obj, dict):
        return None, None
    resp = obj.get("response")
    if isinstance(resp, dict):
        return resp.get("resultCode"), resp.get("resultMsg")
    return obj.get("resultCode"), obj.get("resultMsg")

def is_success(rc):
    if rc is None:
        return True  # 어떤 응답은 코드가 아예 없기도 해서, 데이터가 있으면 성공으로 보자
    return str(rc).strip().lower() in {"success", "00", "0", "ok", "normal_service", "normalservice"}

def find_result_list(obj: dict):
    """
    일별 서비스 응답에서 list[dict] 형태의 데이터 배열을 찾아 반환
    - 가장 길이가 긴 list[dict]를 선택
    """
    if obj is None:
        return None

    candidates = []

    def walk(x):
        if isinstance(x, list):
            candidates.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(obj)

    best = None
    best_len = 0
    for lst in candidates:
        if len(lst) == 0:
            continue
        if all(isinstance(it, dict) for it in lst):
            if len(lst) > best_len:
                best = lst
                best_len = len(lst)
    return best

def fetch_daily(gennum: str, begindate: str, enddate: str, raw_dir: str) -> pd.DataFrame:
    """
    일별 지하수 측정자료 조회
    """
    params = {
        "KEY": API_KEY,
        "type": "JSON",
        "gennum": str(gennum),
        "begindate": begindate,
        "enddate": enddate,
    }

    r = requests.get(BASE_URL_DAILY, params=params, timeout=TIMEOUT_SEC)
    r.raise_for_status()

    obj = safe_json(r.text)
    if obj is None:
        raise RuntimeError("JSON 파싱 실패(응답이 JSON이 아님).")

    # raw 저장(디버깅)
    raw_path = os.path.join(raw_dir, f"raw_daily_gennum_{gennum}_{begindate}_{enddate}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    rc, rm = extract_result(obj)

    # 응답 코드가 있어도 성공 판정은 넓게
    if rc is not None and (not is_success(rc)):
        raise RuntimeError(f"API 실패: resultCode={rc}, resultMsg={rm}, raw={raw_path}")

    recs = find_result_list(obj)

    if recs is None:
        # 구조를 못 찾으면 전체 평탄화 (그래도 저장은 되게)
        df = pd.json_normalize(obj)
    else:
        df = pd.json_normalize(recs)

    # 메타
    df["gennum"] = str(gennum)
    df["request_begindate"] = begindate
    df["request_enddate"] = enddate
    df["resultCode"] = rc
    df["resultMsg"] = rm
    df["raw_json"] = raw_path
    return df

# =========================================================
# 3) 메인
# =========================================================
def main():
    ensure_dir(OUT_DIR)
    raw_dir = os.path.join(OUT_DIR, "_raw_json")
    ensure_dir(raw_dir)

    ev = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    ev = ev.loc[:, ~ev.columns.astype(str).str.startswith("Unnamed")].copy()

    required = {"id", "event_time", "latitude", "longitude", "gennum"}
    missing = required - set(ev.columns)
    if missing:
        raise ValueError(f"sinkhole_events.csv에 필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {list(ev.columns)}")

    # event_time 파싱
    ev["event_time"] = (
        ev["event_time"].astype(str)
          .str.replace("KST", "", regex=False)
          .str.replace(".", "-", regex=False)
          .str.replace("/", "-", regex=False)
          .str.strip()
    )
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev = ev.dropna(subset=["id", "event_time", "latitude", "longitude", "gennum"]).copy()

    # id 정리
    try:
        ev["id"] = ev["id"].astype(float).astype(int).astype(str)
    except Exception:
        ev["id"] = ev["id"].astype(str)

    summary_rows = []

    for _, row in ev.iterrows():
        event_id = str(row["id"])
        event_dt = row["event_time"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        gennum = str(row["gennum"]).strip()

        start_dt = event_dt - timedelta(days=DAYS_BEFORE)
        end_dt   = event_dt + timedelta(days=DAYS_AFTER)

        beg = start_dt.strftime("%Y%m%d")
        end = end_dt.strftime("%Y%m%d")

        print(f"[Event {event_id}] gennum={gennum} daily {beg}~{end}")

        try:
            df = fetch_daily(gennum, beg, end, raw_dir)

            # 이벤트 메타 추가
            df.insert(0, "event_id", event_id)
            df.insert(1, "event_time", event_dt)
            df.insert(2, "event_latitude", lat)
            df.insert(3, "event_longitude", lon)

            out_path = os.path.join(OUT_DIR, f"event_{event_id}_gennum_{gennum}_daily.csv")
            df.to_csv(out_path, index=False, encoding="utf-8-sig")

            summary_rows.append({
                "event_id": event_id,
                "event_time": str(event_dt),
                "gennum": gennum,
                "begindate": beg,
                "enddate": end,
                "n_rows_downloaded": len(df),
                "output_csv": out_path,
                "raw_json": df["raw_json"].iloc[0] if "raw_json" in df.columns and len(df) > 0 else "",
                "error": ""
            })

        except Exception as e:
            summary_rows.append({
                "event_id": event_id,
                "event_time": str(event_dt),
                "gennum": gennum,
                "begindate": beg,
                "enddate": end,
                "n_rows_downloaded": 0,
                "output_csv": "",
                "raw_json": "",
                "error": str(e)
            })

        time.sleep(SLEEP_SEC)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "download_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nDone. Summary saved: {summary_path}")

if __name__ == "__main__":
    main()
