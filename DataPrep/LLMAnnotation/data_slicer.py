import argparse
import glob
import os
from typing import List, Optional, Tuple

import pandas as pd


THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))


def resolve_xlsx_path(cli_xlsx: Optional[str] = None) -> str:
    if cli_xlsx:
        if not os.path.exists(cli_xlsx):
            raise FileNotFoundError(f"Provided xlsx does not exist: {cli_xlsx}")
        return cli_xlsx

    candidates = [
        os.path.join(PROJECT_ROOT, "DATA", "vlm-workzone_data", "workzone_driving_data.xlsx"),
        os.path.join(PROJECT_ROOT, "DATA", "vlm-work_data", "workzone_driving_data.xlsx"),
        os.path.join(PROJECT_ROOT, "DATA", "vlm-workzone_data", "workzone driving data.xlsx"),
        os.path.join(PROJECT_ROOT, "DATA", "vlm-work_data", "workzone driving data.xlsx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "workzone xlsx not found. Checked: " + "; ".join(candidates)
    )


def parse_timestamp(value) -> Optional[float]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "_" in text:
        text = text.split("_")[-1]
    try:
        return float(text)
    except ValueError:
        return None


def read_workzone_meta(xlsx_path: str, participant_filter: Optional[str]) -> List[dict]:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    metas = []

    for _, row in df.iterrows():
        key = str(row.get("Unnamed: 0", "")).strip()
        if "_" not in key:
            continue
        participant, scenario_code = key.split("_", 1)

        if participant_filter and participant != participant_filter:
            continue

        intervals = [
            (parse_timestamp(row.get("wz1_start")), parse_timestamp(row.get("wz1_end"))),
            (parse_timestamp(row.get("wz2_start")), parse_timestamp(row.get("wz2_end"))),
            (parse_timestamp(row.get("wz3_start_warning")), parse_timestamp(row.get("wz3_end"))),
        ]

        clean_intervals: List[Tuple[float, float]] = []
        for start, end in intervals:
            if start is None or end is None:
                continue
            if end < start:
                continue
            clean_intervals.append((start, end))

        if clean_intervals:
            metas.append(
                {
                    "participant": participant,
                    "scenario_code": scenario_code,
                    "intervals": clean_intervals,
                }
            )

    return metas


def find_gaze_csv(input_dir: str, participant: str, scenario_code: str) -> Optional[str]:
    pattern = os.path.join(
        input_dir, f"{participant}_{scenario_code}_*_merged_gaze_labelled.csv"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        matches.sort()
    return matches[0]


def sample_one_segment(segment_df: pd.DataFrame, ts_col: str, fps: float) -> pd.DataFrame:
    if segment_df.empty:
        return segment_df

    keep_rows = []
    last_kept_ts = None
    step = 1.0 / fps

    for idx, row in segment_df.iterrows():
        ts = float(row[ts_col])
        if last_kept_ts is None or (ts - last_kept_ts) >= step:
            keep_rows.append(idx)
            last_kept_ts = ts

    if not keep_rows:
        return segment_df.iloc[0:0].copy()
    return segment_df.loc[keep_rows].copy()


def slice_one_file(csv_path: str, intervals: List[Tuple[float, float]], fps: float) -> Tuple[int, int, str]:
    df = pd.read_csv(csv_path)
    if "ui_timestamp" not in df.columns:
        raise ValueError(f"ui_timestamp column not found in: {csv_path}")

    work = df.sort_values("ui_timestamp", kind="stable").copy()
    # One row per frame before fps sampling to align with "frame slicing".
    if "ui_image" in work.columns:
        frame_level = work.drop_duplicates(subset=["ui_image"], keep="first")
    else:
        frame_level = work.drop_duplicates(subset=["ui_timestamp"], keep="first")

    sampled_parts = []
    for start, end in intervals:
        seg = frame_level[(frame_level["ui_timestamp"] >= start) & (frame_level["ui_timestamp"] <= end)]
        sampled_parts.append(sample_one_segment(seg, "ui_timestamp", fps))

    if sampled_parts:
        sliced = pd.concat(sampled_parts, ignore_index=False)
        sliced = sliced.sort_values("ui_timestamp", kind="stable")
        if "ui_image" in sliced.columns:
            sliced = sliced.drop_duplicates(subset=["ui_image"], keep="first")
        else:
            sliced = sliced.drop_duplicates(subset=["ui_timestamp"], keep="first")
    else:
        sliced = frame_level.iloc[0:0].copy()

    output_path = csv_path.replace("_merged_gaze_labelled.csv", "_merged_gaze_labelled_sliced.csv")
    sliced.to_csv(output_path, index=False)
    return len(df), len(sliced), output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice gaze labelled CSVs to 3 workzone intervals and sample by fps."
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling fps on frame timeline.")
    parser.add_argument("--xlsx", default=None, help="Optional path to workzone xlsx.")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(
            PROJECT_ROOT,
            "vlm-workzone",
            "DataPrep",
            "GazeTargetAnnotation",
            "gaze_target_output",
        ),
        help="Directory containing *_merged_gaze_labelled.csv files.",
    )
    parser.add_argument(
        "--participant",
        default=None,
        help="Optional participant filter, e.g. P2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    xlsx_path = resolve_xlsx_path(args.xlsx)
    print(f"Reading workzone metadata from: {xlsx_path}")
    metas = read_workzone_meta(xlsx_path, args.participant)
    print(f"Loaded {len(metas)} participant-scenario interval records from xlsx.")

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    processed = 0
    missing = 0
    for meta in metas:
        csv_path = find_gaze_csv(args.input_dir, meta["participant"], meta["scenario_code"])
        tag = f"{meta['participant']}_{meta['scenario_code']}"

        if not csv_path:
            missing += 1
            print(f"[MISS] No gaze csv found for {tag}")
            continue

        original_count, sliced_count, output_path = slice_one_file(
            csv_path, meta["intervals"], args.fps
        )
        processed += 1
        print(
            f"[OK] {os.path.basename(csv_path)}: {original_count} -> {sliced_count} rows, "
            f"saved {os.path.basename(output_path)}"
        )

    print(
        f"Done. processed={processed}, missing={missing}, total_meta={len(metas)}, "
        f"fps={args.fps}"
    )


if __name__ == "__main__":
    main()