import argparse
import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_DIR = THIS_DIR / "annotation_outputs" / "workzone_gpt"
WINDOW_NAME = "GPT Annotation Player"


def find_latest_rt_json(json_dir: Path) -> Path:
    candidates = sorted(json_dir.glob("workzone_gpt_rt_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No workzone_gpt_rt_*.json found under: {json_dir}")
    return candidates[-1]


def load_result_json(input_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError(f"No valid 'results' array in: {input_path}")
    return data, results


def safe_imread(path: str, fallback_w: int = 960, fallback_h: int = 540):
    if path and os.path.exists(path):
        frame = cv2.imread(path)
        if frame is not None:
            return frame
    fallback = 40 * (cv2.UMat(fallback_h, fallback_w, cv2.CV_8UC3).get())
    cv2.putText(
        fallback,
        "Image not found",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (70, 180, 255),
        2,
    )
    if path:
        short = path[-100:]
        cv2.putText(
            fallback,
            short,
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )
    return fallback


def fit_image(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def normalize_freedom_text(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def normalize_structured_lines(val: Any) -> List[str]:
    if val is None:
        return ["structured_response: null"]
    if isinstance(val, dict):
        lines = []
        for k, v in val.items():
            lines.append(f"{k}: {v}")
        return lines
    return [f"structured_response: {val}"]


def build_right_panel_lines(record: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    participant = record.get("participant", "?")
    scenario = record.get("scenario", record.get("scenario_prefix", "?"))
    ts = record.get("timestamp", "?")
    gaze = record.get("gaze_target", "?")

    lines.append(f"participant: {participant}")
    lines.append(f"scenario: {scenario}")
    lines.append(f"timestamp: {ts}")
    lines.append(f"gaze_target: {gaze}")
    lines.append("")

    lines.append("[STRUCTURED]")
    lines.extend(normalize_structured_lines(record.get("structured_response")))
    lines.append("")

    lines.append("[FREEDOM]")
    freedom = normalize_freedom_text(record.get("freedom_response"))
    freedom = freedom.replace("\r", " ").replace("\n", " ")
    lines.extend(textwrap.wrap(freedom, width=58))

    return lines


def draw_text_block(canvas, lines: List[str], x: int, y: int, max_w: int):
    header_color = (120, 210, 255)
    text_color = (230, 230, 230)
    muted = (160, 160, 160)

    line_h = 26
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thick = 1

    cy = y
    for line in lines:
        if line == "":
            cy += line_h // 2
            continue

        draw_line = line
        while draw_line:
            # Hard clip for too-long lines by binary search over chars.
            lo, hi = 1, len(draw_line)
            fit = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                part = draw_line[:mid]
                (tw, _), _ = cv2.getTextSize(part, font, scale, thick)
                if tw <= max_w:
                    fit = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            part = draw_line[:fit]
            color = text_color
            if part.startswith("[") and part.endswith("]"):
                color = header_color
            elif part.startswith("structured_response"):
                color = muted

            cv2.putText(canvas, part, (x, cy), font, scale, color, thick, cv2.LINE_AA)
            cy += line_h
            draw_line = draw_line[fit:]



def compose_canvas(record: Dict[str, Any], idx: int, total: int, speed: float, player_fps: float):
    canvas_w = 1680
    canvas_h = 940
    left_w = 1080
    right_w = canvas_w - left_w

    canvas = 18 * (cv2.UMat(canvas_h, canvas_w, cv2.CV_8UC3).get())

    img = safe_imread(record.get("image_path", ""))
    view = fit_image(img, left_w - 30, canvas_h - 110)
    vh, vw = view.shape[:2]
    vx = (left_w - vw) // 2
    vy = 20 + (canvas_h - 120 - vh) // 2
    canvas[vy : vy + vh, vx : vx + vw] = view

    cv2.line(canvas, (left_w, 0), (left_w, canvas_h), (70, 70, 70), 2)

    title = f"Frame {idx + 1}/{total} | Player FPS: {player_fps:.2f} | Speed: {speed:.2f}x"
    help_text = "Space Pause/Play | Left/Right Prev/Next | +/- Speed | q Quit"
    cv2.putText(canvas, title, (24, canvas_h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (220, 220, 220), 2)
    cv2.putText(canvas, help_text, (24, canvas_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1)

    lines = build_right_panel_lines(record)
    draw_text_block(canvas, lines, left_w + 22, 42, right_w - 40)

    return canvas


def play_records(results: List[Dict[str, Any]], player_fps: float, start_speed: float):
    total = len(results)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1720, 980)

    state = {
        "idx": 0,
        "playing": True,
        "speed": max(0.1, start_speed),
        "dragging": False,
        "last_tick": time.perf_counter(),
    }

    def on_trackbar(val: int):
        state["idx"] = max(0, min(total - 1, val))
        state["dragging"] = True

    cv2.createTrackbar("Frame", WINDOW_NAME, 0, total - 1, on_trackbar)

    while True:
        record = results[state["idx"]]
        canvas = compose_canvas(
            record=record,
            idx=state["idx"],
            total=total,
            speed=state["speed"],
            player_fps=player_fps,
        )
        cv2.imshow(WINDOW_NAME, canvas)

        now = time.perf_counter()
        elapsed = now - state["last_tick"]
        state["last_tick"] = now

        if state["playing"] and not state["dragging"]:
            step = max(1, int(elapsed * player_fps * state["speed"]))
            state["idx"] += step
            if state["idx"] >= total:
                state["idx"] = total - 1
                state["playing"] = False
            cv2.setTrackbarPos("Frame", WINDOW_NAME, state["idx"])
        else:
            state["dragging"] = False

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            state["playing"] = not state["playing"]
        elif key in (ord("+"), ord("=")):
            state["speed"] = min(16.0, state["speed"] * 2.0)
        elif key in (ord("-"), ord("_")):
            state["speed"] = max(0.1, state["speed"] / 2.0)
        elif key == 81:  # Left arrow
            state["idx"] = max(0, state["idx"] - 1)
            cv2.setTrackbarPos("Frame", WINDOW_NAME, state["idx"])
        elif key == 83:  # Right arrow
            state["idx"] = min(total - 1, state["idx"] + 1)
            cv2.setTrackbarPos("Frame", WINDOW_NAME, state["idx"])

    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Play GPT annotation JSON as a slider player. Left panel shows image, "
            "right panel lists GPT responses line by line."
        )
    )
    p.add_argument(
        "--input",
        default=None,
        help="Path to workzone_gpt_rt_*.json (default: latest under annotation_outputs/workzone_gpt)",
    )
    p.add_argument(
        "--json-dir",
        default=str(DEFAULT_JSON_DIR),
        help="Directory to search latest run JSON when --input is omitted",
    )
    p.add_argument("--player-fps", type=float, default=4.0, help="Base player fps")
    p.add_argument("--start-speed", type=float, default=1.0, help="Initial playback speed")
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Only print parsed info and exit (no GUI)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.input:
        input_path = Path(args.input).resolve()
    else:
        input_path = find_latest_rt_json(Path(args.json_dir).resolve())

    data, results = load_result_json(input_path)

    print(f"[INFO] input json: {input_path}")
    print(f"[INFO] model: {data.get('model', 'unknown')}")
    print(f"[INFO] run_ts: {data.get('run_ts', 'unknown')}")
    print(f"[INFO] total records: {len(results)}")

    if args.print_only:
        sample = results[0]
        print("[INFO] sample image_path:", sample.get("image_path", ""))
        print("[INFO] sample gaze_target:", sample.get("gaze_target", ""))
        return

    play_records(
        results=results,
        player_fps=max(0.5, min(60.0, args.player_fps)),
        start_speed=args.start_speed,
    )


if __name__ == "__main__":
    main()
