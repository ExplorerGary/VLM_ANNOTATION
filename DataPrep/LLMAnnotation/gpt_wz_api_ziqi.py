# Workzone-aware single-frame annotation pipeline — OpenAI GPT (Batch API)
#
# New pipeline:
# - Resolve merged_ui frames via WhereIsData
# - Read gaze labels from *_merged_gaze_labelled.csv
# - Inject gaze_target into PromptImproved user message
# - Support P*/S* workflow for build / submit / collect / run

import argparse
import base64
import csv
import importlib.util
import json
import os
import re
import sys
import time
from bisect import bisect_left
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

GAZE_ANNOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "GazeTargetAnnotation"))
if GAZE_ANNOT_DIR not in sys.path:
    sys.path.insert(0, GAZE_ANNOT_DIR)

from PromptImproved import PromptImproved as LLMPrompt

_WHERE_IS_DATA_PATH = os.path.join(GAZE_ANNOT_DIR, "where_is_data.py")
_spec = importlib.util.spec_from_file_location("where_is_data", _WHERE_IS_DATA_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load where_is_data module from {_WHERE_IS_DATA_PATH}")
_where_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_where_mod)
WhereIsData = _where_mod.WhereIsData
DataLocationError = _where_mod.DataLocationError

# ---------------------------------------------------------------------------
# Defaults (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "annotation_outputs", "workzone_gpt")
DEFAULT_MODEL = "gpt-4o-mini"
IMAGE_SIZE = 1024
DEFAULT_GAZE_CSV_DIR = os.path.join(GAZE_ANNOT_DIR, "gaze_target_output")
DEFAULT_TIMESTAMP_TOLERANCE = 0.20


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- build ----
    b = sub.add_parser("build", help="Build batch request .jsonl (no API calls)")
    b.add_argument("-p", "--participant", required=True, help="Participant, e.g. P2 or 2")
    b.add_argument("-s", "--scenario", required=True, help="Scenario prefix/full, e.g. S1 or S1_normal")
    b.add_argument("--data-root", default=None, help="Dataset root for WhereIsData")
    b.add_argument("--gaze-csv-dir", default=DEFAULT_GAZE_CSV_DIR)
    b.add_argument("--gaze-csv-file", default=None, help="Explicit gaze CSV path (supports sliced CSV)")
    b.add_argument("--prefer-sliced", action="store_true", help="Prefer *_merged_gaze_labelled_sliced.csv when auto matching")
    b.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    b.add_argument("--max-tokens", type=int, default=512)
    b.add_argument("--detail", default="high", choices=["low", "high"])
    b.add_argument("--timestamp-tolerance", type=float, default=DEFAULT_TIMESTAMP_TOLERANCE)
    b.add_argument("--max-frames", type=int, default=None, help="Optional quick test limiter")

    # ---- submit ----
    s = sub.add_parser("submit", help="Upload .jsonl and create Batch Job")
    s.add_argument("--batch-input", required=True, help="Path to .jsonl from 'build'")
    s.add_argument("--manifest", default=None, help="Path to build manifest JSON")
    s.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # ---- collect ----
    c = sub.add_parser("collect", help="Poll batch status, download & merge results")
    c.add_argument("--batch-id", required=True, help="Batch Job ID from 'submit'")
    c.add_argument("--manifest", default=None, help="Path to build manifest JSON")
    c.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    c.add_argument("--poll-interval", type=int, default=60)

    # ---- run (real-time) ----
    r = sub.add_parser("run", help="Real-time API calls (faster, not batch-discounted)")
    r.add_argument("-p", "--participant", required=True, help="Participant, e.g. P2 or 2")
    r.add_argument("-s", "--scenario", required=True, help="Scenario prefix/full, e.g. S1 or S1_normal")
    r.add_argument("--data-root", default=None, help="Dataset root for WhereIsData")
    r.add_argument("--gaze-csv-dir", default=DEFAULT_GAZE_CSV_DIR)
    r.add_argument("--gaze-csv-file", default=None, help="Explicit gaze CSV path (supports sliced CSV)")
    r.add_argument("--prefer-sliced", action="store_true", help="Prefer *_merged_gaze_labelled_sliced.csv when auto matching")
    r.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--detail", default="high", choices=["low", "high"])
    r.add_argument("--timestamp-tolerance", type=float, default=DEFAULT_TIMESTAMP_TOLERANCE)
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--max-frames", type=int, default=None, help="Optional quick test limiter")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers: participant/scenario/csv discovery
# ---------------------------------------------------------------------------
def _normalize_participant(p: str) -> str:
    s = str(p).strip().upper()
    if s.startswith("P"):
        s = s[1:]
    if not s.isdigit():
        raise ValueError(f"Invalid participant: {p}")
    return f"P{int(s)}"


def _normalize_scenario_prefix(s: str) -> str:
    raw = str(s).strip()
    if "_" in raw:
        first = raw.split("_", 1)[0]
    else:
        first = raw
    first_u = first.upper()
    if first_u.startswith("S"):
        first_u = first_u[1:]
    if not first_u.isdigit():
        raise ValueError(f"Invalid scenario: {s}")
    return f"S{int(first_u)}"


def _parse_scenario_from_csv_name(csv_name: str) -> str:
    m = re.match(
        r"^(P\d+)_(S\d+_[^_]+)_merged_gaze_labelled(?:_sliced)?\.csv$",
        csv_name,
    )
    if not m:
        raise ValueError(f"Unexpected labelled CSV name format: {csv_name}")
    return m.group(2)


def _find_labelled_csv(
    gaze_csv_dir: str,
    participant: str,
    scenario_arg: str,
    prefer_sliced: bool = False,
) -> tuple[Path, str]:
    base = Path(gaze_csv_dir)
    if not base.exists():
        raise FileNotFoundError(f"Gaze CSV directory not found: {base}")

    s = str(scenario_arg).strip()
    if "_" in s:
        scenario_prefix = _normalize_scenario_prefix(s)
        scenario_glob = s
        if not scenario_glob.startswith("S"):
            scenario_glob = f"{scenario_prefix}_{s.split('_', 1)[1]}"
        candidates = [
            f"{participant}_{scenario_glob}_merged_gaze_labelled_sliced.csv",
            f"{participant}_{scenario_glob}_merged_gaze_labelled.csv",
        ]
    else:
        scenario_prefix = _normalize_scenario_prefix(s)
        candidates = [
            f"{participant}_{scenario_prefix}_*_merged_gaze_labelled_sliced.csv",
            f"{participant}_{scenario_prefix}_*_merged_gaze_labelled.csv",
        ]

    if not prefer_sliced:
        candidates = list(reversed(candidates))

    matches = []
    for pattern in candidates:
        matches = sorted(base.glob(pattern))
        if matches:
            break

    if not matches:
        raise FileNotFoundError(f"No labelled gaze CSV matched any of {candidates} under {base}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple gaze CSV matched {pattern}: {[m.name for m in matches]}")

    matched = matches[0]
    scenario_full = _parse_scenario_from_csv_name(matched.name)
    return matched, scenario_full


def _resolve_labelled_csv_from_args(args, participant: str) -> tuple[Path, str]:
    if args.gaze_csv_file:
        p = Path(args.gaze_csv_file)
        if not p.exists():
            raise FileNotFoundError(f"Explicit gaze csv not found: {p}")
        scenario_full = _parse_scenario_from_csv_name(p.name)
        if not p.name.startswith(f"{participant}_"):
            raise ValueError(
                f"Participant mismatch: --participant {participant} but csv is {p.name}"
            )
        return p, scenario_full

    return _find_labelled_csv(
        args.gaze_csv_dir,
        participant,
        args.scenario,
        prefer_sliced=args.prefer_sliced,
    )


# ---------------------------------------------------------------------------
# Helpers: gaze mapping + frame selection
# ---------------------------------------------------------------------------
def _parse_ts_from_merged_name(name: str):
    m = re.search(r"merged_(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def _load_labelled_gaze_map(csv_path: Path):
    by_ts = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ui_timestamp", "gaze_target"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain columns {required}: {csv_path}")

        temp = {}
        for row in reader:
            try:
                ts = float(row["ui_timestamp"])
            except (TypeError, ValueError):
                continue
            label = (row.get("gaze_target") or "uncertain").strip() or "uncertain"
            key = f"{ts:.6f}"
            if key not in temp:
                temp[key] = []
            temp[key].append(label)

    sorted_ts = []
    for key, labels in temp.items():
        ts = float(key)
        major = Counter(labels).most_common(1)[0][0]
        by_ts[key] = major
        sorted_ts.append(ts)

    sorted_ts.sort()
    return sorted_ts, by_ts


def _nearest_gaze_target(ts: float, sorted_ts: list[float], ts_to_label: dict[str, str], tol: float) -> str:
    if not sorted_ts:
        return "uncertain"

    pos = bisect_left(sorted_ts, ts)
    candidates = []
    if pos < len(sorted_ts):
        candidates.append(sorted_ts[pos])
    if pos > 0:
        candidates.append(sorted_ts[pos - 1])
    if not candidates:
        return "uncertain"

    best = min(candidates, key=lambda x: abs(x - ts))
    if abs(best - ts) > tol:
        return "uncertain"
    return ts_to_label.get(f"{best:.6f}", "uncertain")


def _pick_closest_merged_frames(
    merged_items: list[tuple[float, str]],
    desired_ts: list[float],
    tol: float,
) -> list[tuple[float, str]]:
    if not merged_items or not desired_ts:
        return []

    merged_items = sorted(merged_items, key=lambda x: x[0])
    merged_ts = [x[0] for x in merged_items]
    chosen_idx = set()

    for target_ts in desired_ts:
        pos = bisect_left(merged_ts, target_ts)
        candidates = []
        if pos < len(merged_ts):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue

        best_i = min(candidates, key=lambda i: abs(merged_ts[i] - target_ts))
        if abs(merged_ts[best_i] - target_ts) <= tol:
            chosen_idx.add(best_i)

    return [merged_items[i] for i in sorted(chosen_idx)]


def _resolve_targets(
    participant: str,
    scenario_full: str,
    data_root: str | None,
    gaze_csv_path: Path,
    tolerance: float,
    max_frames: int | None,
):
    finder = WhereIsData(data_root=data_root)
    try:
        entries = finder.path_to_required_data(participant, scenario_full, seg_type="all")
    except DataLocationError as e:
        raise RuntimeError(str(e)) from e

    merged_paths = sorted({e["merged_img"] for e in entries})
    sorted_ts, ts_to_label = _load_labelled_gaze_map(gaze_csv_path)

    merged_items = []
    for image_path in merged_paths:
        ts = _parse_ts_from_merged_name(os.path.basename(image_path))
        if ts is None:
            continue
        merged_items.append((ts, image_path))

    use_sliced_mode = gaze_csv_path.name.endswith("_merged_gaze_labelled_sliced.csv")
    if use_sliced_mode:
        merged_items = _pick_closest_merged_frames(merged_items, sorted_ts, tolerance)

    targets = []
    for ts, image_path in merged_items:
        gaze_target = _nearest_gaze_target(ts, sorted_ts, ts_to_label, tolerance)
        targets.append(
            {
                "participant": participant,
                "scenario": scenario_full,
                "scenario_prefix": scenario_full.split("_", 1)[0],
                "timestamp": ts,
                "image_path": image_path,
                "gaze_target": gaze_target,
            }
        )

    targets.sort(key=lambda x: x["timestamp"])
    if max_frames is not None and max_frames > 0:
        targets = targets[:max_frames]

    if not targets:
        raise RuntimeError("No targets resolved from merged_ui + labelled gaze CSV.")

    return targets


# ---------------------------------------------------------------------------
# Image + prompt + request
# ---------------------------------------------------------------------------
def encode_image(path: str, image_size: int) -> str:
    from PIL import Image
    import io

    img = Image.open(path).convert("RGB")
    if image_size:
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _render_user_message(template: str, gaze_target: str) -> str:
    return template.replace("{gaze_target}", gaze_target)


def _make_request(
    custom_id: str,
    model: str,
    system_msg: str,
    user_msg: str,
    b64_image: str,
    detail: str,
    max_tokens: int,
    json_mode: bool,
) -> dict:
    token_field = "max_completion_tokens" if model.lower().startswith("gpt-5") else "max_tokens"
    body = {
        "model": model,
        token_field: max_tokens,
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": detail,
                        },
                    },
                    {"type": "text", "text": user_msg},
                ],
            },
        ],
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _make_custom_id(mode: str, t: dict) -> str:
    return (
        f"{mode}__{t['participant']}__{t['scenario']}"
        f"__{t['timestamp']:.6f}"
    )


def parse_custom_id(custom_id: str) -> dict:
    parts = custom_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"Invalid custom_id: {custom_id}")
    return {
        "mode": parts[0],
        "participant": parts[1],
        "scenario": parts[2],
        "timestamp": float(parts[3]),
    }


# ---------------------------------------------------------------------------
# Commands: build / submit / collect / run
# ---------------------------------------------------------------------------
def cmd_build(args):
    participant = _normalize_participant(args.participant)
    gaze_csv_path, scenario_full = _resolve_labelled_csv_from_args(args, participant)

    print(f"Resolved labelled gaze CSV: {gaze_csv_path}")
    print(f"Resolved scenario: {scenario_full}")

    targets = _resolve_targets(
        participant=participant,
        scenario_full=scenario_full,
        data_root=args.data_root,
        gaze_csv_path=gaze_csv_path,
        tolerance=args.timestamp_tolerance,
        max_frames=args.max_frames,
    )
    print(f"Resolved {len(targets)} merged_ui targets.")

    freedom_prompt = LLMPrompt(seed="FREEDOM")
    structured_prompt = LLMPrompt(seed="STRUCTURED")

    os.makedirs(args.output_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{participant}_{scenario_full}_{run_ts}"
    out_jsonl = os.path.join(args.output_dir, f"batch_requests_{prefix}.jsonl")
    manifest_path = os.path.join(args.output_dir, f"batch_targets_{prefix}.json")

    total = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, t in enumerate(targets):
            print(f"  Encoding frame {i+1}/{len(targets)}: {t['image_path']}", end="\r")
            b64 = encode_image(t["image_path"], args.image_size)

            for mode, prompt in [("freedom", freedom_prompt), ("structured", structured_prompt)]:
                cid = _make_custom_id(mode, t)
                request = _make_request(
                    custom_id=cid,
                    model=args.model,
                    system_msg=prompt.system_message,
                    user_msg=_render_user_message(prompt.user_message, t["gaze_target"]),
                    b64_image=b64,
                    detail=args.detail,
                    max_tokens=args.max_tokens,
                    json_mode=(mode == "structured"),
                )
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
                total += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_ts": run_ts,
                "participant": participant,
                "scenario": scenario_full,
                "gaze_csv": str(gaze_csv_path),
                "total_targets": len(targets),
                "targets": targets,
                "batch_input": out_jsonl,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nWrote {total} requests -> {out_jsonl}")
    print(f"Saved target manifest -> {manifest_path}")
    print("\nNext step:")
    print(f"  python gpt_wz_api_ziqi.py submit --batch-input {out_jsonl} --manifest {manifest_path}")


def _infer_manifest_from_batch_input(batch_input: str) -> str | None:
    name = os.path.basename(batch_input)
    if not name.startswith("batch_requests_"):
        return None
    candidate = os.path.join(
        os.path.dirname(batch_input),
        name.replace("batch_requests_", "batch_targets_", 1).replace(".jsonl", ".json"),
    )
    return candidate if os.path.exists(candidate) else None


def cmd_submit(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    manifest = args.manifest or _infer_manifest_from_batch_input(args.batch_input)

    print(f"Uploading {args.batch_input} ...")
    with open(args.batch_input, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"  File uploaded: {upload.id}")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch job created: {batch.id}  status={batch.status}")

    os.makedirs(args.output_dir, exist_ok=True)
    info_path = os.path.join(args.output_dir, f"batch_job_{batch.id}.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_id": batch.id,
                "file_id": upload.id,
                "input": args.batch_input,
                "manifest": manifest,
                "submitted_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"  Job info saved -> {info_path}")
    print("\nNext step:")
    print(f"  python gpt_wz_api_ziqi.py collect --batch-id {batch.id}")


def _load_manifest_map(manifest_path: str | None):
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    targets = obj.get("targets", [])
    mapping = {}
    for t in targets:
        key = (t.get("participant"), t.get("scenario"), float(t.get("timestamp")))
        mapping[key] = t
    return mapping


def _find_manifest_from_batch_info(output_dir: str, batch_id: str) -> str | None:
    info_path = os.path.join(output_dir, f"batch_job_{batch_id}.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    manifest = info.get("manifest")
    if manifest and os.path.exists(manifest):
        return manifest
    return _infer_manifest_from_batch_input(info.get("input", ""))


def cmd_collect(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Polling batch job {args.batch_id} ...")
    while True:
        batch = client.batches.retrieve(args.batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else "?"
        total = batch.request_counts.total if batch.request_counts else "?"
        print(f"  status={status}  completed={completed}/{total}")

        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(args.poll_interval)

    if status != "completed":
        print(f"[ERROR] Batch ended with status: {status}")
        sys.exit(1)

    manifest_path = args.manifest or _find_manifest_from_batch_info(args.output_dir, args.batch_id)
    target_map = _load_manifest_map(manifest_path)

    print(f"Downloading results (file_id={batch.output_file_id}) ...")
    content = client.files.content(batch.output_file_id)
    raw_lines = content.text.strip().splitlines()
    print(f"  Downloaded {len(raw_lines)} result lines.")

    responses = {}
    errors = []
    for line in raw_lines:
        obj = json.loads(line)
        cid = obj["custom_id"]
        if obj.get("error"):
            errors.append({"custom_id": cid, "error": obj["error"]})
            continue
        text = obj["response"]["body"]["choices"][0]["message"]["content"]
        responses[cid] = text

    if errors:
        print(f"  [WARN] {len(errors)} failed requests")

    merged = {}
    for cid, text in responses.items():
        meta = parse_custom_id(cid)
        k = (meta["participant"], meta["scenario"], meta["timestamp"])

        if k not in merged:
            t = target_map.get(k, {})
            merged[k] = {
                "participant": meta["participant"],
                "scenario": meta["scenario"],
                "scenario_prefix": meta["scenario"].split("_", 1)[0],
                "timestamp": meta["timestamp"],
                "image_path": t.get("image_path", ""),
                "gaze_target": t.get("gaze_target", ""),
                "freedom_response": None,
                "structured_response": None,
            }

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text

        if meta["mode"] == "freedom":
            merged[k]["freedom_response"] = parsed
        else:
            merged[k]["structured_response"] = parsed

    results = sorted(
        merged.values(),
        key=lambda r: (r["participant"], r["scenario"], r["timestamp"]),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"workzone_gpt_annotations_{run_ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_id": args.batch_id,
                "manifest": manifest_path,
                "run_ts": run_ts,
                "total": len(results),
                "errors": len(errors),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(results)} annotated frames -> {out_path}")
    if errors:
        err_path = out_path.replace(".json", "_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors -> {err_path}")


def _call_api(client, model, system_msg, user_msg, b64_image, detail, max_tokens, json_mode=True):
    token_field = "max_completion_tokens" if model.lower().startswith("gpt-5") else "max_tokens"
    kwargs = {
        "model": model,
        token_field: max_tokens,
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": detail,
                        },
                    },
                    {"type": "text", "text": user_msg},
                ],
            },
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    wait = 2
    for _ in range(6):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                raise

    raise RuntimeError("Max retries exceeded due to rate limit")


def cmd_run(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    participant = _normalize_participant(args.participant)
    gaze_csv_path, scenario_full = _resolve_labelled_csv_from_args(args, participant)

    print(f"Resolved labelled gaze CSV: {gaze_csv_path}")
    print(f"Resolved scenario: {scenario_full}")

    targets = _resolve_targets(
        participant=participant,
        scenario_full=scenario_full,
        data_root=args.data_root,
        gaze_csv_path=gaze_csv_path,
        tolerance=args.timestamp_tolerance,
        max_frames=args.max_frames,
    )

    print(f"Resolved {len(targets)} merged_ui targets.")
    print(f"Total API calls: {len(targets) * 2} (freedom + structured)")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    freedom_prompt = LLMPrompt(seed="FREEDOM")
    structured_prompt = LLMPrompt(seed="STRUCTURED")

    tasks = []
    for t in targets:
        for mode, prompt in [("freedom", freedom_prompt), ("structured", structured_prompt)]:
            tasks.append((t, mode, prompt))

    results_map = {}
    done = 0
    total = len(tasks)

    def _process(task):
        t, mode, prompt = task
        b64 = encode_image(t["image_path"], args.image_size)
        user_msg = _render_user_message(prompt.user_message, t["gaze_target"])
        text = _call_api(
            client,
            args.model,
            prompt.system_message,
            user_msg,
            b64,
            args.detail,
            args.max_tokens,
            json_mode=(mode == "structured"),
        )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        return t, mode, parsed

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                t, mode, parsed = future.result()
            except Exception as e:
                task = futures[future]
                print(f"  [ERROR] {task[0]['participant']} {task[1]}: {e}")
                continue

            done += 1
            k = (t["participant"], t["scenario"], t["timestamp"])
            if k not in results_map:
                results_map[k] = {
                    "participant": t["participant"],
                    "scenario": t["scenario"],
                    "scenario_prefix": t["scenario_prefix"],
                    "timestamp": t["timestamp"],
                    "image_path": t["image_path"],
                    "gaze_target": t["gaze_target"],
                    "freedom_response": None,
                    "structured_response": None,
                }

            if mode == "freedom":
                results_map[k]["freedom_response"] = parsed
            else:
                results_map[k]["structured_response"] = parsed

            print(f"  [{done}/{total}] {t['participant']}/{t['scenario']} ts={t['timestamp']:.3f} [{mode}]")

    results = sorted(
        results_map.values(),
        key=lambda r: (r["participant"], r["scenario"], r["timestamp"]),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"workzone_gpt_rt_{run_ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_ts": run_ts,
                "model": args.model,
                "participant": participant,
                "scenario": scenario_full,
                "total": len(results),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(results)} annotated frames -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "run":
        cmd_run(args)
