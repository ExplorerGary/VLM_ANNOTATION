"""Repair incomplete workzone annotation JSON files.

This utility scans an annotation result JSON for null response fields,
reruns the same run-mode prompts for only the missing modes, and writes a
repaired JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import gpt_wz_api_ziqi as pipeline

RESPONSE_FIELDS = ("freedom_response", "structured_response", "response")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair null responses in a workzone annotation JSON by rerunning the missing modes.",
    )
    parser.add_argument("--input", required=True, help="Path to an annotation JSON file to repair")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the repaired JSON. Defaults to <input>_repaired.json",
    )
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON to recover missing metadata")
    parser.add_argument("--data-root", default=None, help="Dataset root for WhereIsData")
    parser.add_argument("--model", default=pipeline.DEFAULT_MODEL)
    parser.add_argument("--image-size", type=int, default=pipeline.IMAGE_SIZE)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--detail", default="high", choices=["low", "high"])
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Report missing entries without calling the API")
    return parser.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_manifest_map(manifest_path: str | None) -> dict[tuple[str, str, float], dict]:
    if not manifest_path or not os.path.exists(manifest_path):
        return {}

    manifest = _load_json(manifest_path)
    mapping = {}
    for target in manifest.get("targets", []):
        try:
            key = (
                target.get("participant"),
                target.get("scenario"),
                float(target.get("timestamp")),
            )
        except (TypeError, ValueError):
            continue
        mapping[key] = target
    return mapping


def _repair_candidate_key(entry: dict) -> tuple[str, str, float] | None:
    participant = entry.get("participant")
    scenario = entry.get("scenario")
    timestamp = entry.get("timestamp")
    if participant is None or scenario is None or timestamp is None:
        return None
    try:
        return participant, scenario, float(timestamp)
    except (TypeError, ValueError):
        return None


def _find_missing_modes(entry: dict) -> list[str]:
    missing = []
    if entry.get("freedom_response") is None and entry.get("response") is None:
        missing.append("freedom")
    if entry.get("structured_response") is None:
        missing.append("structured")
    return missing


def _recover_entry_metadata(entry: dict, manifest_map: dict[tuple[str, str, float], dict]) -> dict:
    recovered = dict(entry)
    key = _repair_candidate_key(entry)
    if key and key in manifest_map:
        source = manifest_map[key]
        recovered.setdefault("image_path", source.get("image_path", ""))
        recovered.setdefault("gaze_target", source.get("gaze_target", ""))
        recovered.setdefault("scenario_prefix", source.get("scenario_prefix", recovered.get("scenario", "").split("_", 1)[0]))
    return recovered


def _run_mode(client: OpenAI, mode: str, entry: dict, model: str, image_size: int, detail: str, max_tokens: int) -> tuple[str, object]:
    prompt = pipeline.LLMPrompt(seed="FREEDOM" if mode == "freedom" else "STRUCTURED")
    image_path = entry.get("image_path")
    if not image_path:
        raise RuntimeError(f"Missing image_path for {entry.get('participant')} {entry.get('scenario')} {entry.get('timestamp')}")

    gaze_target = entry.get("gaze_target") or "uncertain"
    b64 = pipeline.encode_image(image_path, image_size)
    user_msg = pipeline._render_user_message(prompt.user_message, gaze_target)
    text = pipeline._call_api(
        client,
        model,
        prompt.system_message,
        user_msg,
        b64,
        detail,
        max_tokens,
        json_mode=(mode == "structured"),
    )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    return mode, parsed


def repair_annotations(args: argparse.Namespace) -> dict:
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Annotation JSON not found: {input_path}")

    payload = _load_json(input_path)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Input JSON must contain a results array")

    manifest_path = args.manifest or payload.get("manifest")
    manifest_map = _load_manifest_map(manifest_path)

    repair_jobs = []
    for index, entry in enumerate(results):
        if not isinstance(entry, dict):
            continue
        missing_modes = _find_missing_modes(entry)
        if not missing_modes:
            continue
        recovered_entry = _recover_entry_metadata(entry, manifest_map)
        repair_jobs.append((index, recovered_entry, missing_modes))

    report = {
        "input": input_path,
        "manifest": manifest_path,
        "missing_entries": len(repair_jobs),
        "missing_modes": sum(len(job[2]) for job in repair_jobs),
        "repaired_modes": 0,
        "dry_run": bool(args.dry_run),
    }

    print(f"Loaded {len(results)} entries from {input_path}")
    print(f"Found {report['missing_entries']} entries with missing responses")
    print(f"Missing modes total: {report['missing_modes']}")

    if not repair_jobs:
        payload["repair_report"] = report
        return payload

    if args.dry_run:
        for index, entry, missing_modes in repair_jobs:
            print(
                f"  [DRY-RUN] index={index} participant={entry.get('participant')} scenario={entry.get('scenario')} ts={entry.get('timestamp')} missing={missing_modes}"
            )
        payload["repair_report"] = report
        return payload

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    repaired = 0

    tasks = []
    for index, entry, missing_modes in repair_jobs:
        for mode in missing_modes:
            tasks.append((index, entry, mode))

    def _process(task: tuple[int, dict, str]) -> tuple[int, str, object]:
        index, entry, mode = task
        mode_name, parsed = _run_mode(
            client=client,
            mode=mode,
            entry=entry,
            model=args.model,
            image_size=args.image_size,
            detail=args.detail,
            max_tokens=args.max_tokens,
        )
        return index, mode_name, parsed

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process, task): task for task in tasks}
        for future in as_completed(futures):
            index, mode, parsed = future.result()
            entry = results[index]
            if mode == "freedom":
                entry["freedom_response"] = parsed
                if entry.get("response") is None:
                    entry.pop("response", None)
            else:
                entry["structured_response"] = parsed
            repaired += 1
            report["repaired_modes"] = repaired
            print(
                f"  repaired index={index} participant={entry.get('participant')} scenario={entry.get('scenario')} ts={entry.get('timestamp')} mode={mode}"
            )

    payload["results"] = results
    payload["repair_report"] = {
        **report,
        "completed_at": datetime.now().isoformat(),
    }
    return payload


def main() -> int:
    args = parse_args()
    repaired_payload = repair_annotations(args)

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else input_path.replace(".json", "_repaired.json")
    _save_json(output_path, repaired_payload)
    print(f"Saved repaired JSON -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())