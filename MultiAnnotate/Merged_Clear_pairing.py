import argparse
import json
import os
import re
from bisect import bisect_left
from datetime import datetime
from typing import Dict, List, Optional, Tuple


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA", "vlm-workzone_data")
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "multi-binding-result")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MERGED_PATTERN = re.compile(r"^merged_(\d+(?:\.\d+)?)\.[^.]+$", re.IGNORECASE)
FRONT_PATTERN = re.compile(r"^camera_(\d+)\.[^.]+$", re.IGNORECASE)


def _normalize_participant(value: str) -> str:
	token = str(value).strip().upper()
	if token.startswith("P"):
		token = token[1:]
	if not token.isdigit():
		raise ValueError(f"Invalid participant: {value}")
	return f"P{int(token)}"


def _normalize_scenario_prefix(value: str) -> str:
	token = str(value).strip()
	if "_" in token:
		token = token.split("_", 1)[0]
	token = token.upper()
	if token.startswith("S"):
		token = token[1:]
	if not token.isdigit():
		raise ValueError(f"Invalid scenario: {value}")
	return f"S{int(token)}"


def _is_image(path: str) -> bool:
	return path.lower().endswith(IMAGE_EXTENSIONS)


def _list_images(folder: str) -> List[str]:
	if not os.path.isdir(folder):
		return []
	files = []
	for name in os.listdir(folder):
		full = os.path.join(folder, name)
		if os.path.isfile(full) and _is_image(name):
			files.append(full)
	files.sort()
	return files


def _resolve_participant_dir(data_root: str, participant: str) -> str:
	direct = os.path.join(data_root, participant)
	if os.path.isdir(direct):
		return direct

	for level1_name in sorted(os.listdir(data_root)):
		level1_path = os.path.join(data_root, level1_name)
		if not os.path.isdir(level1_path):
			continue
		nested = os.path.join(level1_path, participant)
		if os.path.isdir(nested):
			return nested

	raise FileNotFoundError(f"Participant not found under data root: {participant}")


def _resolve_scenario_dir(participant_dir: str, scenario_arg: str) -> Tuple[str, str]:
	if "_" in scenario_arg:
		scenario_full = scenario_arg
		if not scenario_full.startswith("S"):
			prefix = _normalize_scenario_prefix(scenario_arg)
			scenario_full = f"{prefix}_{scenario_arg.split('_', 1)[1]}"
		scenario_dir = os.path.join(participant_dir, scenario_full)
		if not os.path.isdir(scenario_dir):
			raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
		return scenario_dir, scenario_full

	prefix = _normalize_scenario_prefix(scenario_arg)
	matches = [
		name
		for name in sorted(os.listdir(participant_dir))
		if os.path.isdir(os.path.join(participant_dir, name)) and name.startswith(prefix + "_")
	]
	if not matches:
		raise FileNotFoundError(
			f"No scenario matched prefix '{prefix}' under {participant_dir}"
		)
	if len(matches) > 1:
		raise RuntimeError(
			f"Multiple scenarios matched '{prefix}', please pass full name: {matches}"
		)

	scenario_full = matches[0]
	return os.path.join(participant_dir, scenario_full), scenario_full


def _pick_latest_carla_dir(scenario_dir: str) -> str:
	carla_dirs = [
		os.path.join(scenario_dir, name)
		for name in os.listdir(scenario_dir)
		if name.startswith("carla_data_") and os.path.isdir(os.path.join(scenario_dir, name))
	]
	if not carla_dirs:
		raise FileNotFoundError(f"No carla_data_* found under {scenario_dir}")
	carla_dirs.sort()
	return carla_dirs[-1]


def _extract_merged_ts_ms(path: str) -> Optional[int]:
	name = os.path.basename(path)
	match = MERGED_PATTERN.match(name)
	if not match:
		return None
	return int(round(float(match.group(1)) * 1000.0))


def _extract_front_ts_ms(path: str) -> Optional[int]:
	name = os.path.basename(path)
	match = FRONT_PATTERN.match(name)
	if not match:
		return None
	return int(match.group(1))


def _resolve_nearest(target: int, sorted_values: List[int]) -> Tuple[Optional[int], Optional[int]]:
	if not sorted_values:
		return None, None

	pos = bisect_left(sorted_values, target)
	candidates: List[int] = []
	if pos < len(sorted_values):
		candidates.append(sorted_values[pos])
	if pos > 0:
		candidates.append(sorted_values[pos - 1])
	if not candidates:
		return None, None

	best = min(candidates, key=lambda x: abs(x - target))
	return best, abs(best - target)


def build_front_to_merged_pairs(
	data_root: str,
	participant_arg: str,
	scenario_arg: str,
	tolerance_ms: int,
) -> Dict:
	participant = _normalize_participant(participant_arg)
	participant_dir = _resolve_participant_dir(data_root, participant)
	scenario_dir, scenario_full = _resolve_scenario_dir(participant_dir, scenario_arg)

	merged_ui_dir = os.path.join(scenario_dir, "merged_ui")
	if not os.path.isdir(merged_ui_dir):
		raise FileNotFoundError(f"merged_ui folder missing: {merged_ui_dir}")

	carla_dir = _pick_latest_carla_dir(scenario_dir)
	front_dir = os.path.join(carla_dir, "camera_front")
	if not os.path.isdir(front_dir):
		raise FileNotFoundError(f"camera_front folder missing: {front_dir}")

	merged_imgs = _list_images(merged_ui_dir)
	front_imgs = _list_images(front_dir)

	if not merged_imgs:
		raise FileNotFoundError(f"No merged UI images found: {merged_ui_dir}")
	if not front_imgs:
		raise FileNotFoundError(f"No front camera images found: {front_dir}")

	merged_ts_to_path: Dict[int, str] = {}
	for p in merged_imgs:
		ts = _extract_merged_ts_ms(p)
		if ts is None:
			continue
		merged_ts_to_path[ts] = p

	if not merged_ts_to_path:
		raise RuntimeError(
			"No valid merged timestamp parsed. Expected names like merged_1765482685.171000.jpg"
		)

	merged_ts_sorted = sorted(merged_ts_to_path.keys())
	pairs = []
	unmatched_front = []
	used_merged = set()

	for front_path in front_imgs:
		front_ts = _extract_front_ts_ms(front_path)
		if front_ts is None:
			unmatched_front.append(
				{
					"front_camera": front_path,
					"reason": "invalid_front_filename",
				}
			)
			continue

		nearest_ts, delta_ms = _resolve_nearest(front_ts, merged_ts_sorted)
		if nearest_ts is None or delta_ms is None or delta_ms > tolerance_ms:
			unmatched_front.append(
				{
					"front_camera": front_path,
					"front_timestamp_ms": front_ts,
					"reason": "no_merged_within_tolerance",
					"tolerance_ms": tolerance_ms,
				}
			)
			continue

		merged_path = merged_ts_to_path[nearest_ts]
		used_merged.add(nearest_ts)
		pairs.append(
			{
				"front_camera": front_path,
				"front_timestamp_ms": front_ts,
				"merged_ui": merged_path,
				"merged_timestamp_ms": nearest_ts,
				"time_diff_ms": delta_ms,
			}
		)

	unmatched_merged = [
		{
			"merged_ui": merged_ts_to_path[ts],
			"merged_timestamp_ms": ts,
			"reason": "not_selected_by_any_front_frame",
		}
		for ts in merged_ts_sorted
		if ts not in used_merged
	]

	summary = {
		"participant": participant,
		"scenario": scenario_full,
		"data_root": os.path.abspath(data_root),
		"front_dir": os.path.abspath(front_dir),
		"merged_ui_dir": os.path.abspath(merged_ui_dir),
		"counts": {
			"front_total": len(front_imgs),
			"merged_total": len(merged_ts_sorted),
			"paired": len(pairs),
			"unmatched_front": len(unmatched_front),
			"unmatched_merged": len(unmatched_merged),
		},
		"tolerance_ms": tolerance_ms,
		"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"pairs": pairs,
		"unmatched_front": unmatched_front,
		"unmatched_merged": unmatched_merged,
	}
	return summary


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Pair camera_front frames to merged_ui frames by nearest timestamp. "
			"Output one JSON under multi-binding-result."
		)
	)
	parser.add_argument("-p", "--participant", required=True, help="Participant, e.g. P8 or 8")
	parser.add_argument(
		"-s",
		"--scenario",
		required=True,
		help="Scenario, e.g. S1 or S1_normal",
	)
	parser.add_argument(
		"--data-root",
		default=DEFAULT_DATA_ROOT,
		help="Dataset root, default: DATA/vlm-workzone_data",
	)
	parser.add_argument(
		"--output-dir",
		default=DEFAULT_OUTPUT_DIR,
		help="Output folder for pairing JSON",
	)
	parser.add_argument(
		"--tolerance-ms",
		type=int,
		default=250,
		help="Maximum allowed timestamp difference for a valid pair",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.tolerance_ms < 0:
		raise ValueError("--tolerance-ms must be >= 0")

	result = build_front_to_merged_pairs(
		data_root=args.data_root,
		participant_arg=args.participant,
		scenario_arg=args.scenario,
		tolerance_ms=args.tolerance_ms,
	)

	os.makedirs(args.output_dir, exist_ok=True)
	out_name = f"{result['participant']}_{result['scenario']}_front_merged_pairing.json"
	out_path = os.path.join(args.output_dir, out_name)

	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2)

	print(f"Output: {out_path}")
	print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
