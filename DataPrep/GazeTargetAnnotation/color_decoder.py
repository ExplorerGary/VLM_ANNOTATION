import csv
import json
import os
import re
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

from where_is_data import DataLocationError, WhereIsData

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


THIS = os.path.dirname(os.path.abspath(__file__))
BIND_JSON_PATH = os.path.join(THIS, "bind_output", "img_mapping_raw.json")
OUT_DIR = os.path.join(THIS, "gaze_target_output")


carla_colored_classes = {
    0: [0, 0, 0],
    1: [128, 64, 128],
    2: [244, 35, 232],
    3: [70, 70, 70],
    4: [102, 102, 156],
    5: [190, 153, 153],
    6: [153, 153, 153],
    7: [250, 170, 30],
    8: [220, 220, 0],
    9: [107, 142, 35],
    10: [152, 251, 152],
    11: [70, 130, 180],
    12: [220, 20, 60],
    13: [255, 0, 0],
    14: [0, 0, 142],
    15: [0, 0, 70],
    16: [0, 60, 100],
    17: [0, 80, 100],
    18: [0, 0, 230],
    19: [119, 11, 32],
    20: [110, 190, 160],
    21: [170, 120, 50],
    22: [55, 90, 80],
    23: [45, 60, 150],
    24: [157, 234, 50],
    25: [81, 0, 81],
    26: [150, 100, 100],
    27: [230, 150, 140],
    28: [180, 165, 180],
}
'''
Value	Tag	Converted color	Description
0	Unlabeled	(0, 0, 0)	Elements that have not been categorized are considered Unlabeled. This category is meant to be empty or at least contain elements with no collisions.
1	Roads	(128, 64, 128)	Part of ground on which cars usually drive.
E.g. lanes in any directions, and streets.
2	SideWalks	(244, 35, 232)	Part of ground designated for pedestrians or cyclists. Delimited from the road by some obstacle (such as curbs or poles), not only by markings. This label includes a possibly delimiting curb, traffic islands (the walkable part), and pedestrian zones.
3	Building	(70, 70, 70)	Buildings like houses, skyscrapers,... and the elements attached to them.
E.g. air conditioners, scaffolding, awning or ladders and much more.
4	Wall	(102, 102, 156)	Individual standing walls. Not part of a building.
5	Fence	(190, 153, 153)	Barriers, railing, or other upright structures. Basically wood or wire assemblies that enclose an area of ground.
6	Pole	(153, 153, 153)	Small mainly vertically oriented pole. If the pole has a horizontal part (often for traffic light poles) this is also considered pole.
E.g. sign pole, traffic light poles.
7	TrafficLight	(250, 170, 30)	Traffic light boxes without their poles.
8	TrafficSign	(220, 220, 0)	Signs installed by the state/city authority, usually for traffic regulation. This category does not include the poles where signs are attached to.
E.g. traffic- signs, parking signs, direction signs...
9	Vegetation	(107, 142, 35)	Trees, hedges, all kinds of vertical vegetation. Ground-level vegetation is considered Terrain.
10	Terrain	(152, 251, 152)	Grass, ground-level vegetation, soil or sand. These areas are not meant to be driven on. This label includes a possibly delimiting curb.
11	Sky	(70, 130, 180)	Open sky. Includes clouds and the sun.
12	Pedestrian	(220, 20, 60)	Humans that walk
13	Rider	(255, 0, 0)	Humans that ride/drive any kind of vehicle or mobility system
E.g. bicycles or scooters, skateboards, horses, roller-blades, wheel-chairs, etc. .
14	Car	(0, 0, 142)	Cars, vans
15	Truck	(0, 0, 70)	Trucks
16	Bus	(0, 60, 100)	Busses
17	Train	(0, 80, 100)	Trains
18	Motorcycle	(0, 0, 230)	Motorcycle, Motorbike
19	Bicycle	(119, 11, 32)	Bicylces
20	Static	(110, 190, 160)	Elements in the scene and props that are immovable.
E.g. fire hydrants, fixed benches, fountains, bus stops, etc.
21	Dynamic	(170, 120, 50)	Elements whose position is susceptible to change over time.
E.g. Movable trash bins, buggies, bags, wheelchairs, animals, etc.
22	Other	(55, 90, 80)	Everything that does not belong to any other category.
23	Water	(45, 60, 150)	Horizontal water surfaces.
E.g. Lakes, sea, rivers.
24	RoadLine	(157, 234, 50)	The markings on the road.
25	Ground	(81, 0, 81)	Any horizontal ground-level structures that does not match any other category. For example areas shared by vehicles and pedestrians, or flat roundabouts delimited from the road by a curb.
26	Bridge	(150, 100, 100)	Only the structure of the bridge. Fences, people, vehicles, an other elements on top of it are labeled separately.
27	RailTrack	(230, 150, 140)	All kind of rail tracks that are non-drivable by cars.
E.g. subway and train rail tracks.
28	GuardRail	(180, 165, 180)	All types of guard rails/crash barriers.
'''

class ColorDecoder:
    def __init__(self, color_mapping: Dict[int, List[int]]):
        self.color_mapping = color_mapping
        self.inverse_mapping = {tuple(v): k for k, v in color_mapping.items()}

    def decode(self, rgb_color: List[int]):
        rgb_tuple = tuple(rgb_color)
        return self.inverse_mapping.get(rgb_tuple, "unknown")

    @staticmethod
    def decode_red(red_value: int):
        # For raw segmentation, class id is encoded in red channel.
        return red_value if 0 <= red_value <= 255 else "unknown"


def _safe_int_from_float_str(value: str) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _get_image_size(path: str, size_cache: Dict[str, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if path in size_cache:
        return size_cache[path]
    if not path or not os.path.isfile(path):
        size_cache[path] = None
        return None
    with Image.open(path) as img:
        size_cache[path] = img.size
    return size_cache[path]


def _resolve_raw_seg_path(seg_path: str) -> Optional[str]:
    raw_path = seg_path
    if "segmentation_color_" in seg_path:
        raw_path = seg_path.replace("segmentation_color_", "segmentation_raw_")
    if os.path.isfile(raw_path):
        return raw_path
    return None


def _estimate_gaze_space(rows: List[Dict[str, str]]) -> Optional[Tuple[int, int]]:
    max_x = -1
    max_y = -1
    for row in rows:
        x = _safe_int_from_float_str(row.get("gaze_x", ""))
        y = _safe_int_from_float_str(row.get("gaze_y", ""))
        if x is None or y is None:
            continue
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    if max_x < 0 or max_y < 0:
        return None

    return (max_x + 1, max_y + 1)


def _resolve_nearest_timestamp_seg(
    ui_ts: str,
    sorted_ts: List[float],
    ts_to_seg: Dict[str, str],
    tolerance_sec: float = 0.2,
) -> Optional[str]:
    try:
        target = float(ui_ts)
    except (TypeError, ValueError):
        return None

    if not sorted_ts:
        return None

    pos = bisect_left(sorted_ts, target)
    candidates: List[float] = []
    if pos < len(sorted_ts):
        candidates.append(sorted_ts[pos])
    if pos > 0:
        candidates.append(sorted_ts[pos - 1])
    if not candidates:
        return None

    best = min(candidates, key=lambda t: abs(t - target))
    if abs(best - target) > tolerance_sec:
        return None

    return ts_to_seg.get(f"{best:.6f}")


def _extract_red_values_for_rows(
    seg_path: str,
    rows: List[Dict[str, str]],
    size_cache: Dict[str, Tuple[int, int]],
    gaze_space: Optional[Tuple[int, int]],
) -> None:
    with Image.open(seg_path) as img:
        rgb = img.convert("RGB")
        seg_width, seg_height = rgb.size
        pixels = rgb.load()

        for row in rows:
            x = _safe_int_from_float_str(row.get("gaze_x", ""))
            y = _safe_int_from_float_str(row.get("gaze_y", ""))
            if x is None or y is None:
                row["gaze_target"] = "unknown"
                continue

            # Gaze coordinates are in merged UI image space; rescale to seg space if needed.
            if x < 0 or y < 0 or x >= seg_width or y >= seg_height:
                if gaze_space is not None:
                    gaze_width, gaze_height = gaze_space
                    if gaze_width > 0 and gaze_height > 0:
                        x = int(round(x * seg_width / gaze_width))
                        y = int(round(y * seg_height / gaze_height))

            if x < 0 or y < 0 or x >= seg_width or y >= seg_height:
                merged_path = row.get("_merged_path", "")
                merged_size = _get_image_size(merged_path, size_cache)
                if merged_size is not None:
                    ui_width, ui_height = merged_size
                    if ui_width > 0 and ui_height > 0:
                        x = int(round(x * seg_width / ui_width))
                        y = int(round(y * seg_height / ui_height))

            if x < 0 or y < 0 or x >= seg_width or y >= seg_height:
                row["gaze_target"] = "unknown"
                continue

            red_value = pixels[x, y][0]
            row["gaze_target"] = str(ColorDecoder.decode_red(red_value))


def _load_mapping_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_ui_to_seg_index(mapping_block: Dict[str, str]) -> Dict[str, str]:
    ui_to_seg: Dict[str, str] = {}
    for seg_path, merged_path in mapping_block.items():
        raw_seg_path = _resolve_raw_seg_path(seg_path)
        if not raw_seg_path:
            continue
        ui_to_seg[os.path.basename(merged_path)] = raw_seg_path
    return ui_to_seg


def _build_timestamp_to_seg_index(mapping_block: Dict[str, str]) -> Dict[str, str]:
    # Supports merged filename pattern: merged_1764105165.262000.jpg
    pattern = re.compile(r"^merged_(\d+\.\d+)\.[^.]+$", re.IGNORECASE)
    ts_to_seg: Dict[str, str] = {}
    for seg_path, merged_path in mapping_block.items():
        raw_seg_path = _resolve_raw_seg_path(seg_path)
        if not raw_seg_path:
            continue
        name = os.path.basename(merged_path)
        match = pattern.match(name)
        if not match:
            continue
        try:
            ts_key = f"{float(match.group(1)):.6f}"
        except ValueError:
            continue
        ts_to_seg[ts_key] = raw_seg_path
    return ts_to_seg


def _sorted_mapping_timestamps(ts_to_seg: Dict[str, str]) -> List[float]:
    vals: List[float] = []
    for key in ts_to_seg.keys():
        try:
            vals.append(float(key))
        except ValueError:
            continue
    vals.sort()
    return vals


def _build_seg_to_merged_path(mapping_block: Dict[str, str]) -> Dict[str, str]:
    seg_to_merged: Dict[str, str] = {}
    for seg_path, merged_path in mapping_block.items():
        raw_seg_path = _resolve_raw_seg_path(seg_path)
        if not raw_seg_path:
            continue
        seg_to_merged[raw_seg_path] = merged_path
    return seg_to_merged


def _read_merge_log_rows(merge_log_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(merge_log_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_gaze_target_csvs(
    mapping_json_path: str = BIND_JSON_PATH,
    output_dir: str = OUT_DIR,
    data_root: Optional[str] = None,
) -> Dict[str, int]:
    where = WhereIsData(data_root=data_root)
    payload = _load_mapping_json(mapping_json_path)
    participants = payload.get("participants", {})

    os.makedirs(output_dir, exist_ok=True)

    file_count = 0
    row_count = 0
    error_count = 0
    unknown_count = 0
    unresolved_raw_mapping = 0
    size_cache: Dict[str, Tuple[int, int]] = {}

    participant_items = list(participants.items())
    for participant, p_block in tqdm(participant_items, desc="Participants"):
        scenarios = p_block.get("scenarios", {})
        scenario_items = list(scenarios.items())
        for scenario, s_block in tqdm(scenario_items, desc=f"{participant} scenarios", leave=False):
            mapping_block = s_block.get("mapping", {})
            if not mapping_block:
                continue

            for seg_path in mapping_block.keys():
                if _resolve_raw_seg_path(seg_path) is None:
                    unresolved_raw_mapping += 1

            try:
                required_entries = where.path_to_required_data(
                    participant_id=participant,
                    scenario_id=scenario,
                    seg_type="raw",
                    strict_pairing=False,
                )
            except DataLocationError:
                error_count += 1
                continue

            if not required_entries:
                error_count += 1
                continue

            merge_log_path = required_entries[0].get("merged_log")
            if not merge_log_path or not os.path.isfile(merge_log_path):
                error_count += 1
                continue

            rows, fieldnames = _read_merge_log_rows(merge_log_path)
            if not fieldnames:
                error_count += 1
                continue

            if "gaze_target" not in fieldnames:
                fieldnames = fieldnames + ["gaze_target"]

            ui_to_seg = _build_ui_to_seg_index(mapping_block)
            ts_to_seg = _build_timestamp_to_seg_index(mapping_block)
            sorted_ts = _sorted_mapping_timestamps(ts_to_seg)
            seg_to_merged = _build_seg_to_merged_path(mapping_block)
            gaze_space = _estimate_gaze_space(rows)

            grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in rows:
                ui_name = row.get("ui_image", "")
                seg_path = ui_to_seg.get(ui_name)

                if not seg_path:
                    ui_ts = row.get("ui_timestamp", "")
                    try:
                        ts_key = f"{float(ui_ts):.6f}"
                    except (TypeError, ValueError):
                        ts_key = ""
                    if ts_key:
                        seg_path = ts_to_seg.get(ts_key)

                if not seg_path:
                    seg_path = _resolve_nearest_timestamp_seg(
                        row.get("ui_timestamp", ""),
                        sorted_ts,
                        ts_to_seg,
                        tolerance_sec=0.2,
                    )

                if not seg_path:
                    row["gaze_target"] = "unknown"
                    continue
                row["_merged_path"] = seg_to_merged.get(seg_path, "")
                grouped_rows[seg_path].append(row)

            grouped_items = list(grouped_rows.items())
            for seg_path, seg_rows in tqdm(grouped_items, desc=f"{participant}/{scenario} seg", leave=False):
                if not os.path.isfile(seg_path):
                    for row in seg_rows:
                        row["gaze_target"] = "unknown"
                    continue
                _extract_red_values_for_rows(seg_path, seg_rows, size_cache, gaze_space)

            for row in rows:
                if row.get("gaze_target") == "unknown":
                    unknown_count += 1
                row.pop("_merged_path", None)

            output_name = f"{participant}_{scenario}_merged_gaze.csv"
            output_path = os.path.join(output_dir, output_name)
            _write_csv(output_path, rows, fieldnames)

            file_count += 1
            row_count += len(rows)

    return {
        "generated_files": file_count,
        "total_rows": row_count,
        "errors": error_count,
        "unknown_rows": unknown_count,
        "unresolved_raw_mapping": unresolved_raw_mapping,
    }


def main():
    report = generate_gaze_target_csvs()
    print("Gaze target generation finished")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()