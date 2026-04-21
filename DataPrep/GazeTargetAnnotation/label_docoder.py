'''
Value	Tag	Converted color	Description
0	Unlabeled	(0, 0, 0)	In our case, it's worker and the cone a worker is holding. 
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
21	Dynamic	(170, 120, 50)	Elements whose position is susceptible to change over time. (in our case, it's only the fixed cone on the ground.
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

import argparse
import csv
from pathlib import Path

# 这个文件将读取我们之前标注的gaze target标签，并把它们转换成我们想要的prompt输入格式。这个文件的主要作用是把我们之前的标签和新的prompt设计连接起来，方便我们在后续的LLM annotation中使用。

OUTPUT_LABELS = [
    "road center", 
    "workzone element", 
    "cone", 
    "vehicle", 
    "traffic sign",
    "traffic light", 
    "uncertain", 
    "irrelevant"
]

# "workzone area" 目前被放弃，因为这东西比较模糊，容易和road center混淆，而且我们采集的数据里也没有明确这方面的数据，所以就先不使用这个标签了。
# 不过倒是可以让LLM在日后的output或者reasoning里描述一下这个东西，比如说"the driver is looking at the area where the workers are, but not at the workers themselves, which may indicate some level of awareness but not full attention to the main hazard" 之类的。
# 比如加一个： is focusing on workzone area <true/false>，

SEG_TO_GAZE = {

    # ROAD CENTER
    "road center": [1, 2, 24, 25],  # RoadLine, Ground

    # WORKZONE ELEMENT（工人/施工相关：pre-recorded / vr worker, and the cone holding by vr worker）
    "workzone element": [0],  # Unlabeled (我们在carla里把工人和工人拿着的cone都标注成了unlabeled)

    # CONE
    "cone": [21],  # Dynamic（我们的任务里定义为cone）

    # VEHICLE
    "vehicle": [14, 15, 16, 17, 18, 19],

    # TRAFFIC SIGN
    "traffic sign": [8],  # TrafficSign

    # TRAFFIC LIGHT
    "traffic light": [7],

    # UNCERTAIN
    "uncertain": [22, "unknown"],  # Other

    # IRRELEVANT（背景）
    "irrelevant": [
        3, 4, 5, 6, 9, 10, 11, 12, 13,
        20, 23, 26, 27, 28
    ]


}


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = THIS_DIR / "gaze_target_output"


def _build_seg_id_to_label():
    mapping = {}
    for label, seg_ids in SEG_TO_GAZE.items():
        for seg_id in seg_ids:
            mapping[seg_id] = label
    return mapping


SEG_ID_TO_LABEL = _build_seg_id_to_label()


def _normalize_p(value: str) -> str:
    s = str(value).strip().upper()
    if s.startswith("P"):
        s = s[1:]
    if not s.isdigit():
        raise ValueError(f"Invalid participant: {value}")
    return f"P{int(s)}"


def _normalize_s(value: str) -> str:
    s = str(value).strip().upper()
    if s.startswith("S"):
        s = s[1:]
    if not s.isdigit():
        raise ValueError(f"Invalid scenario: {value}")
    return f"S{int(s)}"


def _decode_target(raw_value: str) -> str:
    if raw_value is None:
        return "uncertain"

    token = str(raw_value).strip().lower()
    if token == "":
        return "uncertain"

    if token in SEG_ID_TO_LABEL:
        return SEG_ID_TO_LABEL[token]

    try:
        seg_id = int(float(token))
    except ValueError:
        return "uncertain"

    return SEG_ID_TO_LABEL.get(seg_id, "uncertain")


def _find_input_csv(input_dir: Path, participant: str, scenario: str) -> Path:
    pattern = f"{participant}_{scenario}_*_merged_gaze.csv"
    matches = sorted(input_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No input CSV found for pattern: {pattern} under {input_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple input CSV files matched. Please keep only one file per P/S. "
            f"Matched: {[m.name for m in matches]}"
        )
    return matches[0]


def decode_csv(input_csv: Path, output_csv: Path) -> int:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {input_csv}")
        if "gaze_target" not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required column 'gaze_target': {input_csv}"
            )

        rows = []
        for row in reader:
            raw = row.get("gaze_target", "")
            row["gaze_target_raw"] = raw
            row["gaze_target"] = _decode_target(raw)
            rows.append(row)

    fieldnames = list(reader.fieldnames)
    if "gaze_target_raw" not in fieldnames:
        insert_at = fieldnames.index("gaze_target")
        fieldnames.insert(insert_at, "gaze_target_raw")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode gaze_target class IDs in CSV to prompt labels."
    )
    parser.add_argument("-p", "--participant", required=True, help="Participant, e.g. 2 or P2")
    parser.add_argument("-s", "--scenario", required=True, help="Scenario, e.g. 1 or S1")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing *_merged_gaze.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for *_labelled.csv (default: same as input-dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    participant = _normalize_p(args.participant)
    scenario = _normalize_s(args.scenario)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_csv = _find_input_csv(input_dir, participant, scenario)
    output_csv = output_dir / f"{input_csv.stem}_labelled.csv"

    count = decode_csv(input_csv, output_csv)
    print(f"Input : {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Rows  : {count}")


if __name__ == "__main__":
    main()

