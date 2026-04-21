# 这个代码负责把 segmentation mask 和 merged ui 进行绑定，
# 生成一个 JSON 文件，按 participant -> scenario 分组，
# 每个 scenario 下有一个 mapping 字典：
# key 是 segmentation mask 的路径，value 是 merged ui 的路径。

'''
保存 JSON 文件格式如下：

{
    "participants": {
        "P2": {
            "scenarios": {
                "S1_normal": {
                    "mapping": {
                        "segmentation_mask_path_1": "merged_ui_path_1",
                        "segmentation_mask_path_2": "merged_ui_path_2"
                    }
                },
                "S2_noWarning": {
                    "mapping": {
                        "segmentation_mask_path_3": "merged_ui_path_3"
                    }
                }
            }
        },
        "P3": {
            "scenarios": {
                "S1_normal": {
                    "mapping": {
                        "segmentation_mask_path_4": "merged_ui_path_4"
                    }
                }
            }
        }
    }
}
'''
import json
import os
import re
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from where_is_data import SCENARIO_LIST, DataLocationError, WhereIsData


THIS = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS, "bind_output")
os.makedirs(OUT_DIR, exist_ok=True)


'''
运行方式：
1. 利用where is data的接口，遍历所有 participant 和 scenario，获取 segmentation mask img 和 merged ui txt 中的 img 路径。
2. 对于每个 participant 和 scenario，调用 ImgBinder 的 add_mapping 方法，把 segmentation mask 路径和 merged ui 路径进行绑定，存储在一个字典里。
3. 最后把整个数据结构保存成一个 JSON 文件，命名为 img_mapping.json，保存在 OUT_DIR 目录下。

格式：
segmentation mask 文件名： "segmentation_color_001610_1764105334772.png"
split("_")后，第二个是color还是raw，第4个是时间戳（1764105334772），可以用来匹配 merged ui txt中的时间戳。

txt 里每一行数据都有形如 1764105164.643000 这样的时间戳，可以把它乘以1000，得到毫秒级的时间戳（1764105164643），然后和 segmentation mask 的时间戳进行匹配，找到对应的 merged ui img 路径。

然后把类似于 merged_1764105164.643000.png 和 segmentation_color_000000_1764105164643.png 绑在一起

你可以调控时间戳的匹配精度，比如允许误差在100ms以内，这样可以更灵活地匹配到对应的 merged ui img。

'''


class ImgBinder:
    SEG_TS_PATTERN = re.compile(r"^segmentation_(raw|color)_\d+_(\d+)\.[^.]+$", re.IGNORECASE)
    MERGED_TS_PATTERN = re.compile(r"^merged_(\d+\.\d+)\.[^.]+$", re.IGNORECASE)

    def __init__(
        self,
        data_root: Optional[str] = None,
        seg_type: str = "raw",
        tolerance_ms: int = 100,
    ):
        self.where = WhereIsData(data_root=data_root)
        self.seg_type = seg_type
        self.tolerance_ms = tolerance_ms
        self.data = {"participants": {}}
        self.errors: List[Dict[str, str]] = []

    def add_mapping(self, participant: str, scenario: str, seg_mask_path: str, merged_ui_path: str):
        if participant not in self.data["participants"]:
            self.data["participants"][participant] = {"scenarios": {}}
        if scenario not in self.data["participants"][participant]["scenarios"]:
            self.data["participants"][participant]["scenarios"][scenario] = {"mapping": {}}

        self.data["participants"][participant]["scenarios"][scenario]["mapping"][seg_mask_path] = merged_ui_path

    def get_data(self):
        return self.data

    @staticmethod
    def _to_ms(ts_str: str) -> int:
        return int(round(float(ts_str) * 1000))

    @staticmethod
    def _is_image(path: str) -> bool:
        lower = path.lower()
        return lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))

    def _parse_seg_timestamp_ms(self, seg_path: str) -> int:
        name = os.path.basename(seg_path)
        match = self.SEG_TS_PATTERN.match(name)
        if not match:
            raise ValueError(f"Invalid segmentation filename format: {name}")
        return int(match.group(2))

    def _list_seg_images(self, seg_dir: str) -> List[str]:
        names = sorted(os.listdir(seg_dir))
        images = [
            os.path.join(seg_dir, n)
            for n in names
            if os.path.isfile(os.path.join(seg_dir, n)) and self._is_image(n)
        ]

        if self.seg_type == "all":
            return images

        keyword = f"segmentation_{self.seg_type}_"
        return [p for p in images if keyword in os.path.basename(p).lower()]

    def _build_merged_image_index(self, merged_dir: str) -> Tuple[List[int], Dict[int, str]]:
        ts_list: List[int] = []
        ts_to_path: Dict[int, str] = {}

        for name in sorted(os.listdir(merged_dir)):
            full = os.path.join(merged_dir, name)
            if not os.path.isfile(full) or not self._is_image(name):
                continue
            match = self.MERGED_TS_PATTERN.match(name)
            if not match:
                continue
            ms = self._to_ms(match.group(1))
            ts_list.append(ms)
            ts_to_path[ms] = full

        return ts_list, ts_to_path

    def _resolve_nearest(self, target_ms: int, sorted_ms: List[int]) -> Tuple[Optional[int], Optional[int]]:
        if not sorted_ms:
            return None, None

        pos = bisect_left(sorted_ms, target_ms)
        candidates: List[int] = []
        if pos < len(sorted_ms):
            candidates.append(sorted_ms[pos])
        if pos > 0:
            candidates.append(sorted_ms[pos - 1])

        best_ms = min(candidates, key=lambda v: abs(v - target_ms))
        return best_ms, abs(best_ms - target_ms)

    def _timeline_from_merge_log(self, merge_log_path: str, merged_dir: str) -> List[Tuple[int, str]]:
        merged_ms_list, merged_index = self._build_merged_image_index(merged_dir)
        merged_ms_list = sorted(set(merged_ms_list))
        if not merged_ms_list:
            raise FileNotFoundError(f"No merged image found in {merged_dir}")

        timeline: List[Tuple[int, str]] = []
        seen = set()

        with open(merge_log_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                raise ValueError(f"Empty merge log: {merge_log_path}")

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                ts_str = parts[0].strip()
                try:
                    gaze_ms = self._to_ms(ts_str)
                except ValueError:
                    continue

                nearest_ms, _ = self._resolve_nearest(gaze_ms, merged_ms_list)
                if nearest_ms is None:
                    continue

                if nearest_ms in seen:
                    continue
                seen.add(nearest_ms)
                timeline.append((nearest_ms, merged_index[nearest_ms]))

        if timeline:
            timeline.sort(key=lambda x: x[0])
            return timeline

        # Fallback: use merged images directly if log parsing fails.
        return [(ms, merged_index[ms]) for ms in merged_ms_list]

    def _find_match_from_timeline(
        self,
        seg_ts_ms: int,
        merged_timeline: List[Tuple[int, str]],
    ) -> Optional[str]:
        merged_ts_sorted = [ts for ts, _ in merged_timeline]
        nearest_ms, delta = self._resolve_nearest(seg_ts_ms, merged_ts_sorted)
        if nearest_ms is None or delta is None or delta > self.tolerance_ms:
            return None

        for ts, path in merged_timeline:
            if ts == nearest_ms:
                return path
        return None

    def bind_one(self, participant: str, scenario: str) -> int:
        entries = self.where.path_to_required_data(
            participant_id=participant,
            scenario_id=scenario,
            seg_type=self.seg_type,
            strict_pairing=False,
        )
        if not entries:
            return 0

        seg_dir = os.path.dirname(entries[0]["seg_img"])
        merged_dir = os.path.dirname(entries[0]["merged_img"])
        merge_log_path = entries[0]["merged_log"]

        seg_images = self._list_seg_images(seg_dir)
        if not seg_images:
            raise FileNotFoundError(f"No seg images for seg_type='{self.seg_type}' in {seg_dir}")

        merged_timeline = self._timeline_from_merge_log(merge_log_path, merged_dir)

        mapped_count = 0
        for seg_path in tqdm(
            seg_images,
            desc=f"{participant}/{scenario} seg",
            leave=False,
        ):
            try:
                seg_ts_ms = self._parse_seg_timestamp_ms(seg_path)
            except ValueError:
                continue

            merged_path = self._find_match_from_timeline(seg_ts_ms, merged_timeline)
            if merged_path is None:
                continue

            self.add_mapping(participant, scenario, seg_path, merged_path)
            mapped_count += 1

        return mapped_count

    def _discover_participants(self) -> List[str]:
        participants = set()

        for name in sorted(os.listdir(self.where.data_root)):
            p1 = os.path.join(self.where.data_root, name)
            if not os.path.isdir(p1):
                continue

            if name.startswith("P"):
                participants.add(name)

            # Support one extra nesting level.
            for sub in sorted(os.listdir(p1)):
                p2 = os.path.join(p1, sub)
                if os.path.isdir(p2) and sub.startswith("P"):
                    participants.add(sub)

        return sorted(participants)

    def bind_all(
        self,
        participants: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        continue_on_error: bool = True,
    ) -> Dict[str, int]:
        participants = participants or self._discover_participants()
        scenarios = scenarios or SCENARIO_LIST

        mapped_total = 0
        total_tasks = len(participants) * len(scenarios)
        task_iter = ((participant, scenario) for participant in participants for scenario in scenarios)
        for participant, scenario in tqdm(task_iter, total=total_tasks, desc="Binding scenarios"):
            try:
                mapped_total += self.bind_one(participant, scenario)
            except (DataLocationError, OSError, ValueError) as e:
                self.errors.append(
                    {
                        "participant": participant,
                        "scenario": scenario,
                        "error": str(e),
                    }
                )
                if not continue_on_error:
                    raise

        return {
            "participants": len(participants),
            "scenarios_per_participant": len(scenarios),
            "mapped_pairs": mapped_total,
            "error_count": len(self.errors),
        }

    def save_json(self, output_path: Optional[str] = None) -> str:
        output_path = output_path or os.path.join(OUT_DIR, "img_mapping.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        return output_path


def main():
    binder = ImgBinder(seg_type="color", tolerance_ms=100)
    report = binder.bind_all(continue_on_error=True)
    out_path = binder.save_json()

    print("Bind finished")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Output: {out_path}")
    if binder.errors:
        print("Errors (first 10):")
        print(json.dumps(binder.errors[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()