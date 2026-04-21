import os
from typing import Dict, List, Optional, Set, Tuple


THIS = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SEG_TYPE_OPTIONS = ("all", "raw", "color")
SCENARIO_LIST = [
	"S1_normal",
	"S2_noWarning",
	"S3_rainy",
	"S4_curve",
	"S5_nighttime",
	"S6_truck",
]


class DataLocationError(Exception):
	"""Base exception for dataset path resolution errors."""


class ParticipantNotFoundError(DataLocationError):
	pass


class ScenarioNotFoundError(DataLocationError):
	pass


class MissingRequiredPathError(DataLocationError):
	pass


class DataAlignmentError(DataLocationError):
	pass


class WhereIsData:
	"""Unified interface to locate segmentation/merged assets for one participant scenario."""

	def __init__(self, data_root: Optional[str] = None) -> None:
		self.data_root = os.path.abspath(data_root or self._resolve_default_data_root())
		if not os.path.isdir(self.data_root):
			raise MissingRequiredPathError(
				f"DATA root does not exist: {self.data_root}. "
				"Please pass data_root explicitly, or set env VLM_WORKZONE_DATA_DIR."
			)

	@staticmethod
	def _resolve_default_data_root() -> str:
		env = os.getenv("VLM_WORKZONE_DATA_DIR")
		if env:
			return env

		candidates = [
			os.path.abspath(os.path.join(THIS, "..", "..", "..", "DATA", "vlm-workzone_data")),
			os.path.abspath(os.path.join(THIS, "..", "..", "..", "DATA")),
		]
		for path in candidates:
			if os.path.isdir(path):
				return path

		return candidates[0]

	def _resolve_participant_dir(self, participant_id: str) -> str:
		direct = os.path.join(self.data_root, participant_id)
		if os.path.isdir(direct):
			return direct

		# Support an extra study-level folder: <data_root>/<study>/<participant>
		for name in sorted(os.listdir(self.data_root)):
			level1 = os.path.join(self.data_root, name)
			if not os.path.isdir(level1):
				continue
			nested = os.path.join(level1, participant_id)
			if os.path.isdir(nested):
				return nested

		raise ParticipantNotFoundError(
			f"Participant '{participant_id}' not found under {self.data_root}"
		)

	@staticmethod
	def _list_images(folder: str) -> List[str]:
		if not os.path.isdir(folder):
			return []
		names = [
			n for n in os.listdir(folder)
			if os.path.isfile(os.path.join(folder, n))
			and n.lower().endswith(IMAGE_EXTENSIONS)
		]
		names.sort()
		return [os.path.join(folder, n) for n in names]

	@staticmethod
	def _pick_latest_carla_dir(scenario_dir: str) -> str:
		carla_dirs = [
			os.path.join(scenario_dir, d)
			for d in os.listdir(scenario_dir)
			if d.startswith("carla_data_") and os.path.isdir(os.path.join(scenario_dir, d))
		]
		if not carla_dirs:
			raise MissingRequiredPathError(
				f"No carla_data_* directory found in {scenario_dir}"
			)

		# Lexicographic order of carla_data_YYYY-MM-DD_HH-MM-SS is chronological.
		carla_dirs.sort()
		return carla_dirs[-1]

	@staticmethod
	def _resolve_merge_log_path(merged_ui_dir: str) -> str:
		preferred_txt = os.path.join(merged_ui_dir, "merge_log.txt")
		if os.path.isfile(preferred_txt):
			return preferred_txt

		# Backward compatibility for historical dumps that may use csv.
		fallback_csv = os.path.join(merged_ui_dir, "merge_log.csv")
		if os.path.isfile(fallback_csv):
			return fallback_csv

		for name in sorted(os.listdir(merged_ui_dir)):
			lower = name.lower()
			if lower.endswith(".txt") and "merge" in lower:
				return os.path.join(merged_ui_dir, name)

		raise MissingRequiredPathError(
			f"No merge log found in {merged_ui_dir}. Expected merge_log.txt."
		)

	@staticmethod
	def _validate_seg_type(seg_type: str) -> str:
		seg_type_normalized = seg_type.strip().lower()
		if seg_type_normalized not in SEG_TYPE_OPTIONS:
			raise ValueError(
				f"Invalid seg_type '{seg_type}'. Supported values: {SEG_TYPE_OPTIONS}"
			)
		return seg_type_normalized

	@staticmethod
	def _filter_segmentation_images(seg_imgs: List[str], seg_type: str) -> List[str]:
		if seg_type == "all":
			return seg_imgs

		keyword = f"segmentation_{seg_type}_"
		return [p for p in seg_imgs if keyword in os.path.basename(p).lower()]

	def list_path_to_required_data(
		self,
		participant_id: str,
		scenario_id: str,
		seg_type: str = "color",
		strict_pairing: bool = False,
	) -> List[Dict[str, str]]:
		"""
		Return list[dict] with keys: seg_img, merged_img, merged_log.
		Also keeps merged_csv for backward compatibility.
		seg_type supports: all/raw/color, default color.
		"""
		seg_type = self._validate_seg_type(seg_type)

		participant_dir = self._resolve_participant_dir(participant_id)
		scenario_dir = os.path.join(participant_dir, scenario_id)
		if not os.path.isdir(scenario_dir):
			raise ScenarioNotFoundError(
				f"Scenario '{scenario_id}' not found for participant '{participant_id}'"
			)

		carla_dir = self._pick_latest_carla_dir(scenario_dir)
		seg_dir = os.path.join(carla_dir, "segmentation")
		if not os.path.isdir(seg_dir):
			raise MissingRequiredPathError(f"Segmentation dir missing: {seg_dir}")

		merged_ui_dir = os.path.join(scenario_dir, "merged_ui")
		if not os.path.isdir(merged_ui_dir):
			raise MissingRequiredPathError(f"Merged UI dir missing: {merged_ui_dir}")

		all_seg_imgs = self._list_images(seg_dir)
		if not all_seg_imgs:
			raise MissingRequiredPathError(f"No segmentation images found in {seg_dir}")

		seg_imgs = self._filter_segmentation_images(all_seg_imgs, seg_type)
		if not seg_imgs:
			raise MissingRequiredPathError(
				f"No segmentation images found for seg_type='{seg_type}' in {seg_dir}"
			)

		merged_img_dir = os.path.join(merged_ui_dir, "imgs")
		if not os.path.isdir(merged_img_dir):
			# Some datasets store merged images directly under merged_ui.
			merged_img_dir = merged_ui_dir

		merged_imgs = self._list_images(merged_img_dir)
		if not merged_imgs:
			raise MissingRequiredPathError(f"No merged images found in {merged_img_dir}")

		merge_log = self._resolve_merge_log_path(merged_ui_dir)

		if strict_pairing and len(seg_imgs) != len(merged_imgs):
			raise DataAlignmentError(
				"Seg/merged image count mismatch: "
				f"seg={len(seg_imgs)}, merged={len(merged_imgs)}"
			)

		pair_count = min(len(seg_imgs), len(merged_imgs))
		result: List[Dict[str, str]] = []
		for i in range(pair_count):
			result.append(
				{
					"seg_img": seg_imgs[i],
					"merged_img": merged_imgs[i],
					"merged_log": merge_log,
					"merged_csv": merge_log,
				}
			)
		return result

	def _extract_participant_scenario_from_path(self, img_path: str) -> Tuple[str, str]:
		"""Extract (participant_id, scenario_id) from an absolute image path."""
		normalized = os.path.abspath(img_path)
		parts = normalized.replace("\\", "/").split("/")

		participant = None
		scenario = None
		for i, token in enumerate(parts):
			if participant is None and token.startswith("P") and token[1:].isdigit():
				if i + 1 < len(parts) and parts[i + 1].startswith("S"):
					participant = token
					scenario = parts[i + 1]
					break

		if not participant or not scenario:
			raise DataLocationError(f"Cannot infer participant/scenario from path: {img_path}")

		return participant, scenario

	def path_to_required_data_by_img_paths(
		self,
		img_paths: List[str],
		seg_type: str = "color",
		strict_pairing: bool = False,
	) -> List[Dict[str, str]]:
		"""
		Batch resolve required data by a list of segmentation image paths.
		This minimizes repeated directory scanning by grouping requests with
		participant/scenario and calling list_path_to_required_data once per group.
		"""
		if not img_paths:
			return []

		seg_type = self._validate_seg_type(seg_type)

		group_keys: Set[Tuple[str, str]] = set()
		for img_path in img_paths:
			participant, scenario = self._extract_participant_scenario_from_path(img_path)
			group_keys.add((participant, scenario))

		lookup: Dict[str, Dict[str, str]] = {}
		for participant, scenario in sorted(group_keys):
			entries = self.list_path_to_required_data(
				participant_id=participant,
				scenario_id=scenario,
				seg_type=seg_type,
				strict_pairing=strict_pairing,
			)
			for entry in entries:
				lookup[os.path.abspath(entry["seg_img"])] = entry

		result: List[Dict[str, str]] = []
		for img_path in img_paths:
			key = os.path.abspath(img_path)
			if key in lookup:
				result.append(lookup[key])

		return result

	def path_to_required_data(
		self,
		participant_id: str,
		scenario_id: str,
		seg_type: str = "color",
		strict_pairing: bool = False,
	) -> List[Dict[str, str]]:
		"""Alias for the unified data path interface requested by callers."""
		return self.list_path_to_required_data(
			participant_id=participant_id,
			scenario_id=scenario_id,
			seg_type=seg_type,
			strict_pairing=strict_pairing,
		)

