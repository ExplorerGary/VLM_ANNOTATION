import json
import os
import re
import sys
import time
from bisect import bisect_left
from datetime import datetime

MULTI_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(MULTI_THIS_DIR, "..", "DataPrep", "LLMAnnotation"))
if BASE_DIR not in sys.path:
	sys.path.insert(0, BASE_DIR)

from gpt_wz_api_ziqi import *  # noqa: F401,F403
import gpt_wz_api_ziqi as base


DEFAULT_MULTI_OUTPUT_DIR = os.path.join(MULTI_THIS_DIR, "gpt_result")
DEFAULT_PAIRING_DIR = os.path.join(MULTI_THIS_DIR, "multi-binding-result")
DEFAULT_EMERGENCY_TOLERANCE = 0.12

MERGED_NAME_PATTERN = re.compile(r"^merged_(\d+(?:\.\d+)?)\.[^.]+$", re.IGNORECASE)
FRONT_NAME_PATTERN = re.compile(r"^camera_(\d+)\.[^.]+$", re.IGNORECASE)


def _parse_ts_from_merged_name(name: str):
	m = MERGED_NAME_PATTERN.match(name)
	return float(m.group(1)) if m else None


def _parse_ts_from_front_name(name: str):
	m = FRONT_NAME_PATTERN.match(name)
	if not m:
		return None
	return int(m.group(1)) / 1000.0


def _resolve_pairing_json_path(participant: str, scenario_full: str) -> str:
	direct = os.path.join(
		DEFAULT_PAIRING_DIR,
		f"{participant}_{scenario_full}_front_merged_pairing.json",
	)
	if os.path.exists(direct):
		return direct

	prefix = f"{participant}_{scenario_full}_front_merged_pairing"
	candidates = [
		os.path.join(DEFAULT_PAIRING_DIR, n)
		for n in sorted(os.listdir(DEFAULT_PAIRING_DIR))
		if n.startswith(prefix) and n.endswith(".json")
	]
	if candidates:
		return candidates[0]

	raise FileNotFoundError(
		f"Pairing json not found for {participant}/{scenario_full} under {DEFAULT_PAIRING_DIR}"
	)


def _load_pairing_payload(participant: str, scenario_full: str) -> dict:
	if not os.path.isdir(DEFAULT_PAIRING_DIR):
		raise FileNotFoundError(f"Pairing directory not found: {DEFAULT_PAIRING_DIR}")
	path = _resolve_pairing_json_path(participant, scenario_full)
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _build_pairing_indices(payload: dict):
	exact_merged_to_front = {}
	front_ts_sorted = []
	front_ts_to_path = {}

	pairs = payload.get("pairs", [])
	pairs_sorted = sorted(pairs, key=lambda x: x.get("time_diff_ms", 10**9))

	for item in pairs_sorted:
		merged_path = item.get("merged_ui", "")
		front_path = item.get("front_camera", "")
		if not merged_path or not front_path:
			continue

		merged_name = os.path.basename(merged_path)
		if merged_name not in exact_merged_to_front:
			exact_merged_to_front[merged_name] = front_path

		front_name = os.path.basename(front_path)
		front_ts = _parse_ts_from_front_name(front_name)
		if front_ts is None:
			continue
		front_ts_to_path[front_ts] = front_path

	for item in payload.get("unmatched_front", []):
		front_path = item.get("front_camera", "")
		if not front_path:
			continue
		front_ts = item.get("front_timestamp_ms")
		if isinstance(front_ts, int):
			ts = front_ts / 1000.0
		else:
			ts = _parse_ts_from_front_name(os.path.basename(front_path))
		if ts is None:
			continue
		if ts not in front_ts_to_path:
			front_ts_to_path[ts] = front_path

	front_ts_sorted = sorted(front_ts_to_path.keys())
	return exact_merged_to_front, front_ts_sorted, front_ts_to_path


def _resolve_nearest(target: float, sorted_values: list[float]):
	if not sorted_values:
		return None, None
	pos = bisect_left(sorted_values, target)
	candidates = []
	if pos < len(sorted_values):
		candidates.append(sorted_values[pos])
	if pos > 0:
		candidates.append(sorted_values[pos - 1])
	if not candidates:
		return None, None
	best = min(candidates, key=lambda v: abs(v - target))
	return best, abs(best - target)


def _resolve_front_image_for_merged(
	merged_image_path: str,
	merged_ts: float,
	exact_merged_to_front: dict,
	front_ts_sorted: list[float],
	front_ts_to_path: dict,
	emergency_tolerance: float,
):
	merged_name = os.path.basename(merged_image_path)

	direct = exact_merged_to_front.get(merged_name)
	if direct and os.path.exists(direct):
		return direct, "paired_json", abs(_parse_ts_from_front_name(os.path.basename(direct)) - merged_ts)

	nearest_ts, delta = _resolve_nearest(merged_ts, front_ts_sorted)
	if nearest_ts is not None and delta is not None and delta <= emergency_tolerance:
		front_path = front_ts_to_path.get(nearest_ts)
		if front_path and os.path.exists(front_path):
			return front_path, "emergency_nearest", delta

	return "", "missing", None


def _resolve_targets(
	participant: str,
	scenario_full: str,
	data_root: str | None,
	gaze_csv_path,
	tolerance: float,
	max_frames: int | None,
):
	targets = base._resolve_targets(
		participant=participant,
		scenario_full=scenario_full,
		data_root=data_root,
		gaze_csv_path=gaze_csv_path,
		tolerance=tolerance,
		max_frames=max_frames,
	)

	payload = _load_pairing_payload(participant, scenario_full)
	exact_merged_to_front, front_ts_sorted, front_ts_to_path = _build_pairing_indices(payload)

	emergency_tolerance = float(os.getenv("MULTI_EMERGENCY_TOLERANCE", str(DEFAULT_EMERGENCY_TOLERANCE)))
	enriched = []
	for t in targets:
		front_path, pair_source, emergency_delta = _resolve_front_image_for_merged(
			merged_image_path=t["image_path"],
			merged_ts=t["timestamp"],
			exact_merged_to_front=exact_merged_to_front,
			front_ts_sorted=front_ts_sorted,
			front_ts_to_path=front_ts_to_path,
			emergency_tolerance=emergency_tolerance,
		)

		if not front_path:
			front_path = t["image_path"]
			pair_source = "fallback_use_merged"

		item = dict(t)
		item["merged_image_path"] = t["image_path"]
		item["front_image_path"] = front_path
		item["pair_source"] = pair_source
		if emergency_delta is not None:
			item["emergency_delta_sec"] = emergency_delta
		enriched.append(item)

	return enriched


def _make_request(
	custom_id: str,
	model: str,
	system_msg: str,
	user_msg: str,
	b64_merged: str,
	b64_front: str,
	detail: str,
	max_tokens: int,
	json_mode: bool,
):
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
							"url": f"data:image/jpeg;base64,{b64_front}",
							"detail": detail,
						},
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{b64_merged}",
							"detail": detail,
						},
					},
					{
						"type": "text",
						"text": "Image#1 is front camera view, Image#2 is merged UI frame.\n" + user_msg,
					},
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


def _call_api(
	client,
	model,
	system_msg,
	user_msg,
	b64_merged,
	b64_front,
	detail,
	max_tokens,
	json_mode=True,
):
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
							"url": f"data:image/jpeg;base64,{b64_front}",
							"detail": detail,
						},
					},
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{b64_merged}",
							"detail": detail,
						},
					},
					{
						"type": "text",
						"text": "Image#1 is front camera view, Image#2 is merged UI frame.\n" + user_msg,
					},
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


def cmd_build(args):
	participant = base._normalize_participant(args.participant)
	gaze_csv_path, scenario_full = base._resolve_labelled_csv_from_args(args, participant)

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
	print(f"Resolved {len(targets)} multi targets (front + merged).")

	freedom_prompt = base.LLMPrompt(seed="FREEDOM")
	structured_prompt = base.LLMPrompt(seed="STRUCTURED")

	os.makedirs(args.output_dir, exist_ok=True)
	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	prefix = f"{participant}_{scenario_full}_{run_ts}"
	out_jsonl = os.path.join(args.output_dir, f"batch_requests_{prefix}.jsonl")
	manifest_path = os.path.join(args.output_dir, f"batch_targets_{prefix}.json")

	total = 0
	with open(out_jsonl, "w", encoding="utf-8") as f:
		for i, t in enumerate(targets):
			print(
				f"  Encoding frame {i+1}/{len(targets)}: front={t['front_image_path']} merged={t['merged_image_path']}",
				end="\r",
			)
			b64_merged = base.encode_image(t["merged_image_path"], args.image_size)
			b64_front = base.encode_image(t["front_image_path"], args.image_size)

			for mode, prompt in [("freedom", freedom_prompt), ("structured", structured_prompt)]:
				cid = base._make_custom_id(mode, t)
				request = _make_request(
					custom_id=cid,
					model=args.model,
					system_msg=prompt.system_message,
					user_msg=base._render_user_message(prompt.user_message, t["gaze_target"]),
					b64_merged=b64_merged,
					b64_front=b64_front,
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
	print(f"  python multi_gpt_wz_api_ziqi.py submit --batch-input {out_jsonl} --manifest {manifest_path}")


def cmd_collect(args):
	client = base.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

	manifest_path = args.manifest or base._find_manifest_from_batch_info(args.output_dir, args.batch_id)
	target_map = base._load_manifest_map(manifest_path)

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
		meta = base.parse_custom_id(cid)
		k = (meta["participant"], meta["scenario"], meta["timestamp"])

		if k not in merged:
			t = target_map.get(k, {})
			merged[k] = {
				"participant": meta["participant"],
				"scenario": meta["scenario"],
				"scenario_prefix": meta["scenario"].split("_", 1)[0],
				"timestamp": meta["timestamp"],
				"image_path": t.get("image_path", ""),
				"merged_image_path": t.get("merged_image_path", t.get("image_path", "")),
				"front_image_path": t.get("front_image_path", ""),
				"pair_source": t.get("pair_source", ""),
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


def cmd_run(args):
	from concurrent.futures import ThreadPoolExecutor, as_completed

	participant = base._normalize_participant(args.participant)
	gaze_csv_path, scenario_full = base._resolve_labelled_csv_from_args(args, participant)

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

	print(f"Resolved {len(targets)} multi targets (front + merged).")
	print(f"Total API calls: {len(targets) * 2} (freedom + structured)")

	client = base.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
	freedom_prompt = base.LLMPrompt(seed="FREEDOM")
	structured_prompt = base.LLMPrompt(seed="STRUCTURED")

	tasks = []
	for t in targets:
		for mode, prompt in [("freedom", freedom_prompt), ("structured", structured_prompt)]:
			tasks.append((t, mode, prompt))

	results_map = {}
	done = 0
	total = len(tasks)

	def _process(task):
		t, mode, prompt = task
		b64_merged = base.encode_image(t["merged_image_path"], args.image_size)
		b64_front = base.encode_image(t["front_image_path"], args.image_size)
		user_msg = base._render_user_message(prompt.user_message, t["gaze_target"])
		text = _call_api(
			client,
			args.model,
			prompt.system_message,
			user_msg,
			b64_merged,
			b64_front,
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
					"merged_image_path": t["merged_image_path"],
					"front_image_path": t["front_image_path"],
					"pair_source": t["pair_source"],
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


if __name__ == "__main__":
	args = base.parse_args()

	if getattr(args, "output_dir", None) == base.DEFAULT_OUTPUT_DIR:
		args.output_dir = DEFAULT_MULTI_OUTPUT_DIR

	if args.command == "build":
		cmd_build(args)
	elif args.command == "submit":
		base.cmd_submit(args)
	elif args.command == "collect":
		cmd_collect(args)
	elif args.command == "run":
		cmd_run(args)


