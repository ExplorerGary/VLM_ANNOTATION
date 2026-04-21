#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

API_SCRIPT="${REPO_ROOT}/DataPrep/LLMAnnotation/gpt_wz_api_ziqi.py"
DECODER_SCRIPT="${REPO_ROOT}/DataPrep/GazeTargetAnnotation/label_docoder.py"
GAZE_DIR="${REPO_ROOT}/DataPrep/GazeTargetAnnotation/gaze_target_output"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-batch}"
PARTICIPANT_RANGE="${2:-2-34}"
SCENARIO_RANGE="${3:-1-6}"

DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/DATA/vlm-workzone_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/annotation_outputs/workzone_gpt}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DETAIL="${DETAIL:-high}"
TIMESTAMP_TOLERANCE="${TIMESTAMP_TOLERANCE:-0.20}"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_FRAMES="${MAX_FRAMES:-}"

if [[ -z "${OPENAI_API_KEY:-}" && -f "${WORKSPACE_ROOT}/OPENAI_API_KEY.txt" ]]; then
	export OPENAI_API_KEY="$(tr -d '\r\n' < "${WORKSPACE_ROOT}/OPENAI_API_KEY.txt")"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
	echo "OPENAI_API_KEY is not set and ${WORKSPACE_ROOT}/OPENAI_API_KEY.txt was not found." >&2
	exit 1
fi

mkdir -p "${OUTPUT_DIR}"

usage() {
	cat <<EOF
Usage: $(basename "$0") [run|batch] [participant_range] [scenario_range]

Defaults:
	mode              batch
	participant_range 2-34
	scenario_range    1-6

Environment overrides:
	PYTHON_BIN DATA_ROOT OUTPUT_DIR IMAGE_SIZE MAX_TOKENS DETAIL TIMESTAMP_TOLERANCE CONCURRENCY MAX_FRAMES
EOF
}

expand_range() {
	local range="$1"
	local start="${range%-*}"
	local end="${range#*-}"

	if [[ ! "${start}" =~ ^[0-9]+$ || ! "${end}" =~ ^[0-9]+$ || "${start}" -gt "${end}" ]]; then
		echo "Invalid range: ${range}" >&2
		exit 1
	fi

	local value="${start}"
	while [[ "${value}" -le "${end}" ]]; do
		printf '%s\n' "${value}"
		value=$((value + 1))
	done
}

decode_labels() {
	local participant="$1"
	local scenario="$2"

	"${PYTHON_BIN}" "${DECODER_SCRIPT}" -p "${participant}" -s "${scenario}" --input-dir "${GAZE_DIR}" --output-dir "${GAZE_DIR}"
}

common_api_args=(
	--data-root "${DATA_ROOT}"
	--gaze-csv-dir "${GAZE_DIR}"
	--output-dir "${OUTPUT_DIR}"
	--image-size "${IMAGE_SIZE}"
	--max-tokens "${MAX_TOKENS}"
	--detail "${DETAIL}"
	--timestamp-tolerance "${TIMESTAMP_TOLERANCE}"
)

if [[ -n "${MAX_FRAMES}" ]]; then
	common_api_args+=(--max-frames "${MAX_FRAMES}")
fi

run_realtime() {
	local participant="$1"
	local scenario="$2"
	"${PYTHON_BIN}" "${API_SCRIPT}" run -p "${participant}" -s "${scenario}" "${common_api_args[@]}" --concurrency "${CONCURRENCY}"
}

build_batch() {
	local participant="$1"
	local scenario="$2"
	"${PYTHON_BIN}" "${API_SCRIPT}" build -p "${participant}" -s "${scenario}" "${common_api_args[@]}"
}

latest_match() {
	local pattern="$1"
	local -a matches=()
	local match

	while IFS= read -r match; do
		matches+=("${match}")
	done < <(compgen -G "${pattern}" || true)

	if (( ${#matches[@]} == 0 )); then
		return 1
	fi

	printf '%s\n' "${matches[@]}" | sort | tail -n 1
}

submit_batch() {
	local participant="$1"
	local scenario="$2"
	local batch_input batch_manifest

	batch_input="$(latest_match "${OUTPUT_DIR}/batch_requests_${participant}_${scenario}_*.jsonl" || true)"
	batch_manifest="$(latest_match "${OUTPUT_DIR}/batch_targets_${participant}_${scenario}_*.json" || true)"

	if [[ -z "${batch_input}" ]]; then
		echo "[skip] No batch request file found for ${participant}/${scenario}." >&2
		return 1
	fi

	if [[ -n "${batch_manifest}" ]]; then
		"${PYTHON_BIN}" "${API_SCRIPT}" submit --batch-input "${batch_input}" --manifest "${batch_manifest}" --output-dir "${OUTPUT_DIR}"
	else
		"${PYTHON_BIN}" "${API_SCRIPT}" submit --batch-input "${batch_input}" --output-dir "${OUTPUT_DIR}"
	fi
}

case "${MODE}" in
	run|batch)
		;;
	-h|--help|help)
		usage
		exit 0
		;;
	*)
		usage >&2
		exit 1
		;;
esac

for participant_index in $(expand_range "${PARTICIPANT_RANGE}"); do
	participant="P${participant_index}"

	for scenario_index in $(expand_range "${SCENARIO_RANGE}"); do
		scenario="S${scenario_index}"

		echo "== ${participant}/${scenario} =="

		if ! decode_labels "${participant_index}" "${scenario_index}"; then
			echo "[warn] label decoder did not complete for ${participant}/${scenario}; continuing." >&2
		fi

		if [[ "${MODE}" == "run" ]]; then
			if ! run_realtime "${participant}" "${scenario}"; then
				echo "[warn] run failed for ${participant}/${scenario}." >&2
			fi
		else
			if ! build_batch "${participant}" "${scenario}"; then
				echo "[warn] build failed for ${participant}/${scenario}." >&2
				continue
			fi

			if ! submit_batch "${participant}" "${scenario}"; then
				echo "[warn] submit failed for ${participant}/${scenario}." >&2
			fi
		fi
	done
done
