# VLM Workzone 标注流程


## 0. 环境准备

在仓库根目录 `D:\VLM_RESEARCH` 下执行：


如果要跑 LLM 标注（`gpt_wz_api_ziqi.py`），先设置 API Key：

```powershell
$env:OPENAI_API_KEY="你的key"
```

## 1. 绑定 segmentation 与 merged UI

脚本：`vlm-workzone/DataPrep/GazeTargetAnnotation/img_binder.py`

作用：把 segmentation mask 路径和 merged UI 路径按时间戳对齐，生成绑定 JSON。

```powershell
python .\vlm-workzone\DataPrep\GazeTargetAnnotation\img_binder.py
```

产物（默认）：

- `vlm-workzone/DataPrep/GazeTargetAnnotation/bind_output/img_mapping.json`

## 2. 语义图初步解码（得到 Carla index）

脚本：`vlm-workzone/DataPrep/GazeTargetAnnotation/color_decoder.py`

作用：根据绑定结果和 gaze 坐标，从语义图中解码每帧 `gaze_target`（Carla class id）。

```powershell
python .\vlm-workzone\DataPrep\GazeTargetAnnotation\color_decoder.py
```

产物目录（默认）：

- `vlm-workzone/DataPrep/GazeTargetAnnotation/gaze_target_output/`

文件命名：

- `P*_S*_***_merged_gaze.csv`

## 3. index -> 自定义 label

脚本：`vlm-workzone/DataPrep/GazeTargetAnnotation/label_docoder.py`

作用：把 `gaze_target` 的 index 映射/聚类到自定义标签体系。

示例（单个 participant/scenario）：

```powershell
python .\vlm-workzone\DataPrep\GazeTargetAnnotation\label_docoder.py -p P2 -s S1
```

产物命名：

- `P*_S*_***_merged_gaze_labelled.csv`

## 4. 按 3 段 workzone 时间切片

脚本：`vlm-workzone/DataPrep/LLMAnnotation/data_slicer.py`

作用：按 `workzone_driving_data.xlsx` 中的 3 个 workzone 时间段切片，并按 fps 采样。

```powershell
python .\vlm-workzone\DataPrep\LLMAnnotation\data_slicer.py --fps 2.0
```

可选：只处理一个参与者

```powershell
python .\vlm-workzone\DataPrep\LLMAnnotation\data_slicer.py --participant P2 --fps 2.0
```

产物命名：

- `P*_S*_***_merged_gaze_labelled_sliced.csv`

## 5. GPT 标注（run / build+submit+collect）

脚本：`vlm-workzone/DataPrep/LLMAnnotation/gpt_wz_api_ziqi.py`

支持两类流程：

- `run`：实时调用接口，直接出结果。
- `build + submit + collect`：Batch API 流程，适合批量任务。

```

产物目录（默认）：

- `vlm-workzone/DataPrep/LLMAnnotation/annotation_outputs/workzone_gpt/`

## In a Nutshell

`merged与seg绑定 -> 解码语义idx -> idx cluster到我们的label -> 切出需要的帧 -> LLM标注`
`img_binder -> color_decoder -> label_docoder -> data_slicer -> gpt_wz_api_ziqi`
