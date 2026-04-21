这个模块将完成把从 seg 图中提取信息，标注这一帧的 gaze target.

流程如下：
1. 把 seg 图和 merged_ui 图进行 one-to-one binding
2. 利用carla自己的color map, 提取这一帧的 label
3. 把过于细碎的 label 对齐到我们目前给出的 ground truth list 中


finder = WhereIsData()

data_color = finder.path_to_required_data("P6", "S1_normal")  # 默认 color
data_raw = finder.path_to_required_data("P6", "S1_normal", seg_type="raw")
data_all = finder.path_to_required_data("P6", "S1_normal", seg_type="all")