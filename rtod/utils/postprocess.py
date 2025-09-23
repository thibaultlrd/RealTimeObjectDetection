from typing import List, Tuple

import numpy as np


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
	x1 = x - w / 2.0
	y1 = y - h / 2.0
	x2 = x + w / 2.0
	y2 = y + h / 2.0
	return x1, y1, x2, y2


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
	if boxes.size == 0:
		return []
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	areas = (x2 - x1) * (y2 - y1)
	order = scores.argsort()[::-1]

	keep: List[int] = []
	while order.size > 0:
		i = order[0]
		keep.append(int(i))
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1)
		h = np.maximum(0.0, yy2 - yy1)
		inter = w * h
		iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
		inds = np.where(iou <= iou_threshold)[0]
		order = order[inds + 1]
	return keep
