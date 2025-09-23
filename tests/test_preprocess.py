import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rtod.utils.postprocess import nms  # noqa: E402


def test_nms():
	boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
	scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)
	keep = nms(boxes, scores, 0.5)
	assert isinstance(keep, list)
	assert len(keep) >= 2
