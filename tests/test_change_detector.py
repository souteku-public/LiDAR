"""
change_detector のユニットテスト
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.change_detector import ChangeDetector


def _pts(*xyzis) -> np.ndarray:
    """(x,y,z,i) タプルのリストから float32 (N,4) 配列を作る"""
    return np.array(xyzis, dtype=np.float32)


class TestChangeDetector:

    def test_first_frame_returns_all_points(self):
        """初回フレームは全点が返される"""
        det = ChangeDetector(voxel_size=0.1)
        pts = _pts((0.0, 0.0, 0.0, 100), (1.0, 0.0, 0.0, 200))
        changed, is_first = det.detect(pts)
        assert is_first is True
        assert changed.shape[0] == 2

    def test_identical_frame_returns_no_points(self):
        """同一フレームでは変化点なし"""
        det = ChangeDetector(voxel_size=0.1)
        pts = _pts((0.0, 0.0, 0.0, 100))
        det.detect(pts)                          # 初回
        changed, is_first = det.detect(pts)      # 同一フレーム
        assert is_first is False
        assert changed.shape[0] == 0

    def test_new_point_detected(self):
        """前フレームにない新規ボクセルの点が検出される"""
        det = ChangeDetector(voxel_size=0.1)
        pts_a = _pts((0.0, 0.0, 0.0, 100))
        pts_b = _pts((0.0, 0.0, 0.0, 100), (5.0, 5.0, 5.0, 200))  # 新規点追加
        det.detect(pts_a)
        changed, _ = det.detect(pts_b)
        assert changed.shape[0] >= 1
        # 新規点 (5.0, 5.0, 5.0) が含まれる
        assert any(np.allclose(r[:3], [5.0, 5.0, 5.0]) for r in changed)

    def test_removed_point_detected(self):
        """前フレームから消えた点が検出される"""
        det = ChangeDetector(voxel_size=0.1)
        pts_a = _pts((0.0, 0.0, 0.0, 100), (5.0, 5.0, 5.0, 200))
        pts_b = _pts((0.0, 0.0, 0.0, 100))  # (5,5,5) が消えた
        det.detect(pts_a)
        changed, _ = det.detect(pts_b)
        assert changed.shape[0] >= 1
        assert any(np.allclose(r[:3], [5.0, 5.0, 5.0]) for r in changed)

    def test_voxel_size_coarser(self):
        """粗いボクセルサイズでは近傍の動きを変化と検出しない"""
        det = ChangeDetector(voxel_size=1.0)
        pts_a = _pts((0.0, 0.0, 0.0, 100))
        pts_b = _pts((0.3, 0.3, 0.3, 100))  # 同じボクセル内
        det.detect(pts_a)
        changed, _ = det.detect(pts_b)
        assert changed.shape[0] == 0

    def test_voxel_size_finer(self):
        """細かいボクセルサイズでは 0.3m 移動でも変化と検出する"""
        det = ChangeDetector(voxel_size=0.1)
        pts_a = _pts((0.0, 0.0, 0.0, 100))
        pts_b = _pts((0.3, 0.3, 0.3, 100))
        det.detect(pts_a)
        changed, _ = det.detect(pts_b)
        assert changed.shape[0] > 0

    def test_reset_clears_state(self):
        """reset() 後は初回フレーム扱いになる"""
        det = ChangeDetector(voxel_size=0.1)
        pts = _pts((0.0, 0.0, 0.0, 100))
        det.detect(pts)     # 初回
        det.detect(pts)     # 2 回目 (変化なし)
        det.reset()
        changed, is_first = det.detect(pts)
        assert is_first is True
        assert changed.shape[0] == 1

    def test_empty_frame(self):
        """空フレームでもクラッシュしない"""
        det = ChangeDetector(voxel_size=0.1)
        empty = np.empty((0, 4), dtype=np.float32)
        det.detect(empty)
        changed, _ = det.detect(empty)
        assert changed.shape[0] == 0
