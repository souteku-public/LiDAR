"""
pcd_writer のユニットテスト
"""
import sys
import os
import struct
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pcd_writer import PCDWriter


def _read_pcd(filepath: str):
    """PCD ファイルを読み込んで (header_dict, points_ndarray) を返す簡易パーサー"""
    with open(filepath, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "DATA binary":
                break
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]
        binary = f.read()

    n = int(header["POINTS"][0])
    # 4 float32 fields = 16 bytes per point
    points = np.frombuffer(binary, dtype=np.float32).reshape(n, 4)
    return header, points


class TestPCDWriter:

    def test_file_created(self, tmp_path):
        writer = PCDWriter(tmp_path)
        pts = np.array([[1.0, 2.0, 3.0, 128.0]], dtype=np.float32)
        path = writer.write(frame_id=1, timestamp=1000.0, points=pts)
        assert path != ""
        assert os.path.exists(path)

    def test_filename_format(self, tmp_path):
        writer = PCDWriter(tmp_path)
        pts = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        path = writer.write(frame_id=42, timestamp=1.5, points=pts)
        fname = os.path.basename(path)
        assert fname.startswith("frame_00000042_")
        assert fname.endswith(".pcd")

    def test_header_fields(self, tmp_path):
        writer = PCDWriter(tmp_path)
        pts = np.array([[1.0, 2.0, 3.0, 200.0],
                        [4.0, 5.0, 6.0, 100.0]], dtype=np.float32)
        path = writer.write(frame_id=1, timestamp=0.0, points=pts)
        header, _ = _read_pcd(path)
        assert header["FIELDS"] == ["x", "y", "z", "intensity"]
        assert header["POINTS"] == ["2"]
        assert header["WIDTH"]  == ["2"]
        assert header["HEIGHT"] == ["1"]
        assert header["TYPE"]   == ["F", "F", "F", "F"]

    def test_point_values_preserved(self, tmp_path):
        writer = PCDWriter(tmp_path)
        original = np.array(
            [[1.5, -2.5, 3.5, 128.0],
             [0.0,  0.0, 0.0,   0.0]],
            dtype=np.float32,
        )
        path = writer.write(frame_id=1, timestamp=0.0, points=original)
        _, recovered = _read_pcd(path)
        np.testing.assert_allclose(recovered, original, rtol=1e-6)

    def test_empty_points_skipped(self, tmp_path):
        writer = PCDWriter(tmp_path)
        empty = np.empty((0, 4), dtype=np.float32)
        path = writer.write(frame_id=1, timestamp=0.0, points=empty)
        assert path == ""
        assert len(list(tmp_path.iterdir())) == 0

    def test_file_count(self, tmp_path):
        writer = PCDWriter(tmp_path)
        pts = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        for i in range(5):
            writer.write(frame_id=i, timestamp=float(i), points=pts)
        assert writer.file_count == 5

    def test_reset_changes_output_dir(self, tmp_path):
        writer = PCDWriter(tmp_path / "dir_a")
        pts = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        writer.write(1, 0.0, pts)
        assert writer.file_count == 1

        writer.reset(tmp_path / "dir_b")
        assert writer.file_count == 0
        writer.write(2, 0.0, pts)
        assert writer.file_count == 1
        assert (tmp_path / "dir_b").exists()

    def test_large_point_cloud(self, tmp_path):
        """大きな点群でも正しく保存・復元できる"""
        writer = PCDWriter(tmp_path)
        rng = np.random.default_rng(0)
        pts = rng.random((10_000, 4), dtype=np.float32)
        path = writer.write(1, 0.0, pts)
        _, recovered = _read_pcd(path)
        np.testing.assert_allclose(recovered, pts, rtol=1e-6)
