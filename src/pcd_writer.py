"""
PCD ファイルライター

PCD v0.7 バイナリ形式でポイントクラウドを保存する。
フィールド: x, y, z, intensity (すべて float32)

ファイル名形式: frame_{frame_id:08d}_{timestamp_ms}.pcd
"""

import os
import struct
import numpy as np
from pathlib import Path


class PCDWriter:
    """
    PCD ファイルを出力ディレクトリに書き出すクラス。

    Parameters
    ----------
    output_dir : str | Path
        保存先ディレクトリ。存在しない場合は自動生成する。
    """

    # PCD v0.7 ヘッダーテンプレート
    _HEADER_TEMPLATE = (
        "# .PCD v0.7 - Point Cloud Data\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH {width}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS {points}\n"
        "DATA binary\n"
    )

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file_count = 0

    # ── パブリック API ────────────────────────────────────────────────────────

    def write(
        self,
        frame_id: int,
        timestamp: float,
        points: np.ndarray,
    ) -> str:
        """
        点群を PCD ファイルとして保存する。

        Parameters
        ----------
        frame_id  : int
        timestamp : float   Unix 時刻 [s]
        points    : ndarray shape (N, 4) float32  [x, y, z, intensity]

        Returns
        -------
        str : 保存したファイルのフルパス
        """
        if points.shape[0] == 0:
            return ""

        # float32 に正規化
        pts = points.astype(np.float32)

        timestamp_ms = int(timestamp * 1000)
        filename = f"frame_{frame_id:08d}_{timestamp_ms}.pcd"
        filepath = self.output_dir / filename

        n = pts.shape[0]
        header = self._HEADER_TEMPLATE.format(width=n, points=n)

        with open(filepath, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(pts.tobytes())   # binary (4 bytes * 4 fields * N points)

        self._file_count += 1
        return str(filepath)

    def reset(self, output_dir: str | Path) -> None:
        """新しい録画セッション開始時に出力先を更新する"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file_count = 0

    @property
    def file_count(self) -> int:
        return self._file_count
