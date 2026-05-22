"""
Benewake Falcon K2 LiDAR UDP Packet Parser

Falcon K2 UDP パケット構造 (Benewake 標準バイナリプロトコル):
  Offset  Size  Type    Description
  0       2     uint8   Magic header: 0xAB, 0xCD
  2       2     uint16  Packet type (0x0101 = point cloud)
  4       4     uint32  Frame ID
  8       2     uint16  Packet index (within frame, 0-based)
  10      2     uint16  Total packets per frame
  12      8     uint64  Timestamp [μs since epoch]
  20      2     uint16  Number of points in this packet
  22      2     uint16  Reserved
  24+     N*14  -       Point records

Point record (14 bytes each):
  0       4     float32  X [m]
  4       4     float32  Y [m]
  8       4     float32  Z [m]
  12      1     uint8    Intensity (0-255)
  13      1     uint8    Tag / flags

実機のファームウェアバージョンによっては構造が異なる場合があります。
その場合は PACKET_MAGIC / POINT_STRUCT / HEADER_STRUCT を調整してください。
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── プロトコル定数 ────────────────────────────────────────────────────────────
PACKET_MAGIC       = bytes([0xAB, 0xCD])
PACKET_TYPE_PCLOUD = 0x0101          # ポイントクラウドパケット

HEADER_STRUCT = struct.Struct('<2sHIHHQHH')   # 24 bytes
HEADER_SIZE   = HEADER_STRUCT.size            # 24

POINT_STRUCT  = struct.Struct('<fffBB')        # 14 bytes
POINT_SIZE    = POINT_STRUCT.size             # 14

MAX_POINTS_PER_PACKET = 200                   # 安全上限


@dataclass
class PointCloudFrame:
    """アセンブル済み 1 フレーム分の点群"""
    frame_id:   int
    timestamp:  float                          # Unix 時刻 [s]
    points:     np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    # points shape: (N, 4)  columns: x, y, z, intensity


@dataclass
class _FrameBuffer:
    frame_id:      int
    total_packets: int
    received:      Dict[int, np.ndarray] = field(default_factory=dict)
    timestamp:     float = 0.0


class FalconK2Parser:
    """
    Falcon K2 UDP パケットをパースし、完全なフレームが揃ったら
    PointCloudFrame を返すアセンブラ。
    """

    def __init__(self) -> None:
        self._buffers: Dict[int, _FrameBuffer] = {}

    # ── パブリック API ────────────────────────────────────────────────────────

    def feed(self, data: bytes) -> Optional[PointCloudFrame]:
        """
        1 つの UDP ペイロードを受け取り、フレームが完成したら
        PointCloudFrame を返す。未完成なら None を返す。
        """
        if len(data) < HEADER_SIZE:
            return None

        try:
            (magic, pkt_type, frame_id,
             pkt_idx, total_pkts,
             timestamp_us,
             n_points, _reserved) = HEADER_STRUCT.unpack_from(data, 0)
        except struct.error:
            return None

        if magic != PACKET_MAGIC:
            return None
        if pkt_type != PACKET_TYPE_PCLOUD:
            return None
        if n_points > MAX_POINTS_PER_PACKET:
            return None

        # 点群データを読み取る
        points = self._parse_points(data, HEADER_SIZE, n_points)

        # フレームバッファを管理
        if frame_id not in self._buffers:
            self._buffers[frame_id] = _FrameBuffer(
                frame_id=frame_id,
                total_packets=total_pkts,
                timestamp=timestamp_us / 1e6,
            )

        buf = self._buffers[frame_id]
        buf.received[pkt_idx] = points

        # フレームが揃ったか確認
        if len(buf.received) >= buf.total_packets:
            frame = self._assemble(buf)
            del self._buffers[frame_id]
            # 古いバッファを破棄 (メモリリーク防止)
            self._gc(frame_id)
            return frame

        return None

    def reset(self) -> None:
        """受信バッファをクリアする"""
        self._buffers.clear()

    # ── 内部メソッド ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_points(data: bytes, offset: int, n_points: int) -> np.ndarray:
        """パケット本体から点群を (N,4) float32 配列として返す"""
        arr = np.empty((n_points, 4), dtype=np.float32)
        for i in range(n_points):
            try:
                x, y, z, intensity, _tag = POINT_STRUCT.unpack_from(data, offset + i * POINT_SIZE)
            except struct.error:
                arr = arr[:i]
                break
            arr[i, 0] = x
            arr[i, 1] = y
            arr[i, 2] = z
            arr[i, 3] = float(intensity)
        return arr

    @staticmethod
    def _assemble(buf: _FrameBuffer) -> PointCloudFrame:
        """バッファ内の全パケットを結合して PointCloudFrame を生成する"""
        chunks = [buf.received[k] for k in sorted(buf.received.keys())]
        points = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float32)
        return PointCloudFrame(
            frame_id=buf.frame_id,
            timestamp=buf.timestamp,
            points=points,
        )

    def _gc(self, current_frame_id: int, keep_last: int = 10) -> None:
        """古いフレームバッファを破棄する"""
        if len(self._buffers) <= keep_last:
            return
        old_ids = sorted(self._buffers.keys())
        for fid in old_ids[:-keep_last]:
            del self._buffers[fid]
