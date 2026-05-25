"""
Benewake Falcon K2 LiDAR UDP Packet Parser
===========================================

実データ解析 (analyze_packet.py の出力) で確定したフォーマット:

  実測ヘッダ + 1点目 (P1):
    6a 17 01 00 90 01 29 a5 2e 9f a0 05 00 00 e0 03
    19 04 15 0e c0 6d 01 42 03 02 63 6c 01 00 00 00
    00 00 30 01 72 5a 01 2a 00 00 21 00 00 00 00 00
    27 01 00 00 00 00 [f4 d9 42 03 03 00 11 00 18] b8

  P1 1点目: x=55.80m, y=0.83m, z=0m, intensity=17

ヘッダー: 54 bytes (offset 0x00-0x35)
  0x00-0x01  bytes     マジック 6a 17
  0x0A-0x0D  uint32LE  パケットサイズ
  0x22-0x23  uint16LE  パケットインデックス (フレーム境界検出に使用)

点データ (9 bytes/point, offset 0x36 から):
  0x00-0x01  uint16LE  X 座標 [mm]   (常に正、前方距離 0-65535mm)
  0x02-0x03  int16LE   Y 座標 [mm]   (横方向、左右 ±32767mm)
  0x04-0x05  int16LE   Z 座標 [mm]   (高さ方向、±32767mm)
  0x06       uint8     Intensity     (0-255)
  0x07-0x08  bytes     metadata      (詳細不明、無視)

  フルパケット(1440B): (1440-54)/9 = 154 点
  末尾パケット(351B):  (351-54)/9  =  33 点

  X == 0 は「戻り信号なし」として NaN で出力する。
"""

import struct
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── プロトコル定数 ────────────────────────────────────────────────────────────
PACKET_MAGIC   = bytes([0x6A, 0x17])
HEADER_SIZE    = 54

# 点フォーマット: X(uint16) + Y(int16) + Z(int16) + intensity(uint8) + 2bytes metadata
POINT_STRUCT   = struct.Struct('<HhhBBB')
POINT_SIZE     = POINT_STRUCT.size   # 9

# パケットインデックスのヘッダー内オフセット (0x22-0x23)
_OFFSET_PKT_IDX = 0x22

# フレームタイムアウト [s] (パケットインデックスがリセットされない場合の保険)
FRAME_TIMEOUT_S = 0.5


@dataclass
class PointCloudFrame:
    """アセンブル済み 1 フレーム分の点群"""
    frame_id:  int
    timestamp: float                   # Unix 時刻 [s]
    points:    np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))


@dataclass
class _FrameBuffer:
    frame_id:    int
    chunks:      List[np.ndarray] = field(default_factory=list)
    last_update: float = 0.0
    last_pkt_idx: int = -1


class FalconK2Parser:
    """
    Falcon K2 UDP パケットをパースしフレームをアセンブルする。

    点フォーマット: 直交座標 XYZ (mm) 直接エンコード
      X: uint16 LE (0-65535 mm, 常に前方)
      Y: int16  LE (-32768 to +32767 mm, 横方向)
      Z: int16  LE (-32768 to +32767 mm, 高さ)

    フレーム境界:
      パケットインデックス (0x22-0x23) の減少を検出して前フレーム確定。
      検出されない場合は FRAME_TIMEOUT_S 経過で強制出力。
    """

    def __init__(self) -> None:
        self._current_buffer: Optional[_FrameBuffer] = None
        self._frame_counter:  int = 0

    # ── パブリック API ────────────────────────────────────────────────────────

    def feed(self, data: bytes) -> Optional[PointCloudFrame]:
        """
        1 つの UDP ペイロードを受け取り、フレームが確定したら
        PointCloudFrame を返す。
        """
        if len(data) < HEADER_SIZE:
            return None
        if data[:2] != PACKET_MAGIC:
            return None

        # パケットサイズ整合性チェック
        try:
            pkt_size = struct.unpack_from('<I', data, 0x0A)[0]
            pkt_idx  = struct.unpack_from('<H', data, _OFFSET_PKT_IDX)[0]
        except struct.error:
            return None
        if pkt_size != len(data):
            return None

        # 点群抽出
        payload = data[HEADER_SIZE:]
        n_points = len(payload) // POINT_SIZE
        if n_points == 0:
            return None
        points = self._parse_points(payload, n_points)

        # ── フレームバッファ管理 ──
        now = time.monotonic()
        completed: Optional[PointCloudFrame] = None

        # パケットインデックスが減少 = 新フレーム
        if (self._current_buffer is not None
                and pkt_idx < self._current_buffer.last_pkt_idx):
            completed = self._emit_frame()

        # タイムアウト確認 (パケット未着の場合の保険)
        elif (self._current_buffer is not None
                and now - self._current_buffer.last_update > FRAME_TIMEOUT_S):
            completed = self._emit_frame()

        # 現パケットをバッファに追加
        if self._current_buffer is None:
            self._current_buffer = _FrameBuffer(
                frame_id=self._frame_counter,
                last_update=now,
                last_pkt_idx=pkt_idx,
            )
        self._current_buffer.chunks.append(points)
        self._current_buffer.last_update = now
        self._current_buffer.last_pkt_idx = pkt_idx

        return completed

    def reset(self) -> None:
        """バッファをクリアして状態をリセットする"""
        self._current_buffer = None
        self._frame_counter = 0

    # ── 内部メソッド ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_points(payload: bytes, n_points: int) -> np.ndarray:
        """
        9バイト/点 (X u16, Y i16, Z i16, intensity, meta×2) を
        XYZ [m] + intensity の (N, 4) float32 配列にデコードする。
        X == 0 (戻り信号なし) の点は NaN で表現する。
        """
        buf = np.frombuffer(payload[: n_points * POINT_SIZE], dtype=np.uint8)
        buf = buf.reshape(n_points, POINT_SIZE)

        # X: uint16 LE (mm)
        x_raw = (buf[:, 0].astype(np.uint32)
                 | (buf[:, 1].astype(np.uint32) << 8))

        # Y: int16 LE (mm) - sign extend
        y_raw = (buf[:, 2].astype(np.uint32)
                 | (buf[:, 3].astype(np.uint32) << 8)).astype(np.int32)
        y_raw[y_raw >= 0x8000] -= 0x10000

        # Z: int16 LE (mm) - sign extend
        z_raw = (buf[:, 4].astype(np.uint32)
                 | (buf[:, 5].astype(np.uint32) << 8)).astype(np.int32)
        z_raw[z_raw >= 0x8000] -= 0x10000

        intens = buf[:, 6]

        # mm → m
        x = x_raw.astype(np.float32) / 1000.0
        y = y_raw.astype(np.float32) / 1000.0
        z = z_raw.astype(np.float32) / 1000.0

        # X == 0 は戻り信号なし → NaN
        invalid = x_raw == 0
        x[invalid] = np.nan
        y[invalid] = np.nan
        z[invalid] = np.nan

        out = np.empty((n_points, 4), dtype=np.float32)
        out[:, 0] = x
        out[:, 1] = y
        out[:, 2] = z
        out[:, 3] = intens.astype(np.float32)
        return out

    def _emit_frame(self) -> Optional[PointCloudFrame]:
        """現在のバッファを PointCloudFrame として返し、新バッファを開始"""
        buf = self._current_buffer
        self._current_buffer = None
        if buf is None or not buf.chunks:
            return None
        points = np.concatenate(buf.chunks, axis=0)
        if points.shape[0] == 0:
            return None
        self._frame_counter += 1
        return PointCloudFrame(
            frame_id=buf.frame_id,
            timestamp=time.time(),
            points=points,
        )
