"""
Benewake Falcon K2 LiDAR UDP Packet Parser
===========================================

実測パケット (hex dump) から確定したフォーマット:

  パケット先頭64Bの例 (P1):
    0000: 6a 17 01 00 90 01 29 a5 2e 9f a0 05 00 00 e0 03
    0010: 19 04 15 0e c0 6d 01 42 03 02 63 6c 01 00 00 00
    0020: 00 00 30 01 72 5a 01 2a 00 00 21 00 00 00 00 00
    0030: 27 01 00 00 00 00 f4 d9 42 03 03 00 11 00 18 b8

ヘッダー (54 bytes, offset 0x00-0x35):
  0x00-0x01  bytes     6a 17       マジック
  0x02-0x03  uint16LE  01 00       バージョン (1)
  0x04-0x05  uint16LE  90 01       定数 (0x0190 = 400, モデル設定?)
  0x06-0x09  uint32LE  -           sub_counter (毎パケット変化)
  0x0A-0x0D  uint32LE  a0 05 00 00 パケットサイズ (1440 or 351)
  0x0E-0x0F  uint16LE  e0 03       定数 (0x03E0 = 992)
  0x10-0x13  uint32LE  -           タイムスタンプ系
  0x14-0x21  18 bytes  -           デバイス共通情報 (全パケット同値)
  0x22-0x23  uint16LE  -           ★パケットインデックス★ (連番)
  0x24-0x25  uint16LE  -           副カウンター (+1ずつ)
  0x26-0x35  16 bytes  -           その他メタデータ

点データ (offset 0x36 から、9 bytes/point):
  0x00-0x01  uint16LE  range       距離 [mm]
  0x02-0x03  uint16LE  azimuth     方位角 [0.01度]
  0x04-0x05  uint16LE  elevation   仰角 [0.01度] (またはチャネルID)
  0x06       uint8     intensity   強度 (0-255)
  0x07       uint8     channel     チャネル/フラグ
  0x08       uint8     flag        フラグ

  フルパケット(1440B): (1440-54)/9 = 154 点
  末尾パケット(351B):  (351-54)/9  =  33 点

XYZ への変換:
  r  = range_mm / 1000   [m]
  az = azimuth × π/18000 [rad]
  el = elevation × π/18000 [rad]
  x  = r × cos(el) × cos(az)
  y  = r × cos(el) × sin(az)
  z  = r × sin(el)

range = 0 の点は「戻り信号なし」として NaN で出力する。
"""

import math
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

# 点フォーマット: range(uint16) + azimuth(uint16) + elevation(uint16) +
#                 intensity(uint8) + channel(uint8) + flag(uint8) = 9 bytes
POINT_STRUCT   = struct.Struct('<HHHBBB')
POINT_SIZE     = POINT_STRUCT.size   # 9

# 角度単位変換: 0.01度 → rad
_ANGLE_TO_RAD  = math.pi / 18000.0   # = (π/180) / 100

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

    フレーム境界検出:
      パケットインデックス (0x22-0x23) が小さくなった (= リセットされた)
      タイミングを新フレームの開始とみなす。例: 469 → 0 や、
      308 → 1 のような減少を検出。
      検出できない場合は FRAME_TIMEOUT_S 経過で強制出力。
    """

    def __init__(self) -> None:
        self._current_buffer: Optional[_FrameBuffer] = None
        self._frame_counter:  int = 0    # 出力フレームID (自前で発行)

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

        # パケットサイズと整合性チェック
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

        # パケットインデックスが減少した = 新フレーム
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
        9バイト/点 (range, azimuth, elevation, intensity, ch, flag) を
        XYZ + intensity の (N, 4) float32 配列にデコードする。
        range == 0 の点は NaN で表現する。
        """
        # 一括 numpy で読む方が速いが、まず正確性重視で struct ループ
        ranges  = np.empty(n_points, dtype=np.float32)
        azis    = np.empty(n_points, dtype=np.float32)
        elvs    = np.empty(n_points, dtype=np.float32)
        intens  = np.empty(n_points, dtype=np.float32)

        # numpy で一括デコード (高速)
        # offsets:  range:0-1, az:2-3, el:4-5, intensity:6, ch:7, flag:8
        buf = np.frombuffer(payload[: n_points * POINT_SIZE], dtype=np.uint8)
        buf = buf.reshape(n_points, POINT_SIZE)

        ranges[:] = (buf[:, 0].astype(np.uint32) | (buf[:, 1].astype(np.uint32) << 8))
        azis[:]   = (buf[:, 2].astype(np.uint32) | (buf[:, 3].astype(np.uint32) << 8))
        elvs[:]   = (buf[:, 4].astype(np.uint32) | (buf[:, 5].astype(np.uint32) << 8))
        intens[:] = buf[:, 6]

        # メートル / ラジアンに変換
        r  = ranges / 1000.0
        az = azis * _ANGLE_TO_RAD
        el = elvs * _ANGLE_TO_RAD

        cos_el = np.cos(el)
        x = r * cos_el * np.cos(az)
        y = r * cos_el * np.sin(az)
        z = r * np.sin(el)

        # range==0 は「戻り信号なし」 → NaN
        invalid = ranges == 0
        x[invalid] = np.nan
        y[invalid] = np.nan
        z[invalid] = np.nan

        out = np.empty((n_points, 4), dtype=np.float32)
        out[:, 0] = x
        out[:, 1] = y
        out[:, 2] = z
        out[:, 3] = intens
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
