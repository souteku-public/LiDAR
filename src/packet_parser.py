"""
Seyond/Innovusion Falcon K2 LiDAR UDP パケットパーサー
======================================================

パケット構造: InnoDataPacket (54バイトヘッダー + InnoBlock1 配列)

ヘッダー (54 bytes on-wire):
  bytes  0-1:  マジック 0x6A 0x17
  bytes  8-11: パケットサイズ (uint32 LE)
  bytes 26-33: frame_idx (uint64 LE)  ← フレーム境界検出に使用
  bytes 38-41: uint32 LE = type (bits 0-7) + item_number (bits 8-31)
  bytes 42-43: item_size (uint16 LE) = 33 (InnoBlock1) または 49 (InnoBlock2)
  bytes 48-49: flags (uint16 LE)

InnoBlock1 (33 bytes = 17バイトヘッダー + 4 × 4バイト InnoChannelPoint):
  bytes  0-1:  h_angle  (int16 LE) [InnoAngleUnit]
  bytes  2-3:  v_angle  (int16 LE) [InnoAngleUnit]
  bytes  4-5:  ts_10us  (uint16 LE)
  bytes  6-7:  scan_idx (uint16 LE)
  bytes  8-16: ビットフィールド (72ビット LE パック):
      bits  0- 8:  scan_id  (9ビット)
      bits  9-17:  h_diff1  (9ビット符号付き)
      bits 18-27:  h_diff2  (10ビット符号付き)
      bits 28-38:  h_diff3  (11ビット符号付き)
      bits 39-46:  v_diff1  (8ビット符号付き)
      bits 47-55:  v_diff2  (9ビット符号付き)
      bits 56-64:  v_diff3  (9ビット符号付き)
      bits 65-71:  in_roi / facet / reserved
  bytes 17-20: CH0 InnoChannelPoint (uint32 LE):
      bits  0-16: radius (17ビット) [InnoDistanceUnit]
      bits 17-24: refl   (8ビット)
  bytes 21-24: CH1 InnoChannelPoint
  bytes 25-28: CH2 InnoChannelPoint
  bytes 29-32: CH3 InnoChannelPoint

チャンネル角度計算 (kInnoFaconVAngleDiffBase = 196):
  CH0: h = h_angle,           v = v_angle
  CHi: h = h_angle + h_diff_i, v = v_angle + v_diff_i + 196 × i

XYZ 変換 (Seyond 座標系: X=上, Y=右, Z=前):
  r      = radius × (1/200)    [m]   (kInnoDistanceUnitPerMeter200 = 200)
  1 unit = π/32768 rad
  t      = r × cos(v_rad)
  x      = r × sin(v_rad)            (上方向)
  y      = t × sin(h_rad)            (右方向)
  z      = t × cos(h_rad)            (前方向)

参照: Seyond InnoLiDAR SDK inno_lidar_packet.h / inno_lidar_packet_utils.cpp
"""

import math
import struct
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── プロトコル定数 ─────────────────────────────────────────────────────────────
PACKET_MAGIC              = bytes([0x6A, 0x17])
HEADER_SIZE               = 54                    # on-wire ヘッダーサイズ [bytes]

# InnoBlock1 レイアウト
BLOCK1_SIZE               = 33                    # InnoBlock1 サイズ [bytes]
BLOCK_HDR_SIZE            = 17                    # ブロックヘッダーサイズ [bytes]
CHANNEL_PT_SIZE           = 4                     # InnoChannelPoint サイズ [bytes]
CHANNELS_PER_BLOCK        = 4                     # チャンネル数
BLOCKS_PER_FULL_PKT       = 42                    # フルパケット (1440B) のブロック数

# 変換定数
METERS_PER_UNIT           = 1.0 / 200.0           # 1 InnoDistanceUnit = 5mm = 0.005m
RAD_PER_UNIT              = math.pi / 32768.0     # 1 InnoAngleUnit [rad]
V_ANGLE_BASE_PER_CH       = 196                   # kInnoFaconVAngleDiffBase

# ヘッダーフィールドオフセット
OFFSET_PKT_SIZE           = 8                     # パケットサイズ (uint32 LE)
OFFSET_FRAME_IDX          = 26                    # frame_idx     (uint64 LE)
OFFSET_TYPE_NUM           = 38                    # type + item_number (uint32 LE)
OFFSET_ITEM_SIZE          = 42                    # item_size     (uint16 LE)
OFFSET_FLAGS              = 48                    # flags         (uint16 LE)

# パケットタイプ
INNO_ITEM_TYPE_SPHERE_POINTCLOUD = 1

FRAME_TIMEOUT_S           = 0.5                   # フレームタイムアウト [s]


# ── ユーティリティ ────────────────────────────────────────────────────────────

def _sign_extend(value: int, bits: int) -> int:
    """n ビット符号なし整数を Python 符号付き整数に変換する。"""
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


# ── データクラス ──────────────────────────────────────────────────────────────

@dataclass
class PointCloudFrame:
    """アセンブル済み 1 フレーム分の点群"""
    frame_id:  int
    timestamp: float                    # Unix 時刻 [s]
    points:    np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )
    # points 列: [x, y, z, refl]
    #   x = 上方向 [m]
    #   y = 右方向 [m]
    #   z = 前方向 [m]
    #   refl = 反射強度 (0-255)


@dataclass
class _FrameBuffer:
    frame_id:       int
    chunks:         List[np.ndarray] = field(default_factory=list)
    last_update:    float = 0.0
    last_frame_idx: int   = -1


# ── ブロック解析 ──────────────────────────────────────────────────────────────

def _parse_blocks(payload: bytes, n_blocks: int) -> np.ndarray:
    """
    payload から n_blocks 個の InnoBlock1 (各 33 bytes) を一括解析し、
    有効点の (N, 4) float32 配列 [x, y, z, refl] を返す。
    有効点がない場合は shape=(0,4) の空配列を返す。
    """
    total = n_blocks * BLOCK1_SIZE
    if len(payload) < total:
        return np.empty((0, 4), dtype=np.float32)

    buf = np.frombuffer(payload[:total], dtype=np.uint8).reshape(n_blocks, BLOCK1_SIZE)

    # ── ブロックヘッダー: h_angle, v_angle (signed int16 LE) ──
    h_ang = buf[:, 0].astype(np.int32) | (buf[:, 1].astype(np.int32) << 8)
    h_ang = np.where(h_ang >= 0x8000, h_ang - 0x10000, h_ang)

    v_ang = buf[:, 2].astype(np.int32) | (buf[:, 3].astype(np.int32) << 8)
    v_ang = np.where(v_ang >= 0x8000, v_ang - 0x10000, v_ang)

    # ── ビットフィールド (bytes 8-16, 72ビット LE) ──
    # numpy uint64 は 64 ビットなので、残り 8 ビット (byte 16) を別途処理する
    h_diff = np.zeros((n_blocks, 3), dtype=np.int32)
    v_diff = np.zeros((n_blocks, 3), dtype=np.int32)

    # bytes 8-15 を uint64 LE として読み込む (bits 0-63)
    bf64 = np.zeros(n_blocks, dtype=np.uint64)
    for b in range(8):
        bf64 |= buf[:, 8 + b].astype(np.uint64) << np.uint64(b * 8)

    # byte 16 (bits 64-71)
    bf_hi = buf[:, 16].astype(np.int32)

    # h_diff1 (bits 9-17, 9ビット)
    h_diff[:, 0] = _sign_extend_np((bf64 >> np.uint64(9)) & np.uint64(0x1FF), 9)
    # h_diff2 (bits 18-27, 10ビット)
    h_diff[:, 1] = _sign_extend_np((bf64 >> np.uint64(18)) & np.uint64(0x3FF), 10)
    # h_diff3 (bits 28-38, 11ビット)
    h_diff[:, 2] = _sign_extend_np((bf64 >> np.uint64(28)) & np.uint64(0x7FF), 11)
    # v_diff1 (bits 39-46, 8ビット)
    v_diff[:, 0] = _sign_extend_np((bf64 >> np.uint64(39)) & np.uint64(0xFF),  8)
    # v_diff2 (bits 47-55, 9ビット)
    v_diff[:, 1] = _sign_extend_np((bf64 >> np.uint64(47)) & np.uint64(0x1FF), 9)
    # v_diff3 (bits 56-64, 9ビット): bits 56-63 in bf64, bit 64 in bf_hi
    v_diff3_lo = ((bf64 >> np.uint64(56)) & np.uint64(0xFF)).astype(np.int32)
    v_diff3_hi = (bf_hi & 0x01) << 8
    v_diff[:, 2] = _sign_extend_np_int32(v_diff3_lo | v_diff3_hi, 9)

    # ── チャンネルごとの角度 (shape: n_blocks × 4) ──
    h_angles = np.column_stack([
        h_ang,
        h_ang + h_diff[:, 0],
        h_ang + h_diff[:, 1],
        h_ang + h_diff[:, 2],
    ]).astype(np.float64)   # (n_blocks, 4)

    v_angles = np.column_stack([
        v_ang,
        v_ang + v_diff[:, 0] + V_ANGLE_BASE_PER_CH,
        v_ang + v_diff[:, 1] + V_ANGLE_BASE_PER_CH * 2,
        v_ang + v_diff[:, 2] + V_ANGLE_BASE_PER_CH * 3,
    ]).astype(np.float64)   # (n_blocks, 4)

    # ── InnoChannelPoint: radius (bits 0-16) + refl (bits 17-24) ──
    channel_pts = np.zeros((n_blocks, 4), dtype=np.uint32)
    for ch in range(CHANNELS_PER_BLOCK):
        off = BLOCK_HDR_SIZE + ch * CHANNEL_PT_SIZE
        channel_pts[:, ch] = (
            buf[:, off    ].astype(np.uint32)
            | (buf[:, off+1].astype(np.uint32) << 8)
            | (buf[:, off+2].astype(np.uint32) << 16)
            | (buf[:, off+3].astype(np.uint32) << 24)
        )

    radius = (channel_pts & 0x1FFFF).astype(np.float64)    # (n_blocks, 4)
    refl   = ((channel_pts >> 17) & 0xFF).astype(np.float32)  # (n_blocks, 4)

    # ── 有効点マスク (radius > 0) ──
    valid = radius > 0.0    # (n_blocks, 4) bool

    if not valid.any():
        return np.empty((0, 4), dtype=np.float32)

    # ── フラット化して有効点のみ XYZ 計算 ──
    r_v    = (radius[valid] * METERS_PER_UNIT)
    h_v    = h_angles[valid] * RAD_PER_UNIT
    v_v    = v_angles[valid] * RAD_PER_UNIT
    refl_v = refl[valid]

    cos_v = np.cos(v_v)
    sin_v = np.sin(v_v)
    sin_h = np.sin(h_v)
    cos_h = np.cos(h_v)

    t = r_v * cos_v
    x = (r_v * sin_v).astype(np.float32)   # 上方向
    y = (t   * sin_h).astype(np.float32)   # 右方向
    z = (t   * cos_h).astype(np.float32)   # 前方向

    n_pts = int(valid.sum())
    out = np.empty((n_pts, 4), dtype=np.float32)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = z
    out[:, 3] = refl_v
    return out


def _sign_extend_np(arr: np.ndarray, bits: int) -> np.ndarray:
    """uint64 配列の下位 bits ビットを int32 符号付きに変換する。"""
    sign_bit = np.uint64(1 << (bits - 1))
    mask     = np.uint64((1 << bits) - 1)
    val      = (arr & mask).astype(np.int64)
    sign_bit_i = int(sign_bit)
    return np.where(val >= sign_bit_i, val - (sign_bit_i << 1), val).astype(np.int32)


def _sign_extend_np_int32(arr: np.ndarray, bits: int) -> np.ndarray:
    """int32 配列の下位 bits ビットを符号付きに変換する。"""
    sign_bit = 1 << (bits - 1)
    mask     = (1 << bits) - 1
    val      = arr & mask
    return np.where(val >= sign_bit, val - (sign_bit << 1), val).astype(np.int32)


# ── メインパーサークラス ──────────────────────────────────────────────────────

class FalconK2Parser:
    """
    Seyond Falcon K2 UDP パケット (InnoDataPacket) をパースし、
    フレーム単位で点群をアセンブルする。

    パケット境界:
      frame_idx (ヘッダー bytes 26-33, uint64) の変化でフレームを確定する。
      FRAME_TIMEOUT_S 経過で強制出力。

    対応パケットタイプ:
      INNO_ITEM_TYPE_SPHERE_POINTCLOUD (type=1), item_size=33 (InnoBlock1) のみ。
    """

    def __init__(self) -> None:
        self._current_buffer: Optional[_FrameBuffer] = None
        self._frame_counter:  int = 0

    # ── パブリック API ────────────────────────────────────────────────────────

    def feed(self, data: bytes) -> Optional[PointCloudFrame]:
        """
        1 つの UDP ペイロードを受け取り、フレームが確定したら
        PointCloudFrame を返す。フレーム未確定の場合は None を返す。
        """
        # ── ヘッダー検証 ──
        if len(data) < HEADER_SIZE:
            return None
        if data[:2] != PACKET_MAGIC:
            return None

        try:
            pkt_size  = struct.unpack_from('<I', data, OFFSET_PKT_SIZE)[0]
            frame_idx = struct.unpack_from('<Q', data, OFFSET_FRAME_IDX)[0]
            type_num  = struct.unpack_from('<I', data, OFFSET_TYPE_NUM)[0]
            item_size = struct.unpack_from('<H', data, OFFSET_ITEM_SIZE)[0]
        except struct.error:
            return None

        if pkt_size != len(data):
            return None

        pkt_type    = type_num & 0xFF
        item_number = (type_num >> 8) & 0xFFFFFF

        # InnoBlock1 (type=1, item_size=33) のみ対応
        if pkt_type != INNO_ITEM_TYPE_SPHERE_POINTCLOUD or item_size != BLOCK1_SIZE:
            return None

        if item_number == 0:
            return None

        payload = data[HEADER_SIZE:]
        if len(payload) < item_number * BLOCK1_SIZE:
            return None

        # ── ブロック解析 ──
        points = _parse_blocks(payload, item_number)

        # ── フレームバッファ管理 ──
        now = time.monotonic()
        completed: Optional[PointCloudFrame] = None

        if self._current_buffer is not None:
            if frame_idx != self._current_buffer.last_frame_idx:
                # frame_idx 変化 → 前フレームを確定
                completed = self._emit_frame()
            elif now - self._current_buffer.last_update > FRAME_TIMEOUT_S:
                # タイムアウト → 強制確定
                completed = self._emit_frame()

        if self._current_buffer is None:
            self._current_buffer = _FrameBuffer(
                frame_id=self._frame_counter,
                last_update=now,
                last_frame_idx=frame_idx,
            )

        if points.shape[0] > 0:
            self._current_buffer.chunks.append(points)
        self._current_buffer.last_update    = now
        self._current_buffer.last_frame_idx = frame_idx

        return completed

    def reset(self) -> None:
        """バッファをクリアして状態をリセットする。"""
        self._current_buffer = None
        self._frame_counter  = 0

    # ── 内部メソッド ─────────────────────────────────────────────────────────

    def _emit_frame(self) -> Optional[PointCloudFrame]:
        """現在のバッファを PointCloudFrame として返し、バッファをリセットする。"""
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
