"""
Benewake Falcon K2 LiDAR UDP Packet Parser
===========================================

診断ログから判明した実際のパケット構造:

  先頭24B実測値: 6a 17 01 00 90 01 d2 e1 6d 3a a0 05 00 00 e0 03 62 10 08 5a 2e d3 de 41

  Offset  Size  Type      Value(例)  Description
  ────────────────────────────────────────────────────────────────
  0       2     bytes     6a 17      マジック (旧実装の AB CD は誤り)
  2       2     uint16LE  00 01      プロトコルバージョン = 1
  4       2     uint16LE  90 01=400  モデル/チャンネル設定
  6       4     uint32LE  変化       サブカウンター (パケット固有)
  10      4     uint32LE  a0 05..    パケットサイズ [bytes] ← 実測確認済み
  14      2     uint16LE  e0 03=992  フレームID (同フレームは同値)
  16      8     float64LE 変化       タイムスタンプ (単位不明 → システム時刻を使用)
  24+     N×12  -         -          点群 (float32 x, y, z) × N 点
                                     N = (packet_size - 24) // 12
                                     フルパケット: (1440-24)//12 = 118 点

フレーム構成:
  - 同じ frame_id を持つパケットをひとつのフレームに集約する
  - frame_id が変わったタイミングで前フレームを確定・出力する
  - タイムアウト (FRAME_TIMEOUT_S) を超えても未確定なら強制出力
"""

import struct
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── プロトコル定数 ────────────────────────────────────────────────────────────
PACKET_MAGIC   = bytes([0x6A, 0x17])     # 実測確認済み

# ヘッダー (24 bytes)
# '<'=little-endian: 2s,H,H,I,I,H,8s → 2+2+2+4+4+2+8 = 24 bytes
HEADER_STRUCT  = struct.Struct('<2sHHIIH8s')
HEADER_SIZE    = HEADER_STRUCT.size   # 24

# 点データ: float32 x, y, z のみ (12 bytes/point)
# 強度チャンネルはペイロードに含まれないため 255 固定とする
POINT_STRUCT   = struct.Struct('<fff')
POINT_SIZE     = POINT_STRUCT.size    # 12

# フレームタイムアウト: この秒数内に同 frame_id のパケットが来なければ強制出力
FRAME_TIMEOUT_S = 0.5


@dataclass
class PointCloudFrame:
    """アセンブル済み 1 フレーム分の点群"""
    frame_id:  int
    timestamp: float                   # Unix 時刻 [s] (システム時刻)
    points:    np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    # points shape: (N, 4)  columns: x, y, z, intensity(=255)


@dataclass
class _FrameBuffer:
    frame_id:    int
    chunks:      List[np.ndarray] = field(default_factory=list)
    last_update: float = 0.0          # monotonic time


class FalconK2Parser:
    """
    Falcon K2 UDP パケットをパースしフレームをアセンブルする。

    フレーム検出方式:
      - ヘッダーの frame_id (bytes 14-15, uint16 LE) が変化したタイミングで
        前フレームを確定して返す。
      - FRAME_TIMEOUT_S を超えて更新がなければ強制出力する。
    """

    def __init__(self) -> None:
        self._buffers: Dict[int, _FrameBuffer] = {}
        self._last_frame_id: Optional[int] = None

    # ── パブリック API ────────────────────────────────────────────────────────

    def feed(self, data: bytes) -> Optional[PointCloudFrame]:
        """
        1 つの UDP ペイロードを受け取り、フレームが確定したら PointCloudFrame を返す。
        まだ確定していなければ None を返す。
        """
        if len(data) < HEADER_SIZE:
            return None

        # ── ヘッダーパース ────────────────────────────────────────────────────
        try:
            (magic, version, model_cfg,
             sub_counter, pkt_size,
             frame_id, ts_bytes) = HEADER_STRUCT.unpack_from(data, 0)
        except struct.error:
            return None

        if magic != PACKET_MAGIC:
            return None

        # ── 点群抽出 ──────────────────────────────────────────────────────────
        payload = data[HEADER_SIZE:]
        n_points = len(payload) // POINT_SIZE   # 余りバイトは無視
        if n_points > 0:
            pts = self._parse_points(payload, n_points)
        else:
            pts = np.empty((0, 4), dtype=np.float32)

        # ── フレームバッファ管理 ──────────────────────────────────────────────
        now = time.monotonic()

        # 前フレームを確定すべきか判定
        completed_frame: Optional[PointCloudFrame] = None

        if self._last_frame_id is not None and frame_id != self._last_frame_id:
            # frame_id が変わった → 前フレームを確定
            completed_frame = self._assemble_and_clear(self._last_frame_id)

        # 現パケットをバッファに追加
        if frame_id not in self._buffers:
            self._buffers[frame_id] = _FrameBuffer(frame_id=frame_id, last_update=now)
        buf = self._buffers[frame_id]
        if pts.shape[0] > 0:
            buf.chunks.append(pts)
        buf.last_update = now

        self._last_frame_id = frame_id

        # タイムアウトした古いバッファを強制出力
        timed_out = self._check_timeouts(now, current_frame_id=frame_id)
        if timed_out is not None and completed_frame is None:
            completed_frame = timed_out

        return completed_frame

    def reset(self) -> None:
        """受信バッファをクリア"""
        self._buffers.clear()
        self._last_frame_id = None

    # ── 内部メソッド ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_points(payload: bytes, n_points: int) -> np.ndarray:
        """ペイロードから (N,4) float32 配列を返す (intensity=255 固定)"""
        arr = np.empty((n_points, 4), dtype=np.float32)
        for i in range(n_points):
            try:
                x, y, z = POINT_STRUCT.unpack_from(payload, i * POINT_SIZE)
            except struct.error:
                arr = arr[:i]
                break
            arr[i, 0] = x
            arr[i, 1] = y
            arr[i, 2] = z
            arr[i, 3] = 255.0   # intensity 未定義 → 255
        return arr

    def _assemble_and_clear(self, frame_id: int) -> Optional[PointCloudFrame]:
        """指定フレームのバッファを結合して PointCloudFrame を返し、バッファを削除"""
        buf = self._buffers.pop(frame_id, None)
        if buf is None:
            return None
        points = np.concatenate(buf.chunks, axis=0) if buf.chunks \
                 else np.empty((0, 4), dtype=np.float32)
        if points.shape[0] == 0:
            return None
        return PointCloudFrame(
            frame_id=frame_id,
            timestamp=time.time(),
            points=points,
        )

    def _check_timeouts(
        self, now: float, current_frame_id: int
    ) -> Optional[PointCloudFrame]:
        """タイムアウトした (current_frame_id 以外の) バッファを強制出力"""
        for fid in list(self._buffers.keys()):
            if fid == current_frame_id:
                continue
            buf = self._buffers[fid]
            if now - buf.last_update > FRAME_TIMEOUT_S:
                logger.debug("フレーム %d タイムアウト (%.3fs)", fid, now - buf.last_update)
                return self._assemble_and_clear(fid)
        return None
