"""
Benewake Falcon K2 LiDAR UDP Packet Parser
===========================================

キャリブレーション済み XYZ 変換 (スキャンパターン CSV 使用)

パケット 1レコードあたり 9 バイト = 4チャンネル距離データ
  Bytes 0-1: CH0 距離 uint16 LE [mm]
  Bytes 2-3: CH1 距離 uint16 LE [mm]
  Bytes 4-5: CH2 距離 uint16 LE [mm]
  Bytes 6-7: CH3 距離 uint16 LE [mm]
  Byte  8:   フラグ (intensity 相当, 0-255)

スキャンパターン CSV:
  calibration/falcon_k_default_scan_pattern.csv
  25204 行 × 8 列 (CH0_H, CH0_V, CH1_H, CH1_V, CH2_H, CH2_V, CH3_H, CH3_V)
  角度はすべて度数法 [degrees]
  NaN 行はスキャン折り返し部分 (無効点として扱う)

角度マッピング (0ベース pkt_idx):
  csv_row = (pkt_idx * RECORDS_PER_PACKET + record_idx) % CSV_ROWS

XYZ 変換 (球面座標系):
  az  = H_angle [rad]   (水平角, 前方=0, 右が正)
  el  = V_angle [rad]   (垂直角, 上が正)
  r   = distance [m]
  x   = r * cos(el) * cos(az)   (前方深度)
  y   = r * cos(el) * sin(az)   (横方向, 右が正)
  z   = r * sin(el)              (高さ)

ヘッダー: 54 bytes (offset 0x00-0x35)
  0x00-0x01  bytes     マジック 6a 17
  0x0A-0x0D  uint32LE  パケットサイズ
  0x22-0x23  uint16LE  パケットインデックス (フレーム境界検出に使用)

フルパケット (1440B): (1440-54)/9 = 154 レコード × 最大 4ch = 最大 616 点
末尾パケット  (351B): ( 351-54)/9 =  33 レコード × 最大 4ch = 最大 132 点
"""

import os
import struct
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── プロトコル定数 ────────────────────────────────────────────────────────────
PACKET_MAGIC          = bytes([0x6A, 0x17])
HEADER_SIZE           = 54

# 点フォーマット: CH0-CH3 uint16 LE + flag uint8 = 9 bytes
POINT_STRUCT          = struct.Struct('<HHHHB')
POINT_SIZE            = POINT_STRUCT.size   # 9

RECORDS_PER_PACKET    = 154          # フルパケット
_OFFSET_PKT_IDX       = 0x22         # パケットインデックスオフセット
FRAME_TIMEOUT_S       = 0.5          # フレームタイムアウト [s]

# ── キャリブレーションデータ ──────────────────────────────────────────────────
_CSV_ROWS = 25204   # スキャンパターン CSV 総行数

# モジュール起動時に CSV を一度だけ読み込む
# shape: (CSV_ROWS, 8)  列順: CH0_H, CH0_V, CH1_H, CH1_V, CH2_H, CH2_V, CH3_H, CH3_V [rad]
_SCAN_ANGLES: Optional[np.ndarray] = None   # ラジアン, float32
_SCAN_VALID:  Optional[np.ndarray] = None   # bool マスク (NaN 行は False)


def _load_calibration() -> None:
    """CSV を読み込み、モジュールレベルの _SCAN_ANGLES / _SCAN_VALID を初期化する。"""
    global _SCAN_ANGLES, _SCAN_VALID

    # src/ の一つ上のディレクトリを基点にする
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "calibration", "falcon_k_default_scan_pattern.csv")

    if not os.path.isfile(csv_path):
        logger.warning("スキャンパターン CSV が見つかりません: %s", csv_path)
        logger.warning("XYZ 変換には CSV が必要です。直交座標フォールバックを使用します。")
        _SCAN_ANGLES = None
        _SCAN_VALID  = None
        return

    try:
        # ヘッダー行をスキップして 9 列 (time + 8 角度) を読み込む
        raw = np.genfromtxt(
            csv_path,
            delimiter=",",
            skip_header=1,
            usecols=range(1, 9),   # time 列を除く 8 列
            dtype=np.float32,
        )
        # 行数不足チェック
        if raw.shape[0] < _CSV_ROWS:
            logger.warning("CSV 行数が想定より少ない: %d < %d", raw.shape[0], _CSV_ROWS)

        # 度 → ラジアン変換
        angles_rad = np.deg2rad(raw)          # shape (N, 8)

        # NaN 行マスク
        valid = np.isfinite(angles_rad[:, 0])  # CH0_H が有限かどうかで判定

        _SCAN_ANGLES = angles_rad
        _SCAN_VALID  = valid
        logger.info(
            "スキャンパターン CSV を読み込みました: %d 行 (有効 %d 行)",
            raw.shape[0], int(valid.sum()),
        )

    except Exception as exc:
        logger.error("CSV 読み込みエラー: %s", exc, exc_info=True)
        _SCAN_ANGLES = None
        _SCAN_VALID  = None


# モジュールロード時に実行
_load_calibration()


# ── データクラス ──────────────────────────────────────────────────────────────

@dataclass
class PointCloudFrame:
    """アセンブル済み 1 フレーム分の点群"""
    frame_id:  int
    timestamp: float                   # Unix 時刻 [s]
    points:    np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )


@dataclass
class _FrameBuffer:
    frame_id:     int
    chunks:       List[np.ndarray] = field(default_factory=list)
    last_update:  float = 0.0
    last_pkt_idx: int = -1


# ── メインパーサークラス ──────────────────────────────────────────────────────

class FalconK2Parser:
    """
    Falcon K2 UDP パケットをパースしフレームをアセンブルする。

    スキャンパターン CSV が利用可能な場合:
      各レコード (9B) を 4 チャンネルの距離 [mm] + フラグとして解釈し、
      CSV から水平・垂直角を取得して XYZ を計算する。
      1 レコードから最大 4 点 (各チャンネル) を出力する。

    CSV が利用できない場合:
      フォールバックとして旧フォーマット (X uint16 mm, Y int16 mm,
      Z int16 mm) の直接変換を使用する。

    フレーム境界:
      パケットインデックス (0x22-0x23) の減少を検出して前フレーム確定。
      FRAME_TIMEOUT_S 経過で強制出力。
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

        # レコード抽出
        payload   = data[HEADER_SIZE:]
        n_records = len(payload) // POINT_SIZE
        if n_records == 0:
            return None

        points = self._parse_points(payload, n_records, pkt_idx)

        # ── フレームバッファ管理 ──
        now = time.monotonic()
        completed: Optional[PointCloudFrame] = None

        if (self._current_buffer is not None
                and pkt_idx < self._current_buffer.last_pkt_idx):
            completed = self._emit_frame()

        elif (self._current_buffer is not None
              and now - self._current_buffer.last_update > FRAME_TIMEOUT_S):
            completed = self._emit_frame()

        if self._current_buffer is None:
            self._current_buffer = _FrameBuffer(
                frame_id=self._frame_counter,
                last_update=now,
                last_pkt_idx=pkt_idx,
            )

        if points is not None and points.shape[0] > 0:
            self._current_buffer.chunks.append(points)
        self._current_buffer.last_update = now
        self._current_buffer.last_pkt_idx = pkt_idx

        return completed

    def reset(self) -> None:
        """バッファをクリアして状態をリセットする"""
        self._current_buffer = None
        self._frame_counter  = 0

    # ── 内部メソッド ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_points(
        payload: bytes,
        n_records: int,
        pkt_idx: int,
    ) -> Optional[np.ndarray]:
        """
        9 バイト/レコード (CH0-CH3 uint16 LE, flag uint8) をデコードし、
        キャリブレーション CSV の角度を使って XYZ [m] + intensity の
        (M, 4) float32 配列を返す。

        CSV が利用不可の場合は旧フォーマットのフォールバックを使用。
        """
        if _SCAN_ANGLES is None:
            return FalconK2Parser._parse_points_fallback(payload, n_records)

        return FalconK2Parser._parse_points_calibrated(
            payload, n_records, pkt_idx
        )

    @staticmethod
    def _parse_points_calibrated(
        payload: bytes,
        n_records: int,
        pkt_idx: int,
    ) -> np.ndarray:
        """
        スキャンパターン CSV を使ったキャリブレーション済み変換。

        各 9B レコードから最大 4 点を出力する。
        距離 == 0 または角度が NaN の点は除外する。
        """
        n_csv_rows = _SCAN_ANGLES.shape[0]

        # レコードをバッファとして読み込む
        buf = np.frombuffer(
            payload[: n_records * POINT_SIZE], dtype=np.uint8
        ).reshape(n_records, POINT_SIZE)

        # 4 チャンネルの距離を一括抽出 (uint16 LE)
        # shape: (n_records, 4)
        ranges_mm = np.empty((n_records, 4), dtype=np.uint32)
        for ch in range(4):
            lo = ch * 2
            hi = ch * 2 + 1
            ranges_mm[:, ch] = (
                buf[:, lo].astype(np.uint32)
                | (buf[:, hi].astype(np.uint32) << 8)
            )

        flag = buf[:, 8]   # shape: (n_records,)

        # CSV 行インデックスを計算 (0 ベース pkt_idx)
        base = (pkt_idx * RECORDS_PER_PACKET) % n_csv_rows
        rec_indices = (base + np.arange(n_records, dtype=np.int32)) % n_csv_rows

        # 角度を取得 (ラジアン)
        # _SCAN_ANGLES 列順: [CH0_H, CH0_V, CH1_H, CH1_V, CH2_H, CH2_V, CH3_H, CH3_V]
        angles = _SCAN_ANGLES[rec_indices]  # shape: (n_records, 8)
        valid_row = _SCAN_VALID[rec_indices]  # shape: (n_records,)

        # 結果バッファ (最大 n_records * 4 点)
        out_list: List[np.ndarray] = []

        for ch in range(4):
            r_mm = ranges_mm[:, ch]
            az   = angles[:, ch * 2]       # 水平角 [rad]
            el   = angles[:, ch * 2 + 1]   # 垂直角 [rad]

            # 有効点マスク: 距離>0, 角度が有限, CSV 行が有効
            valid_pt = (
                (r_mm > 0)
                & valid_row
                & np.isfinite(az)
                & np.isfinite(el)
            )
            if not valid_pt.any():
                continue

            r = r_mm[valid_pt].astype(np.float32) / 1000.0
            az_v = az[valid_pt].astype(np.float32)
            el_v = el[valid_pt].astype(np.float32)

            cos_el = np.cos(el_v)
            x = r * cos_el * np.cos(az_v)
            y = r * cos_el * np.sin(az_v)
            z = r * np.sin(el_v)

            ch_pts = np.empty((valid_pt.sum(), 4), dtype=np.float32)
            ch_pts[:, 0] = x
            ch_pts[:, 1] = y
            ch_pts[:, 2] = z
            ch_pts[:, 3] = flag[valid_pt].astype(np.float32)

            out_list.append(ch_pts)

        if not out_list:
            return np.empty((0, 4), dtype=np.float32)

        return np.concatenate(out_list, axis=0)

    @staticmethod
    def _parse_points_fallback(
        payload: bytes,
        n_points: int,
    ) -> np.ndarray:
        """
        CSV が利用できない場合のフォールバック:
        旧フォーマット (X uint16 mm, Y int16 mm, Z int16 mm, intensity, meta×2)
        として直接変換する。

        無効点 (NaN化対象):
          - X == 0       : 戻り信号なし
          - X == 0xFFFF  : 最大距離オーバーフロー
          - Y or Z == ±int16境界 (-32768 / +32767): 無効測定マーカー
        """
        logger.debug("フォールバックモードで %d 点をデコード中", n_points)

        buf = np.frombuffer(
            payload[: n_points * POINT_SIZE], dtype=np.uint8
        ).reshape(n_points, POINT_SIZE)

        x_raw = (buf[:, 0].astype(np.uint32) | (buf[:, 1].astype(np.uint32) << 8))
        y_raw = (buf[:, 2].astype(np.uint32) | (buf[:, 3].astype(np.uint32) << 8)).astype(np.int32)
        y_raw[y_raw >= 0x8000] -= 0x10000
        z_raw = (buf[:, 4].astype(np.uint32) | (buf[:, 5].astype(np.uint32) << 8)).astype(np.int32)
        z_raw[z_raw >= 0x8000] -= 0x10000
        intens = buf[:, 6]

        x = x_raw.astype(np.float32) / 1000.0
        y = y_raw.astype(np.float32) / 1000.0
        z = z_raw.astype(np.float32) / 1000.0

        invalid = (
            (x_raw == 0)
            | (x_raw == 0xFFFF)
            | (y_raw == 32767) | (y_raw == -32768)
            | (z_raw == 32767) | (z_raw == -32768)
        )
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
        """現在のバッファを PointCloudFrame として返し、バッファをリセット"""
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
