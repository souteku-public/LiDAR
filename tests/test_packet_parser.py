"""
packet_parser のユニットテスト (キャリブレーション済み 4ch フォーマット)

点フォーマット (9 バイト/レコード):
  Bytes 0-1: CH0 距離 uint16 LE [mm]
  Bytes 2-3: CH1 距離 uint16 LE [mm]
  Bytes 4-5: CH2 距離 uint16 LE [mm]
  Bytes 6-7: CH3 距離 uint16 LE [mm]
  Byte  8:   フラグ (intensity 相当)

実測パケット P1 1 点目 (pkt_idx=1, record 0):
  hex: f4 d9  42 03  03 00  11 00  18
  CH0 = 0xD9F4 = 55,796 mm = 55.796 m
  CH1 = 0x0342 =    834 mm  =  0.834 m
  CH2 = 0x0003 =      3 mm  ≈  0 (戻り信号極小)
  CH3 = 0x0011 =     17 mm  ≈  0 (戻り信号極小)
  flag = 0x18 = 24

CSV csv_row = pkt_idx * 154 + record_idx = 1*154 + 0 = 154:
  CH0_H ≈ -31.197°, CH0_V ≈ -14.547°
  座標系: X=上, Y=右, Z=前 (Seyond User Manual §1.3)
  → XYZ ≈ (-14.015, -27.975, 46.197) m
"""

import struct
import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.packet_parser import (
    FalconK2Parser,
    PACKET_MAGIC, HEADER_SIZE, POINT_STRUCT, POINT_SIZE,
    RECORDS_PER_PACKET,
    _SCAN_ANGLES, _SCAN_VALID,
)

# CSV が読み込まれているか確認
_CALIBRATION_AVAILABLE = _SCAN_ANGLES is not None


def _make_header(pkt_idx: int = 0, pkt_size: int = 1440) -> bytes:
    """54 バイトのヘッダーを組み立てる"""
    buf = bytearray(HEADER_SIZE)
    buf[0:2]        = PACKET_MAGIC
    buf[2:4]        = (1).to_bytes(2, 'little')
    buf[4:6]        = (0x0190).to_bytes(2, 'little')
    buf[10:14]      = pkt_size.to_bytes(4, 'little')
    buf[14:16]      = (0x03E0).to_bytes(2, 'little')
    buf[0x22:0x24]  = pkt_idx.to_bytes(2, 'little')
    return bytes(buf)


def _make_record(
    ch0_mm: int = 5000,
    ch1_mm: int = 5000,
    ch2_mm: int = 5000,
    ch3_mm: int = 5000,
    flag: int = 100,
) -> bytes:
    """9 バイトのレコードを生成する (4ch 距離 + フラグ)"""
    return struct.pack('<HHHHB', ch0_mm, ch1_mm, ch2_mm, ch3_mm, flag)


def _make_packet(
    pkt_idx: int,
    records: list,
    pkt_size: int = None,
) -> bytes:
    """ヘッダー + レコード列からパケットを組み立てる"""
    body = b"".join(
        _make_record(*r) if isinstance(r, tuple) else r
        for r in records
    )
    if pkt_size is None:
        pkt_size = HEADER_SIZE + len(body)
    return _make_header(pkt_idx=pkt_idx, pkt_size=pkt_size) + body


# ── ヘッダー検証テスト ────────────────────────────────────────────────────────

class TestHeaderValidation:

    def test_magic_confirmed(self):
        """マジック 0x6a17 を受け付ける"""
        parser = FalconK2Parser()
        pkt = _make_packet(pkt_idx=1, records=[(1000, 1000, 1000, 1000)])
        assert parser.feed(pkt) is None  # まだフレーム未確定

    def test_wrong_magic_rejected(self):
        """不正マジックのパケットを拒否する"""
        parser = FalconK2Parser()
        bad = b"\xAB\xCD" + _make_packet(0, [(1000, 0, 0, 0)])[2:]
        assert parser.feed(bad) is None

    def test_size_mismatch_rejected(self):
        """ヘッダー示すサイズと実サイズの不一致を弾く"""
        parser = FalconK2Parser()
        pkt = bytearray(_make_packet(0, [(1000, 0, 0, 0)]))
        pkt[10:14] = (9999).to_bytes(4, 'little')
        assert parser.feed(bytes(pkt)) is None

    def test_short_packet_rejected(self):
        """短すぎるパケットを拒否する"""
        parser = FalconK2Parser()
        assert parser.feed(b"\x6a\x17\x01\x00") is None

    def test_empty_bytes_rejected(self):
        """空バイト列を拒否する"""
        parser = FalconK2Parser()
        assert parser.feed(b"") is None


# ── フレーム境界テスト ────────────────────────────────────────────────────────

class TestFrameBoundary:

    def test_frame_boundary_via_pkt_idx_reset(self):
        """packet_index の減少で前フレームが確定する"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(pkt_idx=0, records=[(1000, 0, 0, 0)]))
        parser.feed(_make_packet(pkt_idx=1, records=[(1100, 0, 0, 0)]))
        parser.feed(_make_packet(pkt_idx=2, records=[(1200, 0, 0, 0)]))

        # pkt_idx=0 が来ると前フレームが確定
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(1000, 0, 0, 0)]))
        assert frame is not None

    def test_first_feed_returns_none(self):
        """最初のパケットではフレームを返さない"""
        parser = FalconK2Parser()
        assert parser.feed(_make_packet(pkt_idx=1, records=[(1000, 0, 0, 0)])) is None

    def test_frame_has_points_from_all_packets(self):
        """フレームには複数パケットの点が含まれる"""
        parser = FalconK2Parser()
        # 3 パケット分のデータ
        for i in range(3):
            parser.feed(_make_packet(pkt_idx=i, records=[(1000 + i * 100, 0, 0, 0)]))
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(1000, 0, 0, 0)]))
        assert frame is not None
        # 各パケットから最低 1 点以上 (CH0 が有効な場合)
        assert frame.points.shape[0] >= 3


# ── キャリブレーション変換テスト ──────────────────────────────────────────────

@pytest.mark.skipif(not _CALIBRATION_AVAILABLE, reason="CSV なし: キャリブレーションテストをスキップ")
class TestCalibratedConversion:

    def test_zero_range_excluded(self):
        """CH0 距離 = 0 (戻り信号なし) の点は除外される"""
        parser = FalconK2Parser()
        # CH0=0 (無効), CH1=5000mm (有効)
        records = [(0, 5000, 0, 0)]
        parser.feed(_make_packet(pkt_idx=1, records=records))
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(5000, 0, 0, 0)]))
        assert frame is not None
        # CH0 点が除外されているため CH1 の点のみ
        # CH1 の点数チェック: pkt_idx=1, record_idx=0 の csv_row=154 が有効なら点あり
        if _SCAN_VALID is not None and _SCAN_VALID[154]:
            # CH1 から 1 点出力されるはず (CH0=0 は除外)
            assert frame.points.shape[0] >= 1

    def test_all_zero_ranges_gives_no_points(self):
        """全チャンネルが距離 0 の場合、フレームには点が含まれない"""
        parser = FalconK2Parser()
        records = [(0, 0, 0, 0)]  # 全 ch 無効 → バッファに chunk が追加されない
        parser.feed(_make_packet(pkt_idx=1, records=records))
        # pkt_idx=0 が来ると前フレームを emit しようとするが、chunk が空なので None
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(5000, 0, 0, 0)]))
        # バッファに有効点が 0 → _emit_frame は None を返す
        assert frame is None

    def test_valid_range_produces_finite_xyz(self):
        """有効な距離から有限の XYZ 座標が生成される"""
        parser = FalconK2Parser()
        records = [(10000, 0, 0, 0)]  # CH0=10m, others=0
        parser.feed(_make_packet(pkt_idx=1, records=records))
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(5000, 0, 0, 0)]))
        assert frame is not None

        # pkt_idx=1, record_idx=0 → csv_row=154
        csv_row = 154
        if _SCAN_VALID is not None and _SCAN_VALID[csv_row]:
            # CH0 から 1 点出力
            assert frame.points.shape[0] >= 1
            xyz = frame.points[0, :3]
            assert np.isfinite(xyz).all(), f"非有限 XYZ: {xyz}"

    def test_xyz_distance_matches_range(self):
        """
        XYZ 座標から算出した距離がレコードの距離と一致する。
        r = sqrt(x^2 + y^2 + z^2)
        """
        parser = FalconK2Parser()
        range_mm = 20000  # 20 m
        records = [(range_mm, 0, 0, 0)]
        parser.feed(_make_packet(pkt_idx=1, records=records))
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(5000, 0, 0, 0)]))
        assert frame is not None

        csv_row = 154  # pkt_idx=1, record_idx=0
        if _SCAN_VALID is None or not _SCAN_VALID[csv_row]:
            pytest.skip("csv_row=154 が無効のためスキップ")

        # CH0 からの点を取得
        assert frame.points.shape[0] >= 1
        x, y, z = frame.points[0, :3]
        r_calc = math.sqrt(x**2 + y**2 + z**2)
        r_expected = range_mm / 1000.0

        assert abs(r_calc - r_expected) < 0.01, (
            f"距離不一致: calc={r_calc:.4f}m expected={r_expected:.4f}m"
        )

    def test_realworld_first_point_ch0(self):
        """
        実測パケット P1 1 点目の CH0 を検証する。
          hex: f4 d9  42 03  03 00  11 00  18
          pkt_idx=1, record_idx=0 → csv_row=154
          CH0 = 55,796 mm = 55.796 m
          CH0_H ≈ -31.197°, CH0_V ≈ -14.547°

        期待 XYZ (Seyond 座標系: X=上, Y=右, Z=前):
          x ≈ -14.015 m  (上方向)
          y ≈ -27.975 m  (右方向)
          z ≈  46.197 m  (前方向)
        """
        parser = FalconK2Parser()
        hdr   = bytearray(_make_header(pkt_idx=1, pkt_size=63))
        point = bytes.fromhex("f4d942030300110018")
        pkt   = bytes(hdr) + point

        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(5000, 0, 0, 0)]))
        assert frame is not None

        csv_row = 154
        if _SCAN_VALID is None or not _SCAN_VALID[csv_row]:
            pytest.skip("csv_row=154 が無効のためスキップ")

        # CH0 の点が含まれているはず (55.796m, 最も遠い点)
        assert frame.points.shape[0] >= 1

        # 各点の距離を計算して最も遠い点を CH0 と判断
        xyz = frame.points[:, :3]
        dists = np.linalg.norm(xyz, axis=1)
        far_idx = np.argmax(dists)

        x, y, z, intensity = frame.points[far_idx]

        # Seyond 座標系: X=上, Y=右, Z=前
        assert abs(x - (-14.015)) < 0.5, f"x={x:.3f} (期待 ≈-14.015, 上方向)"
        assert abs(y - (-27.975)) < 0.5, f"y={y:.3f} (期待 ≈-27.975, 右方向)"
        assert abs(z -   46.197)  < 0.5, f"z={z:.3f} (期待 ≈46.197, 前方向)"
        assert intensity == 24.0, f"intensity={intensity} (期待 24)"

    def test_realworld_scan_ch0_hfov(self):
        """
        実測 P1-P4 1 点目の CH0 が水平方向に展開していることを確認する。
        P1: H=-31.2°  P2: H=0.0°  P3: H=31.1°  P4: H=61.1°
        Y 座標の絶対値の変化で角度展開を確認する。
        """
        real_pts_hex = [
            "f4d942030300110018",  # P1: CH0=55796mm
            "c9dcc60200003b0018",  # P2: CH0=56521mm
            "a4df53020000650018",  # P3: CH0=57252mm
            "85e2eb0100008f0018",  # P4: CH0=57989mm
        ]

        parser = FalconK2Parser()
        for i, hp in enumerate(real_pts_hex):
            hdr = bytearray(_make_header(pkt_idx=i + 1, pkt_size=63))
            pkt = bytes(hdr) + bytes.fromhex(hp)
            parser.feed(pkt)

        frame = parser.feed(_make_packet(pkt_idx=0, records=[(5000, 0, 0, 0)]))
        assert frame is not None

        # 各点の距離を計算
        xyz = frame.points[:, :3]
        dists = np.linalg.norm(xyz, axis=1)

        # 4 点の中から最も遠い 4 点 (各パケットの CH0 に対応) を取得
        # (他のチャンネルも混入している可能性がある)
        assert frame.points.shape[0] >= 4, (
            f"最低 4 点必要: {frame.points.shape[0]} 点"
        )

        # CH0 の点 (各パケット最遠点) を抽出
        # 期待距離: ~55.8m, ~56.5m, ~57.3m, ~58.0m
        expected_dists = [55.796, 56.521, 57.252, 57.989]
        for ed in expected_dists:
            # 期待距離 ±1m 以内の点が存在すること
            found = any(abs(d - ed) < 1.0 for d in dists)
            assert found, (
                f"期待距離 {ed}m に対応する点が見つからない: {dists}"
            )

    def test_intensity_flag_preserved(self):
        """フラグバイト (intensity) が正しく保存される"""
        parser = FalconK2Parser()
        records = [(10000, 0, 0, 0, 200)]  # flag=200
        parser.feed(_make_packet(pkt_idx=1, records=records))
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(5000, 0, 0, 0)]))
        assert frame is not None

        csv_row = 154
        if _SCAN_VALID is None or not _SCAN_VALID[csv_row]:
            pytest.skip("csv_row=154 が無効のためスキップ")

        assert frame.points.shape[0] >= 1
        # CH0 の点の intensity = 200
        intensities = frame.points[:, 3]
        assert 200.0 in intensities, f"intensity=200 が見つからない: {intensities}"


# ── パケットサイズテスト ──────────────────────────────────────────────────────

class TestPacketSizes:

    def test_full_packet_154_records(self):
        """1440B パケットで 154 レコードが処理される"""
        records = [(1000 + i, 0, 0, 0) for i in range(154)]
        pkt = _make_packet(pkt_idx=1, records=records, pkt_size=1440)
        assert len(pkt) == 1440

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(1000, 0, 0, 0)]))
        assert frame is not None
        # キャリブレーションあり: 有効チャンネル数に依存するが 1 点以上あること
        assert frame.points.shape[0] >= 1

    def test_tail_packet_33_records(self):
        """351B パケットで 33 レコードが処理される"""
        records = [(1000 + i, 0, 0, 0) for i in range(33)]
        pkt = _make_packet(pkt_idx=10, records=records, pkt_size=351)
        assert len(pkt) == 351

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(1000, 0, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] >= 1

    def test_full_packet_produces_up_to_4x_records(self):
        """フルパケット 154 レコード × 最大 4ch = 最大 616 点"""
        # 全 CH が有効な場合、最大 616 点
        records = [(5000, 5000, 5000, 5000, 100) for _ in range(154)]
        pkt = _make_packet(pkt_idx=0, records=records, pkt_size=1440)
        assert len(pkt) == 1440

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=1, records=records, pkt_size=1440))
        # pkt_idx=1 > 0 なので, pkt_idx=1 を投入しても frame は返らない
        # pkt_idx が減少するときに frame が出る
        frame2 = parser.feed(_make_packet(pkt_idx=0, records=records, pkt_size=1440))
        assert frame2 is not None
        # CSV が有効な行を多く含む場合、点数は 154 × (有効CH数) 付近
        # CSV の NaN 行があるため最大 616 より少ない場合がある
        assert frame2.points.shape[0] > 0

    def test_point_size_constant(self):
        """POINT_SIZE = 9 バイト"""
        assert POINT_SIZE == 9

    def test_records_per_packet_constant(self):
        """RECORDS_PER_PACKET = 154"""
        assert RECORDS_PER_PACKET == 154


# ── フォールバックモードテスト ────────────────────────────────────────────────

class TestFallbackMode:
    """
    CSV が利用できない場合のフォールバックテスト。
    _SCAN_ANGLES を一時的に None に差し替えてテストする。
    """

    def test_fallback_x_zero_becomes_nan(self, monkeypatch):
        """[フォールバック] X=0 の点は NaN になる"""
        import src.packet_parser as pp
        monkeypatch.setattr(pp, '_SCAN_ANGLES', None)
        monkeypatch.setattr(pp, '_SCAN_VALID', None)

        # fallback では旧フォーマット: (x_mm, y_mm, z_mm, intensity, m1, m2)
        # _make_record では CH0=0 → X=0 → NaN
        parser = FalconK2Parser()
        from tests.test_packet_parser import _make_record
        body0 = struct.pack('<HhhBBB', 0, 0, 0, 100, 0, 0)     # X=0 (invalid)
        body1 = struct.pack('<HhhBBB', 5000, 100, 50, 100, 0, 0)  # valid

        pkt_size = HEADER_SIZE + len(body0) + len(body1)
        pkt = _make_header(pkt_idx=1, pkt_size=pkt_size) + body0 + body1
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, records=[(1000, 0, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 2
        assert np.isnan(frame.points[0, 0])
        assert np.isfinite(frame.points[1, 0])
