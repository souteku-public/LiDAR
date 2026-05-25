"""
packet_parser のユニットテスト (直交座標フォーマット解析後)

実測パケット P1 1点目: f4 d9 42 03 03 00 11 00 18
  X (uint16 LE): 0xD9F4 = 55,796 mm = 55.80m  前方距離
  Y (int16 LE):  0x0342 = 834 mm  = 0.83m    右方向
  Z (int16 LE):  0x0003 = 3 mm    ≈ 0        高さ
  intensity:     0x11 = 17
"""

import struct
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.packet_parser import (
    FalconK2Parser,
    PACKET_MAGIC, HEADER_SIZE, POINT_STRUCT, POINT_SIZE,
)


def _make_header(pkt_idx: int = 0, pkt_size: int = 1440) -> bytes:
    """54バイトのヘッダーを組み立てる"""
    buf = bytearray(HEADER_SIZE)
    buf[0:2]      = PACKET_MAGIC
    buf[2:4]      = (1).to_bytes(2, 'little')
    buf[4:6]      = (0x0190).to_bytes(2, 'little')
    buf[10:14]    = pkt_size.to_bytes(4, 'little')
    buf[14:16]    = (0x03E0).to_bytes(2, 'little')
    buf[0x22:0x24] = pkt_idx.to_bytes(2, 'little')
    return bytes(buf)


def _make_point(x_mm: int, y_mm: int, z_mm: int,
                intensity: int = 100, m1: int = 0, m2: int = 0) -> bytes:
    """X uint16, Y int16, Z int16 形式で 9バイト点を作る"""
    # int16 to bytes
    y_bytes = (y_mm & 0xFFFF).to_bytes(2, 'little')
    z_bytes = (z_mm & 0xFFFF).to_bytes(2, 'little')
    return x_mm.to_bytes(2, 'little') + y_bytes + z_bytes + bytes([intensity, m1, m2])


def _make_packet(pkt_idx: int, points: list, pkt_size: int = None) -> bytes:
    body = b"".join(_make_point(*p) if isinstance(p, tuple) else p for p in points)
    if pkt_size is None:
        pkt_size = HEADER_SIZE + len(body)
    return _make_header(pkt_idx=pkt_idx, pkt_size=pkt_size) + body


# ── テストケース ──────────────────────────────────────────────────────────────

class TestFalconK2Parser:

    def test_magic_confirmed(self):
        """マジック 0x6a17 を受け付ける"""
        parser = FalconK2Parser()
        pkt = _make_packet(pkt_idx=1, points=[(1000, 0, 0)])
        assert parser.feed(pkt) is None  # まだフレーム未確定

    def test_wrong_magic_rejected(self):
        parser = FalconK2Parser()
        bad = b"\xAB\xCD" + _make_packet(0, [(1000, 0, 0)])[2:]
        assert parser.feed(bad) is None

    def test_size_mismatch_rejected(self):
        """ヘッダーが示すサイズと実サイズの不一致を弾く"""
        parser = FalconK2Parser()
        pkt = bytearray(_make_packet(0, [(1000, 0, 0)]))
        pkt[10:14] = (9999).to_bytes(4, 'little')
        assert parser.feed(bytes(pkt)) is None

    def test_x_zero_becomes_nan(self):
        """X=0 (戻り信号なし) の点は NaN になる"""
        parser = FalconK2Parser()
        pts = [(0, 0, 0), (5000, 100, 50)]
        parser.feed(_make_packet(pkt_idx=1, points=pts))
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 2
        assert np.isnan(frame.points[0, 0])
        assert np.isfinite(frame.points[1, 0])

    def test_xyz_units_meters(self):
        """mm → m 変換が正しい"""
        parser = FalconK2Parser()
        # 10000 mm = 10 m, 5000 mm = 5 m
        parser.feed(_make_packet(pkt_idx=1, points=[(10000, 5000, 2000)]))
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert abs(frame.points[0, 0] - 10.0) < 1e-3
        assert abs(frame.points[0, 1] - 5.0)  < 1e-3
        assert abs(frame.points[0, 2] - 2.0)  < 1e-3

    def test_negative_y_z(self):
        """Y/Z の負値 (int16 signed) が正しくデコードされる"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(pkt_idx=1, points=[(10000, -3000, -1500)]))
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert abs(frame.points[0, 0] - 10.0)   < 1e-3
        assert abs(frame.points[0, 1] - (-3.0)) < 1e-3
        assert abs(frame.points[0, 2] - (-1.5)) < 1e-3

    def test_intensity_preserved(self):
        parser = FalconK2Parser()
        parser.feed(_make_packet(pkt_idx=1, points=[(5000, 0, 0, 200)]))
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points[0, 3] == 200.0

    def test_realworld_first_point(self):
        """
        実測パケット P1 1点目: f4 d9 42 03 03 00 11 00 18
          X = 0xD9F4 = 55,796 mm = 55.796 m
          Y = 0x0342 = 834 mm    = 0.834 m
          Z = 0x0003 = 3 mm      = 0.003 m
          intensity = 0x11 = 17
        """
        parser = FalconK2Parser()
        # ヘッダー54B + 点9B = 63B
        hdr = bytearray(_make_header(pkt_idx=1, pkt_size=63))
        point1 = bytes.fromhex("f4d942030300110018")
        pkt = bytes(hdr) + point1

        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 1

        x, y, z, intensity = frame.points[0]
        assert abs(x - 55.796) < 0.001
        assert abs(y - 0.834)  < 0.001
        assert abs(z - 0.003)  < 0.001
        assert intensity == 17.0

    def test_realworld_scan_progression(self):
        """
        実測5パケットの 1点目を順次デコードし、
        X(前方距離) が単調増加、Y(横方向) が単調減少することを確認
        """
        # P1〜P5 1点目の hex
        points_hex = [
            "f4d942030300110018",  # P1: X=55.796, Y=0.834, Z=0.003
            "c9dcc602000003b0018",  # P2: typo を回避 ← 9バイト
            "a4df53020000650018",  # P3
            "85e2eb010000 8f0018",  # P4
            "6ce58b010000b90018",  # P5
        ]
        # 整形しなおし
        points_hex = [
            "f4d942030300110018",
            "c9dcc60200003b0018",
            "a4df53020000650018",
            "85e2eb0100008f0018",
            "6ce58b010000b90018",
        ]

        results = []
        parser = FalconK2Parser()
        for i, hp in enumerate(points_hex):
            hdr = bytearray(_make_header(pkt_idx=i + 1, pkt_size=63))
            pkt = bytes(hdr) + bytes.fromhex(hp)
            parser.feed(pkt)
        # フレーム境界として pkt_idx=0 を投入
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 5

        # X が単調増加
        xs = frame.points[:, 0]
        for i in range(1, len(xs)):
            assert xs[i] > xs[i-1], f"X が単調増加していない: {xs}"

        # Y が単調減少
        ys = frame.points[:, 1]
        for i in range(1, len(ys)):
            assert ys[i] < ys[i-1], f"Y が単調減少していない: {ys}"

    def test_full_packet_154_points(self):
        """1440Bパケットで 154 点が抽出される"""
        pts = [(1000 + i, 0, 0) for i in range(154)]
        pkt = _make_packet(pkt_idx=1, points=pts, pkt_size=1440)
        assert len(pkt) == 1440

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 154

    def test_tail_packet_33_points(self):
        """351Bパケットで 33 点が抽出される"""
        pts = [(1000 + i, 0, 0) for i in range(33)]
        pkt = _make_packet(pkt_idx=10, points=pts, pkt_size=351)
        assert len(pkt) == 351

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 33

    def test_frame_boundary_via_pkt_idx_reset(self):
        """packet_index 減少で前フレーム確定"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        parser.feed(_make_packet(pkt_idx=1, points=[(1100, 0, 0)]))
        parser.feed(_make_packet(pkt_idx=2, points=[(1200, 0, 0)]))

        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 3

    def test_short_packet_rejected(self):
        parser = FalconK2Parser()
        assert parser.feed(b"\x6a\x17\x01\x00") is None

    def test_empty_bytes_rejected(self):
        parser = FalconK2Parser()
        assert parser.feed(b"") is None
