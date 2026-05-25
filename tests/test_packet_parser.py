"""
packet_parser のユニットテスト (実測パケット解析後)

実測パケット先頭64B (P1):
  6a 17 01 00 90 01 29 a5 2e 9f a0 05 00 00 e0 03
  19 04 15 0e c0 6d 01 42 03 02 63 6c 01 00 00 00
  00 00 30 01 72 5a 01 2a 00 00 21 00 00 00 00 00
  27 01 00 00 00 00 f4 d9 42 03 03 00 11 00 18 b8

確定フォーマット:
  Header: 54 bytes
  Point:  9 bytes (range u16 + az u16 + el u16 + intensity + ch + flag)
  Full packet (1440B): 154 points
  Tail packet (351B):   33 points
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
)


def _make_header(
    pkt_idx: int = 0,
    pkt_size: int = 1440,
    sub_counter: int = 0,
) -> bytes:
    """54バイトのヘッダーを組み立てる"""
    buf = bytearray(HEADER_SIZE)
    buf[0:2]   = PACKET_MAGIC
    buf[2:4]   = (1).to_bytes(2, 'little')          # version
    buf[4:6]   = (0x0190).to_bytes(2, 'little')     # model_cfg
    buf[6:10]  = sub_counter.to_bytes(4, 'little')  # sub_counter
    buf[10:14] = pkt_size.to_bytes(4, 'little')     # packet_size
    buf[14:16] = (0x03E0).to_bytes(2, 'little')     # constant
    buf[0x22:0x24] = pkt_idx.to_bytes(2, 'little')  # packet_index
    return bytes(buf)


def _make_point(range_mm: int, az_001: int, el_001: int,
                intensity: int = 100, channel: int = 0, flag: int = 0) -> bytes:
    return POINT_STRUCT.pack(range_mm, az_001, el_001, intensity, channel, flag)


def _make_packet(pkt_idx: int, points: list, pkt_size: int = None) -> bytes:
    body = b"".join(_make_point(*p) if isinstance(p, tuple) else p for p in points)
    if pkt_size is None:
        pkt_size = HEADER_SIZE + len(body)
    return _make_header(pkt_idx=pkt_idx, pkt_size=pkt_size) + body


# ── テストケース ──────────────────────────────────────────────────────────────

class TestFalconK2Parser:

    def test_magic_confirmed(self):
        """マジック 0x6a17 が受け付けられる"""
        parser = FalconK2Parser()
        pkt = _make_packet(pkt_idx=1, points=[(10000, 0, 0)])
        # 1パケット目はバッファに蓄積されるだけで None
        assert parser.feed(pkt) is None

    def test_wrong_magic_rejected(self):
        parser = FalconK2Parser()
        bad = b"\xAB\xCD" + _make_packet(0, [(1000, 0, 0)])[2:]
        assert parser.feed(bad) is None

    def test_packet_size_mismatch_rejected(self):
        """ヘッダーが示すサイズと実サイズが不一致なら拒否される"""
        parser = FalconK2Parser()
        # サイズフィールドだけ嘘の値にする
        pkt = bytearray(_make_packet(0, [(1000, 0, 0)]))
        pkt[10:14] = (9999).to_bytes(4, 'little')   # 嘘のサイズ
        assert parser.feed(bytes(pkt)) is None

    def test_range_zero_to_nan(self):
        """range=0 の点は NaN になる"""
        parser = FalconK2Parser()
        pts = [(0, 0, 0), (5000, 0, 0)]
        parser.feed(_make_packet(pkt_idx=1, points=pts))
        # pkt_idx 減少で前フレーム確定
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 2
        assert np.isnan(frame.points[0, 0])
        assert np.isfinite(frame.points[1, 0])

    def test_xyz_calculation(self):
        """range=10m, az=0, el=0 なら x=10, y=0, z=0 のはず"""
        parser = FalconK2Parser()
        # 10000 mm = 10 m, az=0, el=0
        parser.feed(_make_packet(pkt_idx=1, points=[(10000, 0, 0)]))
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        x, y, z = frame.points[0, :3]
        assert abs(x - 10.0) < 1e-3
        assert abs(y) < 1e-3
        assert abs(z) < 1e-3

    def test_xyz_90deg_azimuth(self):
        """range=10m, az=90度, el=0 なら x=0, y=10, z=0 のはず"""
        parser = FalconK2Parser()
        # 9000 * 0.01度 = 90度
        parser.feed(_make_packet(pkt_idx=1, points=[(10000, 9000, 0)]))
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        x, y, z = frame.points[0, :3]
        assert abs(x) < 1e-3
        assert abs(y - 10.0) < 1e-3
        assert abs(z) < 1e-3

    def test_xyz_45deg_elevation(self):
        """range=10m, az=0, el=45度 なら x=z=7.07m, y=0"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(pkt_idx=1, points=[(10000, 0, 4500)]))   # 45度
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        x, y, z = frame.points[0, :3]
        expected = 10.0 * math.cos(math.radians(45))
        assert abs(x - expected) < 1e-2
        assert abs(y) < 1e-3
        assert abs(z - expected) < 1e-2

    def test_intensity_preserved(self):
        """intensity が点群に保持される"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(pkt_idx=1, points=[(5000, 0, 0, 200)]))
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points[0, 3] == 200.0

    def test_realworld_first_point(self):
        """
        実測パケット P1 の最初の点をデコードして妥当性を確認:
          点1: f4 d9 42 03 03 00 11 00 18
          range = 0xD9F4 = 55796 mm = 55.796 m
          az    = 0x0342 = 834   * 0.01 deg = 8.34 deg
          el    = 0x0003 = 3     * 0.01 deg = 0.03 deg
          intensity = 0x11 = 17
        """
        parser = FalconK2Parser()
        real_header_hex = (
            "6a17010090012 9a52e9fa0050000e003"        # 0x00-0x0F
            "1904150ec06d014203026 36c01000000"        # 0x10-0x1F
            "0000300172 5a012a0000210000000000"        # 0x20-0x2F
            "270100000000"                              # 0x30-0x35
        ).replace(" ", "")
        real_header = bytes.fromhex(real_header_hex)
        assert len(real_header) == 54

        point1 = bytes.fromhex("f4d942030300110018")    # 9 bytes
        # ヘッダーの pkt_size をテスト用に short にする (54 + 9 = 63)
        hdr = bytearray(real_header)
        hdr[10:14] = (63).to_bytes(4, 'little')
        pkt = bytes(hdr) + point1

        parser.feed(pkt)
        # 次フレームで確定 (pkt_idx 減少をシミュレート)
        # 実測の pkt_idx = 0x0130 = 304, so use 0 to force completion
        nxt = _make_packet(pkt_idx=0, points=[(1000, 0, 0)])
        frame = parser.feed(nxt)
        assert frame is not None
        assert frame.points.shape[0] == 1

        x, y, z, intensity = frame.points[0]
        # range = 55.796m, az = 8.34度, el = 0.03度 → 球面 → XYZ
        r  = 55.796
        az = math.radians(8.34)
        el = math.radians(0.03)
        ex = r * math.cos(el) * math.cos(az)
        ey = r * math.cos(el) * math.sin(az)
        ez = r * math.sin(el)
        assert abs(x - ex) < 0.01
        assert abs(y - ey) < 0.01
        assert abs(z - ez) < 0.01
        assert intensity == 17.0

    def test_full_packet_154_points(self):
        """1440バイトパケットでは 154 点が抽出される"""
        # 154 points × 9 bytes = 1386 bytes + 54 header = 1440
        pts = [(1000, i * 100, 0) for i in range(154)]
        pkt = _make_packet(pkt_idx=1, points=pts, pkt_size=1440)
        assert len(pkt) == 1440

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 154

    def test_tail_packet_33_points(self):
        """351バイトパケットでは 33 点が抽出される"""
        # 33 points × 9 bytes = 297 bytes + 54 header = 351
        pts = [(1000, i * 100, 0) for i in range(33)]
        pkt = _make_packet(pkt_idx=10, points=pts, pkt_size=351)
        assert len(pkt) == 351

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 33

    def test_frame_boundary_via_pkt_idx_reset(self):
        """packet_index が減少したら新フレーム開始"""
        parser = FalconK2Parser()
        # フレーム1: pkt_idx 0, 1, 2
        parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        parser.feed(_make_packet(pkt_idx=1, points=[(1000, 100, 0)]))
        parser.feed(_make_packet(pkt_idx=2, points=[(1000, 200, 0)]))

        # フレーム2 開始 (pkt_idx 0) → フレーム1が確定
        frame = parser.feed(_make_packet(pkt_idx=0, points=[(1000, 0, 0)]))
        assert frame is not None
        assert frame.points.shape[0] == 3

    def test_short_packet_rejected(self):
        parser = FalconK2Parser()
        assert parser.feed(b"\x6a\x17\x01\x00") is None

    def test_empty_bytes_rejected(self):
        parser = FalconK2Parser()
        assert parser.feed(b"") is None
