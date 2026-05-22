"""
packet_parser のユニットテスト
"""
import struct
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.packet_parser import (
    FalconK2Parser,
    HEADER_STRUCT, HEADER_SIZE,
    POINT_STRUCT,  POINT_SIZE,
    PACKET_MAGIC, PACKET_TYPE_PCLOUD,
)


def _make_packet(
    frame_id: int = 1,
    pkt_idx: int = 0,
    total_pkts: int = 1,
    timestamp_us: int = 1_000_000,
    points=None,          # list of (x, y, z, intensity, tag)
) -> bytes:
    """テスト用 UDP パケットを組み立てる"""
    if points is None:
        points = [(1.0, 2.0, 3.0, 128, 0)]

    n_points = len(points)
    header = HEADER_STRUCT.pack(
        PACKET_MAGIC,
        PACKET_TYPE_PCLOUD,
        frame_id,
        pkt_idx,
        total_pkts,
        timestamp_us,
        n_points,
        0,  # reserved
    )
    body = b"".join(POINT_STRUCT.pack(x, y, z, inten, tag) for x, y, z, inten, tag in points)
    return header + body


# ── テストケース ──────────────────────────────────────────────────────────────

class TestFalconK2Parser:

    def test_single_packet_frame(self):
        """1 パケットで完結するフレームが正しくアセンブルされる"""
        parser = FalconK2Parser()
        pkt = _make_packet(frame_id=42, pkt_idx=0, total_pkts=1,
                           points=[(1.0, 2.0, 3.0, 255, 0)])
        frame = parser.feed(pkt)
        assert frame is not None
        assert frame.frame_id == 42
        assert frame.points.shape == (1, 4)
        np.testing.assert_allclose(frame.points[0, :3], [1.0, 2.0, 3.0])
        assert frame.points[0, 3] == 255.0

    def test_multi_packet_frame(self):
        """複数パケットが揃ったときだけフレームを返す"""
        parser = FalconK2Parser()
        pts_a = [(0.1, 0.2, 0.3, 10, 0), (0.4, 0.5, 0.6, 20, 0)]
        pts_b = [(1.0, 1.1, 1.2, 30, 0)]

        pkt0 = _make_packet(frame_id=7, pkt_idx=0, total_pkts=2, points=pts_a)
        pkt1 = _make_packet(frame_id=7, pkt_idx=1, total_pkts=2, points=pts_b)

        # 1 枚目はまだ未完成
        assert parser.feed(pkt0) is None
        # 2 枚目で完成
        frame = parser.feed(pkt1)
        assert frame is not None
        assert frame.frame_id == 7
        assert frame.points.shape == (3, 4)

    def test_invalid_magic_rejected(self):
        """不正なマジックバイトを持つパケットは無視される"""
        parser = FalconK2Parser()
        bad_pkt = b"\x00\x00" + _make_packet()[2:]
        assert parser.feed(bad_pkt) is None

    def test_short_packet_rejected(self):
        """ヘッダーより短いパケットは無視される"""
        parser = FalconK2Parser()
        assert parser.feed(b"\xAB\xCD") is None

    def test_empty_bytes_rejected(self):
        parser = FalconK2Parser()
        assert parser.feed(b"") is None

    def test_timestamp_conversion(self):
        """タイムスタンプが正しく秒単位に変換される"""
        parser = FalconK2Parser()
        ts_us = 2_500_000_000  # 2500 秒
        pkt = _make_packet(timestamp_us=ts_us)
        frame = parser.feed(pkt)
        assert frame is not None
        assert abs(frame.timestamp - 2500.0) < 1e-6

    def test_reset_clears_buffers(self):
        """reset() 後は未完成フレームのバッファがクリアされる"""
        parser = FalconK2Parser()
        pts = [(1.0, 2.0, 3.0, 0, 0)]
        pkt0 = _make_packet(frame_id=99, pkt_idx=0, total_pkts=2, points=pts)
        parser.feed(pkt0)
        parser.reset()
        # バッファが消えているので pkt1 を送っても完成しない
        pkt1 = _make_packet(frame_id=99, pkt_idx=1, total_pkts=2, points=pts)
        # フレームバッファが空なので pkt1 は 1/2 として扱われ未完成
        assert parser.feed(pkt1) is None

    def test_multiple_frames_independent(self):
        """異なる frame_id は独立してアセンブルされる"""
        parser = FalconK2Parser()
        pkt_a = _make_packet(frame_id=1, pkt_idx=0, total_pkts=1)
        pkt_b = _make_packet(frame_id=2, pkt_idx=0, total_pkts=1)
        frame_a = parser.feed(pkt_a)
        frame_b = parser.feed(pkt_b)
        assert frame_a is not None and frame_a.frame_id == 1
        assert frame_b is not None and frame_b.frame_id == 2
