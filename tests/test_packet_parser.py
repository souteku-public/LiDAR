"""
packet_parser のユニットテスト (実測プロトコル準拠)

実測ヘッダー例:
  6a 17 01 00 90 01 d2 e1 6d 3a a0 05 00 00 e0 03 62 10 08 5a 2e d3 de 41

  magic=6a17, version=0x0001, model=0x0190(400),
  sub_counter=変化, pkt_size=1440, frame_id=0x03e0(992)
"""

import struct
import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.packet_parser import (
    FalconK2Parser,
    HEADER_STRUCT, HEADER_SIZE,
    POINT_STRUCT,  POINT_SIZE,
    PACKET_MAGIC,
)


def _make_packet(
    frame_id: int = 1,
    sub_counter: int = 0,
    points=None,      # list of (x, y, z)
    version: int = 1,
    model_cfg: int = 0x0190,
) -> bytes:
    """テスト用 UDP パケットを組み立てる"""
    if points is None:
        points = [(1.0, 2.0, 3.0)]

    body = b"".join(POINT_STRUCT.pack(x, y, z) for x, y, z in points)
    pkt_size = HEADER_SIZE + len(body)

    header = HEADER_STRUCT.pack(
        PACKET_MAGIC,
        version,
        model_cfg,
        sub_counter,
        pkt_size,
        frame_id,
        b'\x00' * 8,   # timestamp bytes (8 bytes)
    )
    return header + body


# ── テストケース ──────────────────────────────────────────────────────────────

class TestFalconK2Parser:

    def test_magic_confirmed(self):
        """実測値と同じマジックバイト 0x6a17 が受け付けられる"""
        parser = FalconK2Parser()
        pkt = _make_packet(frame_id=992, points=[(1.0, 2.0, 3.0)] * 5)
        # 1パケットだけでは frame_id 変化が起きないので None
        result = parser.feed(pkt)
        # まだ frame_id は変化していないので None
        assert result is None

    def test_frame_emitted_on_frame_id_change(self):
        """frame_id が変わったタイミングで前フレームが返される"""
        parser = FalconK2Parser()
        pts = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        pkt_f1 = _make_packet(frame_id=10, points=pts)
        pkt_f2 = _make_packet(frame_id=11, points=[(0.0, 0.0, 0.0)])

        # フレーム10の1パケット目
        assert parser.feed(pkt_f1) is None
        # フレーム11が来た → フレーム10が確定
        frame = parser.feed(pkt_f2)
        assert frame is not None
        assert frame.frame_id == 10
        assert frame.points.shape == (2, 4)

    def test_multi_packet_same_frame_accumulated(self):
        """同じ frame_id のパケットが複数来ると点群が結合される"""
        parser = FalconK2Parser()
        pts_a = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        pts_b = [(3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (5.0, 0.0, 0.0)]

        parser.feed(_make_packet(frame_id=5, points=pts_a))
        parser.feed(_make_packet(frame_id=5, points=pts_b))

        # frame_id を変えて強制確定
        frame = parser.feed(_make_packet(frame_id=6, points=[(0.0, 0.0, 0.0)]))
        assert frame is not None
        assert frame.frame_id == 5
        assert frame.points.shape[0] == 5  # 2 + 3

    def test_xyz_values_preserved(self):
        """XYZ 座標が正しく float32 で保存される"""
        parser = FalconK2Parser()
        pts = [(1.5, -2.5, 3.5)]
        parser.feed(_make_packet(frame_id=1, points=pts))
        frame = parser.feed(_make_packet(frame_id=2, points=[(0.0, 0.0, 0.0)]))
        assert frame is not None
        np.testing.assert_allclose(frame.points[0, :3], [1.5, -2.5, 3.5], rtol=1e-5)

    def test_intensity_defaults_to_255(self):
        """intensity チャンネルは 255 で埋められる"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(frame_id=1, points=[(1.0, 2.0, 3.0)]))
        frame = parser.feed(_make_packet(frame_id=2, points=[(0.0, 0.0, 0.0)]))
        assert frame is not None
        assert frame.points[0, 3] == 255.0

    def test_wrong_magic_rejected(self):
        """マジックバイトが違うパケットは無視される"""
        parser = FalconK2Parser()
        bad = b'\xAB\xCD' + _make_packet()[2:]
        assert parser.feed(bad) is None
        # フレームは累積されない
        good = _make_packet(frame_id=2, points=[(0.0, 0.0, 0.0)])
        result = parser.feed(good)
        # フレーム1は累積されていないはずなのでNone
        assert result is None

    def test_short_packet_rejected(self):
        """24バイト未満のパケットは無視される"""
        parser = FalconK2Parser()
        assert parser.feed(b'\x6a\x17\x01\x00') is None

    def test_empty_bytes_rejected(self):
        parser = FalconK2Parser()
        assert parser.feed(b"") is None

    def test_realworld_header_accepted(self):
        """実測パケット先頭24バイト(+ダミー点)がパースできる"""
        # 実測ヘッダー: 6a17 0100 9001 d2e16d3a a0050000 e003 621008 5a2ed3de41
        header_hex = "6a170100900 1d2e16d3aa0050000e003621008 5a2ed3de41"
        # 実測先頭24B
        real_header = bytes.fromhex("6a17010090 01d2e16d3aa0050000e003621008 5a2ed3de41".replace(" ", ""))
        assert len(real_header) == 24

        # ダミーの点データを追加 (float32 x,y,z = 12 bytes × 1点)
        point_bytes = struct.pack('<fff', 1.0, 2.0, 3.0)
        pkt = real_header + point_bytes

        parser = FalconK2Parser()
        result = parser.feed(pkt)
        # frame_id=992, まだ次フレームが来ていないので None
        assert result is None

        # frame_id を変えて確定させる
        next_pkt = _make_packet(frame_id=993, points=[(0.0, 0.0, 0.0)])
        frame = parser.feed(next_pkt)
        assert frame is not None
        assert frame.frame_id == 992
        assert frame.points.shape == (1, 4)
        np.testing.assert_allclose(frame.points[0, :3], [1.0, 2.0, 3.0], rtol=1e-5)

    def test_timeout_emits_frame(self):
        """タイムアウトで未確定フレームが強制出力される"""
        from src.packet_parser import FRAME_TIMEOUT_S
        parser = FalconK2Parser()
        pts = [(1.0, 2.0, 3.0)]
        pkt = _make_packet(frame_id=100, points=pts)
        parser.feed(pkt)

        # バッファの last_update を古い時刻に書き換えてタイムアウトを模擬
        buf = parser._buffers[100]
        buf.last_update -= (FRAME_TIMEOUT_S + 0.1)

        # 別フレームのパケットを投入して timeout チェックをトリガー
        frame = parser.feed(_make_packet(frame_id=100, points=[(0.0, 0.0, 0.0)]))
        # タイムアウト対象は存在しない(同 frame_id は除外) → None
        assert frame is None

        # 違う frame_id で来るとタイムアウトフレームが返る
        parser._buffers[100].last_update -= (FRAME_TIMEOUT_S + 0.1)
        frame = parser.feed(_make_packet(frame_id=101, points=[(0.0, 0.0, 0.0)]))
        # frame_id=100 がタイムアウトまたは確定 (frame_id 変化で確定)
        assert frame is not None

    def test_reset_clears_state(self):
        """reset() 後は未完成フレームのバッファが消える"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(frame_id=50, points=[(1.0, 2.0, 3.0)]))
        parser.reset()
        # reset後は frame_id=50 のバッファが消えているので、
        # 新しい frame_id=51 を入れても前フレームは返らない
        frame = parser.feed(_make_packet(frame_id=51, points=[(0.0, 0.0, 0.0)]))
        assert frame is None  # フレーム50は消えているのでNone

    def test_full_packet_118_points(self):
        """実際のフルパケット(1440 bytes)では 118 点が抽出される"""
        # ヘッダー24 + 118点×12 = 24 + 1416 = 1440 bytes
        pts = [(float(i), float(i), float(i)) for i in range(118)]
        pkt = _make_packet(frame_id=1, points=pts)
        assert len(pkt) == 1440

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(frame_id=2, points=[(0.0, 0.0, 0.0)]))
        assert frame is not None
        assert frame.points.shape[0] == 118

    def test_partial_packet_handles_remainder(self):
        """端数バイト(3バイト)は無視されクラッシュしない"""
        pts = [(1.0, 2.0, 3.0)] * 27
        body = POINT_STRUCT.pack(1.0, 2.0, 3.0) * 27 + b'\x00\x00\x00'  # 3バイト余り
        pkt_size = HEADER_SIZE + len(body)
        header = HEADER_STRUCT.pack(PACKET_MAGIC, 1, 0x190, 0, pkt_size, 999, b'\x00'*8)
        pkt = header + body

        parser = FalconK2Parser()
        parser.feed(pkt)
        frame = parser.feed(_make_packet(frame_id=1000, points=[(0.0, 0.0, 0.0)]))
        assert frame is not None
        assert frame.points.shape[0] == 27  # 余り3バイトは無視
