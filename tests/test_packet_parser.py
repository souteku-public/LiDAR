"""
packet_parser ユニットテスト (InnoBlock1 フォーマット)

パケット構造:
  54バイト InnoDataPacket ヘッダー + n × 33バイト InnoBlock1

InnoBlock1 (33 bytes):
  bytes  0-1:  h_angle  (int16 LE) [InnoAngleUnit]
  bytes  2-3:  v_angle  (int16 LE) [InnoAngleUnit]
  bytes  4-7:  ts_10us + scan_idx
  bytes  8-16: ビットフィールド (h_diff1/2/3, v_diff1/2/3)
  bytes 17-20: CH0 radius[17] + refl[8] + ...  (uint32 LE)
  bytes 21-24: CH1
  bytes 25-28: CH2
  bytes 29-32: CH3

XYZ (Seyond 座標系: X=上, Y=右, Z=前):
  r  = radius / 200   [m]
  h_rad = h_angle × π/32768
  v_rad = v_angle × π/32768
  x  = r × sin(v_rad)
  y  = r × cos(v_rad) × sin(h_rad)
  z  = r × cos(v_rad) × cos(h_rad)
"""

import math
import struct
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.packet_parser import (
    FalconK2Parser,
    PointCloudFrame,
    PACKET_MAGIC,
    HEADER_SIZE,
    BLOCK1_SIZE,
    BLOCK_HDR_SIZE,
    CHANNEL_PT_SIZE,
    CHANNELS_PER_BLOCK,
    BLOCKS_PER_FULL_PKT,
    METERS_PER_UNIT,
    RAD_PER_UNIT,
    V_ANGLE_BASE_PER_CH,
    OFFSET_PKT_SIZE,
    OFFSET_FRAME_IDX,
    OFFSET_TYPE_NUM,
    OFFSET_ITEM_SIZE,
    INNO_ITEM_TYPE_SPHERE_POINTCLOUD,
    _parse_blocks,
)


# ── テストヘルパー ────────────────────────────────────────────────────────────

def _make_header(
    frame_idx: int = 0,
    n_blocks: int = 1,
    pkt_size: int = None,
    pkt_type: int = INNO_ITEM_TYPE_SPHERE_POINTCLOUD,
    item_size: int = BLOCK1_SIZE,
) -> bytes:
    """54バイトの InnoDataPacket ヘッダーを生成する。"""
    buf = bytearray(HEADER_SIZE)
    # magic
    buf[0:2] = PACKET_MAGIC
    # version
    buf[2] = 1
    # pkt_size at offset 8
    if pkt_size is None:
        pkt_size = HEADER_SIZE + n_blocks * BLOCK1_SIZE
    struct.pack_into('<I', buf, OFFSET_PKT_SIZE, pkt_size)
    # frame_idx at offset 26
    struct.pack_into('<Q', buf, OFFSET_FRAME_IDX, frame_idx)
    # type_num at offset 38: type in bits 0-7, item_number (=n_blocks) in bits 8-31
    type_num = (pkt_type & 0xFF) | ((n_blocks & 0xFFFFFF) << 8)
    struct.pack_into('<I', buf, OFFSET_TYPE_NUM, type_num)
    # item_size at offset 42
    struct.pack_into('<H', buf, OFFSET_ITEM_SIZE, item_size)
    return bytes(buf)


def _make_block1(
    h_angle: int = 0,
    v_angle: int = 0,
    radii:   tuple = (200, 0, 0, 0),   # InnoDistanceUnit; 200 = 1 m
    refls:   tuple = (100, 0, 0, 0),
    h_diffs: tuple = (0, 0, 0),         # (h_diff1, h_diff2, h_diff3)
    v_diffs: tuple = (0, 0, 0),         # (v_diff1, v_diff2, v_diff3)
) -> bytes:
    """33バイトの InnoBlock1 を生成する。"""
    buf = bytearray(BLOCK1_SIZE)

    # h_angle, v_angle (int16 LE)
    struct.pack_into('<h', buf, 0, h_angle)
    struct.pack_into('<h', buf, 2, v_angle)

    # ビットフィールド (bytes 8-16, 72ビット LE)
    h_diff1, h_diff2, h_diff3 = h_diffs
    v_diff1, v_diff2, v_diff3 = v_diffs

    bf = 0
    bf |= (h_diff1 & 0x1FF) << 9
    bf |= (h_diff2 & 0x3FF) << 18
    bf |= (h_diff3 & 0x7FF) << 28
    bf |= (v_diff1 & 0xFF)  << 39
    bf |= (v_diff2 & 0x1FF) << 47
    bf |= (v_diff3 & 0x1FF) << 56

    bf_bytes = (bf & ((1 << 72) - 1)).to_bytes(9, byteorder='little')
    buf[8:17] = bf_bytes

    # InnoChannelPoint: radius (bits 0-16) + refl (bits 17-24)
    for ch in range(CHANNELS_PER_BLOCK):
        offset = BLOCK_HDR_SIZE + ch * CHANNEL_PT_SIZE
        cp_val = (radii[ch] & 0x1FFFF) | ((refls[ch] & 0xFF) << 17)
        struct.pack_into('<I', buf, offset, cp_val)

    return bytes(buf)


def _make_packet(
    frame_idx: int = 0,
    blocks: list = None,
    pkt_size: int = None,
) -> bytes:
    """ヘッダー + ブロック列からパケットを組み立てる。"""
    if blocks is None:
        blocks = [_make_block1()]
    body = b"".join(b if isinstance(b, bytes) else _make_block1(*b) for b in blocks)
    n_blocks = len(body) // BLOCK1_SIZE
    hdr = _make_header(frame_idx=frame_idx, n_blocks=n_blocks, pkt_size=pkt_size)
    return hdr + body


# ── 定数テスト ────────────────────────────────────────────────────────────────

class TestConstants:

    def test_magic_bytes(self):
        """マジック = 0x6A 0x17"""
        assert PACKET_MAGIC == bytes([0x6A, 0x17])

    def test_header_size(self):
        """ヘッダーサイズ = 54 bytes"""
        assert HEADER_SIZE == 54

    def test_block1_size(self):
        """InnoBlock1 = 33 bytes"""
        assert BLOCK1_SIZE == 33

    def test_block_hdr_size(self):
        """ブロックヘッダー = 17 bytes"""
        assert BLOCK_HDR_SIZE == 17

    def test_meters_per_unit(self):
        """1 InnoDistanceUnit = 0.005 m"""
        assert abs(METERS_PER_UNIT - 0.005) < 1e-10

    def test_rad_per_unit(self):
        """1 InnoAngleUnit = π/32768 rad"""
        assert abs(RAD_PER_UNIT - math.pi / 32768) < 1e-15

    def test_v_angle_base(self):
        """kInnoFaconVAngleDiffBase = 196"""
        assert V_ANGLE_BASE_PER_CH == 196

    def test_blocks_per_full_pkt(self):
        """フルパケット = 42 ブロック"""
        assert BLOCKS_PER_FULL_PKT == 42

    def test_full_packet_size(self):
        """54 + 42×33 = 1440 bytes"""
        assert HEADER_SIZE + BLOCKS_PER_FULL_PKT * BLOCK1_SIZE == 1440


# ── ヘッダー検証テスト ────────────────────────────────────────────────────────

class TestHeaderValidation:

    def test_magic_confirmed(self):
        """マジック 0x6A17 を受け付ける"""
        parser = FalconK2Parser()
        pkt = _make_packet(frame_idx=1, blocks=[_make_block1(radii=(200, 0, 0, 0))])
        result = parser.feed(pkt)
        # 最初のパケットでフレームは確定しない
        assert result is None

    def test_wrong_magic_rejected(self):
        """不正マジックのパケットを拒否する"""
        parser = FalconK2Parser()
        pkt = bytearray(_make_packet(frame_idx=0))
        pkt[0] = 0xAB
        pkt[1] = 0xCD
        assert parser.feed(bytes(pkt)) is None

    def test_size_mismatch_rejected(self):
        """ヘッダーの size と実サイズの不一致を弾く"""
        parser = FalconK2Parser()
        pkt = bytearray(_make_packet(frame_idx=0))
        # pkt_size フィールドを故意にずらす
        struct.pack_into('<I', pkt, OFFSET_PKT_SIZE, 9999)
        assert parser.feed(bytes(pkt)) is None

    def test_short_packet_rejected(self):
        """ヘッダー未満の短いパケットを拒否する"""
        parser = FalconK2Parser()
        assert parser.feed(bytes([0x6A, 0x17, 0x01, 0x00])) is None

    def test_empty_bytes_rejected(self):
        """空バイト列を拒否する"""
        parser = FalconK2Parser()
        assert parser.feed(b"") is None

    def test_wrong_type_rejected(self):
        """type != 1 (SPHERE_POINTCLOUD) のパケットを拒否する"""
        parser = FalconK2Parser()
        hdr = _make_header(frame_idx=0, n_blocks=1, pkt_type=2, item_size=BLOCK1_SIZE)
        pkt = hdr + _make_block1()
        assert parser.feed(pkt) is None

    def test_wrong_item_size_rejected(self):
        """item_size != 33 (InnoBlock1 以外) のパケットを拒否する"""
        parser = FalconK2Parser()
        hdr = _make_header(frame_idx=0, n_blocks=1, item_size=49)  # InnoBlock2
        pkt = hdr + _make_block1()
        assert parser.feed(pkt) is None


# ── フレーム境界テスト ────────────────────────────────────────────────────────

class TestFrameBoundary:

    def test_first_feed_returns_none(self):
        """最初のパケットではフレームを返さない"""
        parser = FalconK2Parser()
        pkt = _make_packet(frame_idx=0, blocks=[_make_block1(radii=(200, 0, 0, 0))])
        assert parser.feed(pkt) is None

    def test_frame_boundary_via_frame_idx_change(self):
        """frame_idx の変化で前フレームが確定する"""
        parser = FalconK2Parser()
        # frame_idx=0 で 2 パケット
        parser.feed(_make_packet(frame_idx=0, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        parser.feed(_make_packet(frame_idx=0, blocks=[_make_block1(radii=(400, 0, 0, 0))]))
        # frame_idx=1 が来たら frame_idx=0 のフレームが確定
        frame = parser.feed(_make_packet(frame_idx=1, blocks=[_make_block1(radii=(600, 0, 0, 0))]))
        assert frame is not None
        assert isinstance(frame, PointCloudFrame)

    def test_same_frame_idx_does_not_emit(self):
        """同じ frame_idx ではフレームを返さない"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(frame_idx=5, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        result = parser.feed(_make_packet(frame_idx=5, blocks=[_make_block1(radii=(400, 0, 0, 0))]))
        assert result is None

    def test_frame_has_points_from_all_packets(self):
        """フレームには同じ frame_idx を持つ全パケットの点が含まれる"""
        parser = FalconK2Parser()
        N = 3
        for i in range(N):
            parser.feed(_make_packet(
                frame_idx=0,
                blocks=[_make_block1(radii=(200 + i * 100, 0, 0, 0))],
            ))
        # 新フレームで前フレームを確定
        frame = parser.feed(_make_packet(
            frame_idx=1,
            blocks=[_make_block1(radii=(200, 0, 0, 0))],
        ))
        assert frame is not None
        assert frame.points.shape[0] >= N

    def test_frame_counter_increments(self):
        """フレームが確定するたびに frame_id が増える"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(frame_idx=0, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        f0 = parser.feed(_make_packet(frame_idx=1, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        f1 = parser.feed(_make_packet(frame_idx=2, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        assert f0 is not None
        assert f1 is not None
        assert f1.frame_id > f0.frame_id

    def test_reset_clears_state(self):
        """reset() 後は新規スタートとして動作する"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(frame_idx=0, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        parser.reset()
        # リセット後の最初のパケットは None
        result = parser.feed(_make_packet(frame_idx=1, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        assert result is None


# ── ブロック解析・XYZ 変換テスト ──────────────────────────────────────────────

class TestBlockParsing:

    def test_radius_zero_excluded(self):
        """radius = 0 のチャンネル点は除外される"""
        # CH0=0 (無効), CH1=200 (有効=1m)
        block = _make_block1(radii=(0, 200, 0, 0))
        pts = _parse_blocks(block, 1)
        # CH0 は除外、CH1 から 1 点
        assert pts.shape[0] == 1

    def test_all_zero_radii_gives_empty(self):
        """全チャンネル radius=0 → 空配列"""
        block = _make_block1(radii=(0, 0, 0, 0))
        pts = _parse_blocks(block, 1)
        assert pts.shape == (0, 4)

    def test_all_channels_valid(self):
        """4 チャンネルが有効なら 4 点出力"""
        block = _make_block1(radii=(200, 200, 200, 200), refls=(10, 20, 30, 40))
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 4

    def test_forward_point(self):
        """h=0, v=0 → (x=0, y=0, z=r) 真正面"""
        r_unit = 200   # 200/200 = 1.0 m
        block = _make_block1(h_angle=0, v_angle=0, radii=(r_unit, 0, 0, 0))
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 1
        x, y, z, _ = pts[0]
        assert abs(x) < 1e-5, f"x={x} (期待 0)"
        assert abs(y) < 1e-5, f"y={y} (期待 0)"
        assert abs(z - 1.0) < 1e-5, f"z={z} (期待 1.0)"

    def test_right_point(self):
        """h=π/2, v=0 → (x=0, y=r, z≈0) 真右"""
        r_unit = 200
        # 16384 units = 16384 × π/32768 = π/2
        h_angle_unit = 16384
        block = _make_block1(h_angle=h_angle_unit, v_angle=0, radii=(r_unit, 0, 0, 0))
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 1
        x, y, z, _ = pts[0]
        assert abs(x) < 1e-4, f"x={x} (期待 0)"
        assert abs(y - 1.0) < 1e-4, f"y={y} (期待 1.0)"
        assert abs(z) < 1e-4, f"z={z} (期待 0)"

    def test_up_point(self):
        """h=0, v=π/2 → (x=r, y=0, z≈0) 真上"""
        r_unit = 200
        v_angle_unit = 16384   # π/2
        block = _make_block1(h_angle=0, v_angle=v_angle_unit, radii=(r_unit, 0, 0, 0))
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 1
        x, y, z, _ = pts[0]
        assert abs(x - 1.0) < 1e-4, f"x={x} (期待 1.0)"
        assert abs(y) < 1e-4, f"y={y} (期待 0)"
        assert abs(z) < 1e-4, f"z={z} (期待 0)"

    def test_distance_magnitude(self):
        """sqrt(x²+y²+z²) が radius/200 に等しい"""
        r_unit = 4000   # 20 m
        block = _make_block1(h_angle=1000, v_angle=500, radii=(r_unit, 0, 0, 0))
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 1
        x, y, z, _ = pts[0]
        r_calc = math.sqrt(float(x)**2 + float(y)**2 + float(z)**2)
        r_expected = r_unit * METERS_PER_UNIT
        assert abs(r_calc - r_expected) < 1e-4, (
            f"距離不一致: calc={r_calc:.5f} expected={r_expected:.5f}"
        )

    def test_reflectance_preserved(self):
        """反射強度 (refl) が points[:, 3] に正しく格納される"""
        block = _make_block1(
            radii=(200, 200, 200, 200),
            refls=(10, 50, 100, 200),
        )
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 4
        refl_values = set(pts[:, 3].astype(int))
        assert refl_values == {10, 50, 100, 200}, f"refl={refl_values}"

    def test_multiple_blocks(self):
        """複数ブロックを正しく集約する"""
        n = 5
        blocks = b"".join(
            _make_block1(radii=(200, 200, 0, 0)) for _ in range(n)
        )
        pts = _parse_blocks(blocks, n)
        # 各ブロックから 2 点 (CH0+CH1) → 合計 n×2
        assert pts.shape[0] == n * 2

    def test_output_dtype_float32(self):
        """出力配列は float32"""
        block = _make_block1(radii=(200, 0, 0, 0))
        pts = _parse_blocks(block, 1)
        assert pts.dtype == np.float32

    def test_output_shape_cols(self):
        """出力配列は (N, 4) 形状 (x, y, z, refl)"""
        block = _make_block1(radii=(200, 200, 200, 200))
        pts = _parse_blocks(block, 1)
        assert pts.ndim == 2
        assert pts.shape[1] == 4


# ── ビットフィールド (チャンネル差分) テスト ──────────────────────────────────

class TestBitfieldDiffs:

    def test_h_diff1_applied(self):
        """h_diff1 が CH1 の水平角に加算される
        9ビット符号付きの有効範囲: -256 ～ +255"""
        h_base  = 0
        h_diff1 = 100    # +100 InnoAngleUnit (9ビット符号付き範囲内)

        block = _make_block1(
            h_angle=h_base,
            v_angle=0,
            radii=(200, 200, 0, 0),
            h_diffs=(h_diff1, 0, 0),
            v_diffs=(0, 0, 0),
        )
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 2

        # CH0: h=0 → y=0
        # CH1: h=100 → y = r·cos(v_ch1)·sin(h_ch1) > 0
        ch0_y = pts[0, 1]
        ch1_y = pts[1, 1]
        assert ch1_y > ch0_y, f"CH1.y={ch1_y:.4f} should > CH0.y={ch0_y:.4f}"

    def test_negative_h_diff1(self):
        """負の h_diff1 が CH1 の水平角を減らす"""
        block = _make_block1(
            h_angle=1000,
            v_angle=0,
            radii=(200, 200, 0, 0),
            h_diffs=(-200, 0, 0),
            v_diffs=(0, 0, 0),
        )
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 2
        # CH1 の y は CH0 より小さい (h が小さいので)
        assert pts[1, 1] < pts[0, 1]

    def test_v_angle_base_per_channel(self):
        """v_diff=0 でも CH1-CH3 の仰角に 196, 392, 588 ユニットが加算される"""
        block = _make_block1(
            h_angle=0,
            v_angle=0,
            radii=(200, 200, 200, 200),
            h_diffs=(0, 0, 0),
            v_diffs=(0, 0, 0),
        )
        pts = _parse_blocks(block, 1)
        assert pts.shape[0] == 4

        # v_angle_ch[i] = 196*i → x = r * sin(v_rad)
        # CH0: v=0    → x ≈ 0
        # CH1: v=196  → x = r * sin(196 × π/32768) > 0
        # CH2: v=392  → x > CH1
        # CH3: v=588  → x > CH2
        x_vals = pts[:, 0]
        assert x_vals[0] < x_vals[1] < x_vals[2] < x_vals[3], (
            f"仰角昇順期待: {x_vals}"
        )


# ── フルパケット統合テスト ────────────────────────────────────────────────────

class TestFullPacketIntegration:

    def test_full_packet_42_blocks(self):
        """42ブロック × CH0 有効 → 42点以上"""
        blocks = b"".join(
            _make_block1(
                h_angle=i * 10,
                v_angle=0,
                radii=(200, 0, 0, 0),
            )
            for i in range(BLOCKS_PER_FULL_PKT)
        )
        pts = _parse_blocks(blocks, BLOCKS_PER_FULL_PKT)
        assert pts.shape[0] == BLOCKS_PER_FULL_PKT

    def test_full_packet_all_channels(self):
        """42ブロック × 4ch 全有効 → 168点"""
        blocks = b"".join(
            _make_block1(radii=(200, 200, 200, 200))
            for _ in range(BLOCKS_PER_FULL_PKT)
        )
        pts = _parse_blocks(blocks, BLOCKS_PER_FULL_PKT)
        assert pts.shape[0] == BLOCKS_PER_FULL_PKT * CHANNELS_PER_BLOCK

    def test_packet_via_parser_feed(self):
        """parser.feed() でフレームに点が蓄積される"""
        parser = FalconK2Parser()
        # frame_idx=0 で 3 パケット
        for i in range(3):
            parser.feed(_make_packet(
                frame_idx=0,
                blocks=[_make_block1(radii=(200 + i * 50, 0, 0, 0))],
            ))
        # frame_idx=1 でフレーム確定
        frame = parser.feed(_make_packet(
            frame_idx=1,
            blocks=[_make_block1(radii=(200, 0, 0, 0))],
        ))
        assert frame is not None
        assert frame.points.shape[0] >= 3
        assert frame.points.dtype == np.float32

    def test_all_zero_radii_frame_is_none(self):
        """全 radius=0 のパケットのみのフレームは None を返す"""
        parser = FalconK2Parser()
        # frame_idx=0 で全点無効のパケット
        parser.feed(_make_packet(
            frame_idx=0,
            blocks=[_make_block1(radii=(0, 0, 0, 0))],
        ))
        # frame_idx=1 で前フレームを emit 試みるが chunk が空 → None
        frame = parser.feed(_make_packet(
            frame_idx=1,
            blocks=[_make_block1(radii=(200, 0, 0, 0))],
        ))
        assert frame is None

    def test_finite_xyz(self):
        """有効パケットから非 NaN/非 Inf の XYZ が生成される"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(
            frame_idx=0,
            blocks=[_make_block1(
                h_angle=1000, v_angle=200,
                radii=(2000, 1500, 1000, 500),
            )],
        ))
        frame = parser.feed(_make_packet(
            frame_idx=1,
            blocks=[_make_block1(radii=(200, 0, 0, 0))],
        ))
        assert frame is not None
        assert frame.points.shape[0] == 4
        assert np.isfinite(frame.points[:, :3]).all(), (
            f"非有限値あり: {frame.points}"
        )

    def test_timestamp_is_float(self):
        """PointCloudFrame.timestamp は float"""
        parser = FalconK2Parser()
        parser.feed(_make_packet(frame_idx=0, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        frame = parser.feed(_make_packet(frame_idx=1, blocks=[_make_block1(radii=(200, 0, 0, 0))]))
        assert frame is not None
        assert isinstance(frame.timestamp, float)
