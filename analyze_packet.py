#!/usr/bin/env python3
"""
Falcon K2 生パケット (.bin) の詳細解析ツール
==============================================

各バイト位置の統計と、複数のフィールド配置仮説のスコアリングを行い、
正しいフォーマットを特定する。

使い方:
    python analyze_packet.py path/to/packet_001_1440b.bin
    python analyze_packet.py path/to/packet_dump_xxx/
    python analyze_packet.py path/to/packet_dump_xxx/ --bytes  # バイト位置別解析モード
    python analyze_packet.py path/to/packet_dump_xxx/ --hexdump # 生バイトを HEX で表示
"""

import sys
import os
import glob
import struct
import argparse
import numpy as np


PACKET_MAGIC = bytes([0x6A, 0x17])
HEADER_SIZE  = 54
POINT_SIZE   = 9


def _load_bin(filepath: str) -> bytes:
    with open(filepath, "rb") as f:
        return f.read()


def _parse_packet(data: bytes) -> dict:
    """9バイト/点を XYZ (mm) として読み込んで返す"""
    if len(data) < HEADER_SIZE or data[:2] != PACKET_MAGIC:
        return None

    payload = data[HEADER_SIZE:]
    n = len(payload) // POINT_SIZE
    if n == 0:
        return None
    buf = np.frombuffer(payload[:n * POINT_SIZE], dtype=np.uint8).reshape(n, POINT_SIZE)

    x_mm = (buf[:, 0].astype(np.uint32) | (buf[:, 1].astype(np.uint32) << 8))
    y_mm = (buf[:, 2].astype(np.uint32) | (buf[:, 3].astype(np.uint32) << 8)).astype(np.int32)
    y_mm[y_mm >= 0x8000] -= 0x10000
    z_mm = (buf[:, 4].astype(np.uint32) | (buf[:, 5].astype(np.uint32) << 8)).astype(np.int32)
    z_mm[z_mm >= 0x8000] -= 0x10000
    intens = buf[:, 6]

    return {
        "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm,
        "intensity": intens, "n": n,
    }


def _hist_line(values: np.ndarray, bins: list, label: str = "{:>7.1f}") -> None:
    total = len(values)
    for lo, hi in bins:
        cnt = ((values >= lo) & (values < hi)).sum()
        pct = cnt / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {label.format(lo)} - {label.format(hi)}: {cnt:>7,} ({pct:5.1f}%) {bar}")


def analyze_aggregate(filepaths: list) -> None:
    parsed = []
    for fp in filepaths:
        try:
            data = _load_bin(fp)
            r = _parse_packet(data)
            if r is not None:
                parsed.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ {os.path.basename(fp)}: {e}")

    if not parsed:
        print("  ⚠ パース可能なファイルがありません")
        return

    x_mm = np.concatenate([p["x_mm"] for p in parsed])
    y_mm = np.concatenate([p["y_mm"] for p in parsed])
    z_mm = np.concatenate([p["z_mm"] for p in parsed])
    intens = np.concatenate([p["intensity"] for p in parsed])
    n_total = len(x_mm)

    # ── 無効点判定 ──
    inv_x_zero = x_mm == 0
    inv_x_max  = x_mm == 0xFFFF
    inv_y_pos  = y_mm == 32767
    inv_y_neg  = y_mm == -32768
    inv_z_pos  = z_mm == 32767
    inv_z_neg  = z_mm == -32768
    invalid = inv_x_zero | inv_x_max | inv_y_pos | inv_y_neg | inv_z_pos | inv_z_neg
    valid = ~invalid
    n_valid = valid.sum()
    n_invalid = invalid.sum()

    print(f"\n{'='*72}")
    print(f" 集約統計: {len(parsed)} ファイル, 総点数 {n_total:,}")
    print(f"{'='*72}")

    # ── 無効点ブレイクダウン ──
    print(f"\n■ 無効点ブレイクダウン")
    print(f"    X == 0 (戻り信号なし)        : {inv_x_zero.sum():>8,} "
          f"({inv_x_zero.sum()/n_total*100:5.1f}%)")
    print(f"    X == 0xFFFF (距離オーバー)    : {inv_x_max.sum():>8,} "
          f"({inv_x_max.sum()/n_total*100:5.1f}%)")
    print(f"    Y == +32767 (右側飽和)        : {inv_y_pos.sum():>8,} "
          f"({inv_y_pos.sum()/n_total*100:5.1f}%)")
    print(f"    Y == -32768 (左側飽和)        : {inv_y_neg.sum():>8,} "
          f"({inv_y_neg.sum()/n_total*100:5.1f}%)")
    print(f"    Z == +32767 (上側飽和)        : {inv_z_pos.sum():>8,} "
          f"({inv_z_pos.sum()/n_total*100:5.1f}%)")
    print(f"    Z == -32768 (下側飽和)        : {inv_z_neg.sum():>8,} "
          f"({inv_z_neg.sum()/n_total*100:5.1f}%)")
    print(f"    ─────────────────────────────────")
    print(f"    無効点 合計                   : {n_invalid:>8,} "
          f"({n_invalid/n_total*100:5.1f}%)")
    print(f"    有効点                       : {n_valid:>8,} "
          f"({n_valid/n_total*100:5.1f}%)")

    if n_valid == 0:
        print("\n  ⚠ 有効点がありません")
        return

    # ── 有効点だけの統計 ──
    x_v = x_mm[valid]
    y_v = y_mm[valid]
    z_v = z_mm[valid]
    int_v = intens[valid]

    print(f"\n■ 有効点 X(前方距離) 分布")
    print(f"    範囲: {x_v.min()} 〜 {x_v.max()} mm "
          f"({x_v.min()/1000:.3f} 〜 {x_v.max()/1000:.3f} m)")
    print(f"    平均: {x_v.mean()/1000:.2f} m, 中央値: {np.median(x_v)/1000:.2f} m")
    _hist_line(x_v, [
        (1, 500), (500, 1500), (1500, 3000), (3000, 6000),
        (6000, 12000), (12000, 25000), (25000, 45000), (45000, 65535),
    ], label="{:>5.0f}mm")

    print(f"\n■ 有効点 Y(横方向) 分布")
    print(f"    範囲: {y_v.min()} 〜 {y_v.max()} mm "
          f"({y_v.min()/1000:.3f} 〜 {y_v.max()/1000:.3f} m)")
    print(f"    平均: {y_v.mean()/1000:.2f} m")
    _hist_line(y_v, [
        (-30000, -10000), (-10000, -3000), (-3000, -500),
        (-500, 500), (500, 3000), (3000, 10000), (10000, 30000),
    ], label="{:>6.0f}mm")

    print(f"\n■ 有効点 Z(高さ) 分布")
    print(f"    範囲: {z_v.min()} 〜 {z_v.max()} mm "
          f"({z_v.min()/1000:.3f} 〜 {z_v.max()/1000:.3f} m)")
    print(f"    平均: {z_v.mean()/1000:.2f} m")
    _hist_line(z_v, [
        (-30000, -10000), (-10000, -3000), (-3000, -500),
        (-500, 500), (500, 3000), (3000, 10000), (10000, 30000),
    ], label="{:>6.0f}mm")

    # ── FOV 推定 (forward LiDAR ?) ──
    # tan(θ_H) = |Y|/X, tan(θ_V) = |Z|/X
    valid_for_fov = x_v > 100  # X が十分大きい点
    if valid_for_fov.sum() > 0:
        x_f = x_v[valid_for_fov].astype(np.float64)
        y_f = y_v[valid_for_fov].astype(np.float64)
        z_f = z_v[valid_for_fov].astype(np.float64)
        tan_h = np.abs(y_f) / x_f
        tan_v = np.abs(z_f) / x_f
        # tan の 99 パーセンタイル (異常値除外)
        tan_h_p99 = np.percentile(tan_h, 99)
        tan_v_p99 = np.percentile(tan_v, 99)
        fov_h_p99 = np.degrees(np.arctan(tan_h_p99)) * 2  # 全角FOV
        fov_v_p99 = np.degrees(np.arctan(tan_v_p99)) * 2

        print(f"\n■ FOV 推定 (X>100mm の点に対し |Y|/X, |Z|/X の99%タイルから)")
        print(f"    水平FOV (推定): ±{fov_h_p99/2:.1f}°  (全角 {fov_h_p99:.1f}°)")
        print(f"    垂直FOV (推定): ±{fov_v_p99/2:.1f}°  (全角 {fov_v_p99:.1f}°)")

    print(f"\n■ Intensity")
    print(f"    範囲: {int_v.min()} 〜 {int_v.max()}, 平均: {int_v.mean():.1f}")

    # ── 診断 ──
    print(f"\n{'='*72}")
    print(f" ◆ 診断結果")
    print(f"{'='*72}")

    sat_rate = (inv_y_pos.sum() + inv_y_neg.sum() + inv_z_pos.sum() + inv_z_neg.sum()) / n_total * 100
    over_rate = inv_x_max.sum() / n_total * 100
    nohit_rate = inv_x_zero.sum() / n_total * 100

    if sat_rate > 5:
        print(f"  ⚠ Y/Z 飽和率が {sat_rate:.1f}% と高いです")
        print(f"    → 無効測定マーカーが多数含まれている (フィルター推奨)")
    if over_rate > 5:
        print(f"  ⚠ X=0xFFFF (最大距離オーバー) が {over_rate:.1f}%")
    if nohit_rate > 30:
        print(f"  ⚠ X=0 (戻り信号なし) が {nohit_rate:.1f}% と高いです")
        print(f"    → LiDAR が空(視野外)を多く向いている可能性")

    if valid_for_fov.sum() > 0:
        if fov_h_p99 < 200:
            print(f"  ✓ 水平 FOV {fov_h_p99:.0f}° は前方視 LiDAR として妥当")
        else:
            print(f"  ⚠ 水平 FOV {fov_h_p99:.0f}° が広すぎ (>180°)")
            print(f"    → XYZ の解釈が違う可能性")

        if fov_v_p99 < 60:
            print(f"  ✓ 垂直 FOV {fov_v_p99:.0f}° は前方視 LiDAR として妥当")
        else:
            print(f"  ⚠ 垂直 FOV {fov_v_p99:.0f}° が大きい")
            print(f"    → XYZ の解釈が違う可能性")


def _extract_all_point_bytes(filepaths: list) -> np.ndarray:
    """全パケットから全点(9バイト)を連結した (N, 9) uint8 配列を返す"""
    chunks = []
    for fp in filepaths:
        try:
            with open(fp, "rb") as f:
                data = f.read()
            if len(data) < HEADER_SIZE or data[:2] != PACKET_MAGIC:
                continue
            payload = data[HEADER_SIZE:]
            n = len(payload) // POINT_SIZE
            if n == 0:
                continue
            buf = np.frombuffer(payload[:n * POINT_SIZE], dtype=np.uint8).reshape(n, POINT_SIZE)
            chunks.append(buf)
        except Exception:  # noqa: BLE001
            pass
    if not chunks:
        return np.empty((0, POINT_SIZE), dtype=np.uint8)
    return np.concatenate(chunks, axis=0)


def analyze_bytes(filepaths: list) -> None:
    """
    バイト位置ごとに統計を表示する詳細解析。
    どのフィールドが range / intensity / azimuth / channel かを
    特徴(値域・ユニーク数・連続性)から推定。
    """
    pts = _extract_all_point_bytes(filepaths)
    if pts.shape[0] == 0:
        print("  ⚠ 解析可能な点がありません")
        return

    n = pts.shape[0]
    print(f"\n{'='*72}")
    print(f" バイト位置別解析: {n:,} 点 ({len(filepaths)} ファイル)")
    print(f"{'='*72}")

    # ── 各バイト位置の基本統計 ──
    print(f"\n■ 各バイト位置の値分布 (0-255 を 8 区間に集計)")
    print(f"        ┌─{'─'*32}─┬──────┬──────┬──────┬───────┐")
    print(f"        │ {'分布(8段階)':^32}│ Min  │ Max  │ Mean │ Unique│")
    print(f"        ├─{'─'*32}─┼──────┼──────┼──────┼───────┤")
    for col in range(POINT_SIZE):
        v = pts[:, col]
        # 8区間ヒストグラム
        bars = ""
        for low in range(0, 256, 32):
            cnt = ((v >= low) & (v < low + 32)).sum()
            pct = cnt / n * 100
            ch = "█" if pct > 25 else "▓" if pct > 12 else "▒" if pct > 5 else "·"
            bars += ch * 4
        unique = len(np.unique(v))
        print(f"  byte{col} │ {bars} │ {v.min():>4} │ {v.max():>4} │ {v.mean():>4.0f} │ {unique:>5} │")
    print(f"        └─{'─'*32}─┴──────┴──────┴──────┴───────┘")

    # ── 2バイト組み合わせ (LE) の解釈 ──
    print(f"\n■ 連続2バイトを uint16 LE として読んだときの値域")
    for off in range(POINT_SIZE - 1):
        u16 = pts[:, off].astype(np.uint32) | (pts[:, off + 1].astype(np.uint32) << 8)
        i16 = u16.astype(np.int32)
        i16[i16 >= 0x8000] -= 0x10000
        print(f"  bytes {off}-{off+1}: u16=[{u16.min():>5}, {u16.max():>5}] (avg {u16.mean():>6.0f}) "
              f"i16=[{i16.min():>+6}, {i16.max():>+6}]  ユニーク値 {len(np.unique(u16))}")

    # ── 隣接フレームで「定数」のバイト位置 ──
    print(f"\n■ パケット内最初の5点で各バイト位置がどう変化するか")
    print(f"     (定数列はメタ/フラグの可能性、連続変化は座標の可能性)")
    print(f"        Pt0   Pt1   Pt2   Pt3   Pt4")
    for col in range(POINT_SIZE):
        first5 = pts[:5, col]
        vals = "  ".join(f"0x{v:02x}" for v in first5)
        # 連続性スコア
        diffs = np.diff(first5.astype(np.int32))
        if len(set(first5)) == 1:
            tag = "← 一定値"
        elif np.all(diffs > 0):
            tag = "← 単調増加 (azimuth候補)"
        elif np.all(diffs < 0):
            tag = "← 単調減少"
        else:
            tag = ""
        print(f"  byte{col}: {vals}    {tag}")

    # ── 最初の10点と最後の10点を HEX 表示 ──
    print(f"\n■ 最初の 10 点 (生 hex)")
    for i in range(min(10, n)):
        hex_str = " ".join(f"{b:02x}" for b in pts[i])
        # 解釈情報も併記
        x_mm = int(pts[i, 0]) | (int(pts[i, 1]) << 8)
        y_raw = int(pts[i, 2]) | (int(pts[i, 3]) << 8)
        z_raw = int(pts[i, 4]) | (int(pts[i, 5]) << 8)
        y_i = y_raw - 0x10000 if y_raw >= 0x8000 else y_raw
        z_i = z_raw - 0x10000 if z_raw >= 0x8000 else z_raw
        print(f"  [{i:3d}] {hex_str}  | u16(0-1)={x_mm:>5} i16(2-3)={y_i:>+6} i16(4-5)={z_i:>+6}")

    if n > 20:
        print(f"\n■ 中央付近の 5 点 ({n // 2} 〜 {n // 2 + 4})")
        for i in range(n // 2, min(n // 2 + 5, n)):
            hex_str = " ".join(f"{b:02x}" for b in pts[i])
            x_mm = int(pts[i, 0]) | (int(pts[i, 1]) << 8)
            print(f"  [{i:3d}] {hex_str}  | u16(0-1)={x_mm:>5}")


def score_layouts(filepaths: list) -> None:
    """
    複数のフィールド配置仮説を試して、どれが最も妥当な分布を生むかスコアで比較。
    """
    pts = _extract_all_point_bytes(filepaths)
    if pts.shape[0] == 0:
        print("  ⚠ 点なし")
        return

    n = pts.shape[0]
    print(f"\n{'='*72}")
    print(f" フィールド配置候補のスコア比較 ({n:,} 点)")
    print(f"{'='*72}")

    layouts = [
        ("A", "0-1:X u16 mm / 2-3:Y i16 mm / 4-5:Z i16 mm (現実装)",
         lambda p: _interpret_xyz_direct(p)),
        ("B", "0-1:range u16 mm / 2-3:az i16 0.01° / 4-5:el i16 0.01° (球面)",
         lambda p: _interpret_spherical(p, 0.01)),
        ("C", "0-1:range u16 mm / 2-3:intensity u16 / 4-5:channel u16 / 6-7:azimuth u16",
         lambda p: _interpret_range_intensity_channel_azimuth(p)),
        ("D", "0-1:range u16 cm / 2-3:Y i16 cm / 4-5:Z i16 cm",
         lambda p: _interpret_xyz_direct(p, scale=0.01)),  # cm
    ]

    print(f"\n  各仕様で 99% タイル の |Y|/X, |Z|/X から FOV を推定")
    print(f"  → 60-180° なら妥当な前方視 LiDAR, それ以外は不正解")
    print()
    print(f"  仕様 | 説明")
    print(f"  ─────┼──────────────────────────────────────────────────────────")

    for tag, desc, fn in layouts:
        try:
            xyz, info = fn(pts)
            # 妥当性スコア計算
            valid = (
                np.isfinite(xyz).all(axis=1)
                & (xyz[:, 0] > 0.1)   # X > 10cm
                & (np.abs(xyz[:, 0]) < 200)  # X < 200m
                & (np.abs(xyz[:, 1]) < 200)
                & (np.abs(xyz[:, 2]) < 200)
            )
            if valid.sum() < 10:
                fov_h_str = "N/A"
                fov_v_str = "N/A"
                score = "?"
            else:
                v = xyz[valid]
                tan_h = np.abs(v[:, 1]) / v[:, 0]
                tan_v = np.abs(v[:, 2]) / v[:, 0]
                fov_h = np.degrees(np.arctan(np.percentile(tan_h, 99))) * 2
                fov_v = np.degrees(np.arctan(np.percentile(tan_v, 99))) * 2
                fov_h_str = f"{fov_h:5.1f}°"
                fov_v_str = f"{fov_v:5.1f}°"
                # スコア: 水平60-180度・垂直10-60度なら良し
                if 30 < fov_h < 180 and 5 < fov_v < 60:
                    score = "★★★ 妥当"
                elif 20 < fov_h < 220 and 5 < fov_v < 120:
                    score = "★★  まあまあ"
                elif fov_h > 170 or fov_v > 170:
                    score = "✗   全球状(不正解)"
                else:
                    score = "?   要確認"
            print(f"   {tag}   | {desc}")
            print(f"       └─ 有効点: {valid.sum():>6,}/{n:,}  FOV水平={fov_h_str}  FOV垂直={fov_v_str}  → {score}")
            print()
        except Exception as e:  # noqa: BLE001
            print(f"   {tag}   | {desc}")
            print(f"       └─ エラー: {e}")


def _interpret_xyz_direct(pts: np.ndarray, scale: float = 0.001) -> tuple:
    """bytes 0-1:X u16, 2-3:Y i16, 4-5:Z i16 (mm→m はscale)"""
    x = (pts[:, 0].astype(np.uint32) | (pts[:, 1].astype(np.uint32) << 8)).astype(np.float64) * scale
    y_raw = (pts[:, 2].astype(np.uint32) | (pts[:, 3].astype(np.uint32) << 8)).astype(np.int32)
    y_raw[y_raw >= 0x8000] -= 0x10000
    z_raw = (pts[:, 4].astype(np.uint32) | (pts[:, 5].astype(np.uint32) << 8)).astype(np.int32)
    z_raw[z_raw >= 0x8000] -= 0x10000
    y = y_raw.astype(np.float64) * scale
    z = z_raw.astype(np.float64) * scale
    return np.stack([x, y, z], axis=1), {}


def _interpret_spherical(pts: np.ndarray, angle_unit_deg: float = 0.01) -> tuple:
    """bytes 0-1:range u16 mm, 2-3:az i16, 4-5:el i16, 単位は angle_unit_deg [度/LSB]"""
    r = (pts[:, 0].astype(np.uint32) | (pts[:, 1].astype(np.uint32) << 8)).astype(np.float64) / 1000.0
    az_raw = (pts[:, 2].astype(np.uint32) | (pts[:, 3].astype(np.uint32) << 8)).astype(np.int32)
    az_raw[az_raw >= 0x8000] -= 0x10000
    el_raw = (pts[:, 4].astype(np.uint32) | (pts[:, 5].astype(np.uint32) << 8)).astype(np.int32)
    el_raw[el_raw >= 0x8000] -= 0x10000
    az = np.radians(az_raw.astype(np.float64) * angle_unit_deg)
    el = np.radians(el_raw.astype(np.float64) * angle_unit_deg)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1), {}


def _interpret_range_intensity_channel_azimuth(pts: np.ndarray) -> tuple:
    """
    Velodyne 風: 0-1:range, 2-3:intensity/etc, 4-5:channel/elev_idx, 6-7:azimuth, 8:flag
    elevation はチャネル番号からテーブル参照だが、簡易的に -10°〜+10° の線形分布と仮定
    """
    r = (pts[:, 0].astype(np.uint32) | (pts[:, 1].astype(np.uint32) << 8)).astype(np.float64) / 1000.0
    az_raw = (pts[:, 6].astype(np.uint32) | (pts[:, 7].astype(np.uint32) << 8))
    ch_raw = (pts[:, 4].astype(np.uint32) | (pts[:, 5].astype(np.uint32) << 8))

    # azimuth: uint16 を 0-360° にマップ
    az = np.radians(az_raw.astype(np.float64) * 360.0 / 65536)
    # elevation: チャネル番号を線形に ±10° にマップ (仮定)
    ch_max = max(ch_raw.max(), 1)
    el = np.radians(((ch_raw.astype(np.float64) / ch_max) - 0.5) * 20.0)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1), {}


def hexdump_first_packet(filepaths: list) -> None:
    """最初のファイルの先頭 200 バイトを hex dump"""
    if not filepaths:
        return
    with open(filepaths[0], "rb") as f:
        data = f.read(200)
    print(f"\n■ 最初のファイル {os.path.basename(filepaths[0])} の先頭 200 バイト")
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:04x}: {hex_part:<48}  |{ascii_part}|")


def main():
    ap = argparse.ArgumentParser(description="Falcon K2 生パケット解析")
    ap.add_argument("path")
    ap.add_argument("--bytes", action="store_true", help="バイト位置別の詳細解析")
    ap.add_argument("--score", action="store_true", help="複数のフィールド配置候補をスコアリング")
    ap.add_argument("--hexdump", action="store_true", help="最初のファイルの先頭 200B を表示")
    args = ap.parse_args()
    target = args.path.strip().rstrip('"').rstrip("'")

    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "**", "*.bin"), recursive=True))
        if not files:
            print(f"  ⚠ {target} に .bin がありません")
            sys.exit(1)
        print(f"\n  {len(files)} ファイルを発見")
    else:
        print(f"  ⚠ パス不正: {target}")
        sys.exit(1)

    # フラグなしならデフォルトで全モード実行
    if not (args.bytes or args.score or args.hexdump):
        analyze_aggregate(files)
        analyze_bytes(files)
        score_layouts(files)
        hexdump_first_packet(files)
    else:
        if args.bytes:
            analyze_bytes(files)
        if args.score:
            score_layouts(files)
        if args.hexdump:
            hexdump_first_packet(files)


if __name__ == "__main__":
    main()
