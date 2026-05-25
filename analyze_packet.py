#!/usr/bin/env python3
"""
Falcon K2 生パケット (.bin) の詳細解析ツール (XYZ直交座標フォーマット版)
==========================================================================

保存された packet_dump_xxx/*.bin から、X/Y/Z の分布・飽和率・相関を
表示し、パース仕様の正しさを検証する。

使い方:
    python analyze_packet.py path/to/packet_001_1440b.bin
    python analyze_packet.py path/to/packet_dump_xxx/

仕様 (確定):
    Header: 54 bytes
    Point: 9 bytes
      bytes 0-1: X uint16 LE  [mm]   常に正、前方距離
      bytes 2-3: Y int16  LE  [mm]   横方向
      bytes 4-5: Z int16  LE  [mm]   高さ方向
      byte 6:    intensity uint8
      bytes 7-8: metadata (未確定)

無効点判定:
    X == 0       : 戻り信号なし
    X == 0xFFFF  : 距離オーバーフロー (65.535m 飽和)
    |Y| or |Z| == int16境界 (32767/-32768) : 無効測定マーカー
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


def main():
    ap = argparse.ArgumentParser(description="Falcon K2 生パケット解析 (XYZフォーマット版)")
    ap.add_argument("path")
    args = ap.parse_args()
    target = args.path.strip().rstrip('"').rstrip("'")

    if os.path.isfile(target):
        analyze_aggregate([target])
    elif os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "**", "*.bin"), recursive=True))
        if not files:
            print(f"  ⚠ {target} に .bin がありません")
            sys.exit(1)
        print(f"\n  {len(files)} ファイルを発見")
        analyze_aggregate(files)
    else:
        print(f"  ⚠ パス不正: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
