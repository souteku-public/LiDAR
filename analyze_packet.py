#!/usr/bin/env python3
"""
Falcon K2 生パケット (.bin) の詳細解析ツール
============================================

保存された packet_dump_xxx/*.bin から、各フィールドの分布を統計表示し、
正しいパース仕様を特定するためのツール。

使い方:
    # 単一ファイルを解析
    python analyze_packet.py path/to/packet_001_1440b.bin

    # ディレクトリ内全 .bin を一括解析
    python analyze_packet.py path/to/packet_dump_xxx/

出力例 (各フィールドの分布):
    range:     範囲 / ゼロ点数比率 / ヒストグラム
    azimuth:   0-360° 単位ごとの点数分布
    elevation: 同上
    flag/channel: ユニーク値とその出現数

これにより、以下の判定ができる:
    - azimuth/elevation の単位 (0.01度 vs 0.001度 vs 1/65536回転)
    - 符号付き / 符号なし
    - フィールドの割り当てが正しいか
"""

import sys
import os
import glob
import struct
import argparse
import numpy as np
from pathlib import Path


PACKET_MAGIC = bytes([0x6A, 0x17])
HEADER_SIZE  = 54
POINT_SIZE   = 9


def _load_bin(filepath: str) -> bytes:
    """1つの .bin ファイルを読み込む"""
    with open(filepath, "rb") as f:
        return f.read()


def _parse_points(data: bytes) -> tuple:
    """
    点データを 5 つの解釈で同時にデコードして返す。

    Returns:
        range_u16   : uint16 LE (mm 仮定)
        azimuth_u16 : uint16 LE
        azimuth_i16 : int16  LE
        elevation_u16: uint16 LE
        elevation_i16: int16  LE
        intensity, channel, flag : uint8
    """
    payload = data[HEADER_SIZE:]
    n = len(payload) // POINT_SIZE
    if n == 0:
        return None
    buf = np.frombuffer(payload[:n * POINT_SIZE], dtype=np.uint8).reshape(n, POINT_SIZE)

    range_u16  = buf[:, 0].astype(np.uint32) | (buf[:, 1].astype(np.uint32) << 8)
    az_u16     = buf[:, 2].astype(np.uint32) | (buf[:, 3].astype(np.uint32) << 8)
    el_u16     = buf[:, 4].astype(np.uint32) | (buf[:, 5].astype(np.uint32) << 8)
    # int16 解釈 (sign-extend)
    az_i16     = az_u16.astype(np.int32)
    az_i16[az_i16 >= 0x8000] -= 0x10000
    el_i16     = el_u16.astype(np.int32)
    el_i16[el_i16 >= 0x8000] -= 0x10000

    intensity  = buf[:, 6]
    channel    = buf[:, 7]
    flag       = buf[:, 8]
    return {
        "range_u16":  range_u16,
        "az_u16":     az_u16,
        "az_i16":     az_i16,
        "el_u16":     el_u16,
        "el_i16":     el_i16,
        "intensity":  intensity,
        "channel":    channel,
        "flag":       flag,
        "n":          n,
    }


def _hist_line(values: np.ndarray, bins: list, label_fmt: str = "{:>7.1f}") -> None:
    """分布をテキストヒストグラムで表示"""
    total = len(values)
    for low, high in bins:
        count = ((values >= low) & (values < high)).sum()
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        label_low = label_fmt.format(low)
        label_high = label_fmt.format(high)
        print(f"    {label_low} - {label_high}: {count:>7,} ({pct:5.1f}%) {bar}")


def analyze_single(filepath: str, verbose: bool = True) -> dict:
    """1ファイルを解析して点フィールドを返す"""
    data = _load_bin(filepath)
    if len(data) < HEADER_SIZE:
        print(f"  ⚠ {filepath}: too small")
        return None
    if data[:2] != PACKET_MAGIC:
        print(f"  ⚠ {filepath}: bad magic")
        return None

    if verbose:
        print(f"\n  ─── {os.path.basename(filepath)} ({len(data)} bytes) ───")

    fields = _parse_points(data)
    if fields is None:
        return None

    if verbose:
        print(f"  点数: {fields['n']}")
        # 最初の点と最後の点だけ表示
        for idx in [0, fields['n'] // 2, fields['n'] - 1]:
            print(f"  点 #{idx}: range={fields['range_u16'][idx]} mm, "
                  f"az={fields['az_u16'][idx]} (0x{fields['az_u16'][idx]:04x}), "
                  f"el={fields['el_u16'][idx]} (0x{fields['el_u16'][idx]:04x}), "
                  f"intensity={fields['intensity'][idx]}, "
                  f"ch={fields['channel'][idx]}, flag={fields['flag'][idx]}")

    return fields


def analyze_aggregate(filepaths: list) -> None:
    """全 .bin ファイルを集約して統計を表示"""
    all_fields = []
    for fp in filepaths:
        result = analyze_single(fp, verbose=False)
        if result:
            all_fields.append(result)

    if not all_fields:
        print("  ⚠ 解析できるファイルがありません")
        return

    # 全フィールドを連結
    range_all = np.concatenate([f["range_u16"] for f in all_fields])
    az_u16    = np.concatenate([f["az_u16"]    for f in all_fields])
    az_i16    = np.concatenate([f["az_i16"]    for f in all_fields])
    el_u16    = np.concatenate([f["el_u16"]    for f in all_fields])
    el_i16    = np.concatenate([f["el_i16"]    for f in all_fields])
    intensity = np.concatenate([f["intensity"] for f in all_fields])
    channel   = np.concatenate([f["channel"]   for f in all_fields])
    flag      = np.concatenate([f["flag"]      for f in all_fields])

    n_total = len(range_all)
    n_zero  = (range_all == 0).sum()
    n_nonzero = n_total - n_zero

    print(f"\n{'='*72}")
    print(f" 集約統計: {len(all_fields)} ファイル, 総点数 {n_total:,}")
    print(f"{'='*72}")

    # ── Range ──
    print(f"\n■ Range (uint16 LE, mm 仮定)")
    print(f"    全点数      : {n_total:,}")
    print(f"    range=0     : {n_zero:,} ({n_zero/n_total*100:.1f}%) ← 戻り信号なし")
    print(f"    range>0     : {n_nonzero:,} ({n_nonzero/n_total*100:.1f}%) ← 有効測定")
    if n_nonzero > 0:
        nz = range_all[range_all > 0]
        print(f"    range値 範囲: {nz.min()} 〜 {nz.max()} mm "
              f"({nz.min()/1000:.2f} 〜 {nz.max()/1000:.2f} m)")
        print(f"    range分布 (有効点のみ):")
        _hist_line(nz, [
            (1, 500), (500, 1000), (1000, 2000), (2000, 5000),
            (5000, 10000), (10000, 30000), (30000, 60000), (60000, 65536),
        ], label_fmt="{:>6.0f}mm")

    # ── Azimuth ──
    print(f"\n■ Azimuth フィールド (offset 2-3)")
    print(f"    uint16 値域 : {az_u16.min()} 〜 {az_u16.max()}")
    print(f"    int16  値域 : {az_i16.min()} 〜 {az_i16.max()}")
    print(f"\n  解釈A: uint16 × 0.01度 (現実装)")
    print(f"    実値域 = 0 〜 {az_u16.max()*0.01:.1f}°  "
          f"(360°超え= {(az_u16 >= 36000).sum():,} 点)")
    print(f"    分布:")
    _hist_line(az_u16, [
        (0, 1000), (1000, 3000), (3000, 6000), (6000, 12000),
        (12000, 18000), (18000, 24000), (24000, 30000),
        (30000, 36000), (36000, 65536),
    ], label_fmt="{:>6.0f}")

    print(f"\n  解釈B: int16 × 0.01度 (符号付き、±327度範囲)")
    print(f"    実値域 = {az_i16.min()*0.01:.1f}° 〜 {az_i16.max()*0.01:.1f}°")
    print(f"    -180〜+180度範囲: {((az_i16 >= -18000) & (az_i16 <= 18000)).sum():,} 点")

    print(f"\n  解釈C: uint16 × 360/65536 (フル円周エンコード)")
    az_deg_c = az_u16 * 360.0 / 65536
    print(f"    実値域 = {az_deg_c.min():.2f}° 〜 {az_deg_c.max():.2f}°")

    # ── Elevation ──
    print(f"\n■ Elevation フィールド (offset 4-5)")
    print(f"    uint16 値域 : {el_u16.min()} 〜 {el_u16.max()}")
    print(f"    int16  値域 : {el_i16.min()} 〜 {el_i16.max()}")
    print(f"\n  解釈A: uint16 × 0.01度 (現実装)")
    print(f"    実値域 = 0 〜 {el_u16.max()*0.01:.1f}°")
    print(f"    分布:")
    _hist_line(el_u16, [
        (0, 10), (10, 100), (100, 500), (500, 2000),
        (2000, 9000), (9000, 18000), (18000, 36000), (36000, 65536),
    ], label_fmt="{:>6.0f}")

    print(f"\n  解釈B: int16 × 0.01度")
    print(f"    実値域 = {el_i16.min()*0.01:.1f}° 〜 {el_i16.max()*0.01:.1f}°")

    # ── Intensity ──
    print(f"\n■ Intensity (uint8, offset 6)")
    print(f"    値域: {intensity.min()} 〜 {intensity.max()}, 平均: {intensity.mean():.1f}")
    print(f"    0以外: {(intensity > 0).sum():,} 点 ({(intensity > 0).sum()/n_total*100:.1f}%)")

    # ── Channel ──
    print(f"\n■ Channel (uint8, offset 7)")
    unique_ch = np.unique(channel)
    if len(unique_ch) <= 20:
        for v in unique_ch:
            count = (channel == v).sum()
            print(f"    値 {v:3d}: {count:,} ({count/n_total*100:.1f}%)")
    else:
        print(f"    {len(unique_ch)} 種類の値 (範囲 {unique_ch.min()}〜{unique_ch.max()})")

    # ── Flag ──
    print(f"\n■ Flag (uint8, offset 8)")
    unique_fl = np.unique(flag)
    if len(unique_fl) <= 20:
        for v in unique_fl:
            count = (flag == v).sum()
            print(f"    値 0x{v:02x} ({v:3d}): {count:,} ({count/n_total*100:.1f}%)")
    else:
        print(f"    {len(unique_fl)} 種類の値")

    # ── 診断結果 ──
    print(f"\n{'='*72}")
    print(f" ◆ 診断結果")
    print(f"{'='*72}")
    valid_range = range_all > 0

    # Azimuth の解釈を判定
    az_high = (az_u16 >= 36000).sum() / max(n_nonzero, 1) * 100
    if az_high < 1:
        print(f"  ✓ Azimuth は 解釈A (uint16 × 0.01度, 0-360°範囲) が妥当")
        print(f"    → 現実装と同じで OK")
    else:
        print(f"  ⚠ Azimuth が解釈A だと 360°超える点が {az_high:.1f}%")
        if az_i16.min() < -1000 and az_i16.max() < 18000:
            print(f"    → 解釈B (int16 × 0.01度) のほうが自然")
            print(f"      実値域: {az_i16.min()*0.01:.1f}° 〜 {az_i16.max()*0.01:.1f}°")
        else:
            print(f"    → 解釈C か、フィールド割り当てが違う可能性")

    # Elevation の解釈を判定
    el_max = el_u16.max()
    if el_max < 9000:
        print(f"\n  ✓ Elevation 値域は 0〜{el_max*0.01:.1f}° (前方視 LiDAR として妥当)")
    elif el_i16.min() < -1000 and abs(el_i16.max()) < 9000:
        print(f"\n  ✓ Elevation は int16 解釈すると ±{abs(el_i16).max()*0.01:.1f}° (妥当)")
    else:
        print(f"\n  ⚠ Elevation 値域が大きすぎ ({el_u16.min()}〜{el_u16.max()})")
        print(f"    → フィールド割り当てが違う可能性、または別の単位")

    # FOV のヒント
    if n_nonzero > 0:
        valid_az = az_u16[valid_range]
        valid_el = el_u16[valid_range]
        print(f"\n  有効測定点(range>0)の角度分布:")
        print(f"    Azimuth   範囲 : {valid_az.min()}〜{valid_az.max()} "
              f"(0.01°換算: {valid_az.min()*0.01:.1f}° 〜 {valid_az.max()*0.01:.1f}°)")
        print(f"    Elevation 範囲 : {valid_el.min()}〜{valid_el.max()} "
              f"(0.01°換算: {valid_el.min()*0.01:.1f}° 〜 {valid_el.max()*0.01:.1f}°)")


def main():
    ap = argparse.ArgumentParser(description="Falcon K2 生パケット解析ツール")
    ap.add_argument("path", help=".bin ファイル または .bin を含むディレクトリ")
    args = ap.parse_args()

    target = args.path.strip().rstrip('"').rstrip("'")

    if os.path.isfile(target):
        analyze_single(target, verbose=True)
        analyze_aggregate([target])
    elif os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "**", "*.bin"), recursive=True))
        if not files:
            print(f"  ⚠ {target} に .bin ファイルがありません")
            sys.exit(1)
        print(f"\n  {len(files)} ファイルを発見:")
        for f in files[:5]:
            print(f"    - {os.path.basename(f)} ({os.path.getsize(f)} bytes)")
        if len(files) > 5:
            print(f"    ...他 {len(files) - 5} ファイル")
        analyze_aggregate(files)
    else:
        print(f"  ⚠ パスが見つかりません: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
