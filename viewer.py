#!/usr/bin/env python3
"""
Falcon K2 LiDAR Recorder - PCD ビューアー
==========================================

保存された .pcd ファイルを 3D で表示するツール。

使い方:
    # 1 ファイルを指定して表示
    python viewer.py path/to/file.pcd

    # ディレクトリを指定 → ファイル一覧から選んで表示
    python viewer.py path/to/session_dir/

    # 連番アニメーション再生 (open3d 必須)
    python viewer.py --animate path/to/session_dir/
    python viewer.py --animate path/to/session_dir/ --fps 20
    python viewer.py --animate path/to/session_dir/ --accumulate

    # 引数なしで起動 → デフォルト出力先から選択
    python viewer.py

必要ライブラリ:
    pip install open3d          # 3D ビューアー (推奨)
    pip install matplotlib      # open3d がない場合のフォールバック
    pip install numpy           # 必須 (既にインストール済みのはず)
"""

import sys
import os
import struct
import glob
import time
import argparse
import numpy as np
from pathlib import Path


# ── PCD 読み込み ──────────────────────────────────────────────────────────────

def read_pcd(filepath: str) -> np.ndarray:
    """
    PCD v0.7 バイナリファイルを読み込んで (N,4) float32 配列を返す。
    columns: x, y, z, intensity
    """
    with open(filepath, "rb") as f:
        n_points = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("POINTS"):
                n_points = int(line.split()[1])
            if line == "DATA binary":
                break
        binary = f.read()

    if n_points == 0:
        return np.empty((0, 4), dtype=np.float32)

    points = np.frombuffer(binary, dtype=np.float32).reshape(-1, 4)
    return points


def filter_valid(points: np.ndarray) -> np.ndarray:
    """
    NaN / Inf を含む点を除去して有効点のみを返す。

    Falcon K2 は「戻り信号なし」の点を float32 NaN で送信する。
    表示や変化検出の前にこのフィルターを通す。
    """
    xyz = points[:, :3]
    valid = np.isfinite(xyz).all(axis=1)   # x,y,z がすべて有限値の行
    return points[valid]


def pcd_info(filepath: str, points: np.ndarray) -> None:
    """ファイル情報をコンソールに出力する"""
    size_kb = os.path.getsize(filepath) / 1024
    valid = filter_valid(points)
    nan_count = len(points) - len(valid)

    print(f"\n{'─'*50}")
    print(f"  ファイル  : {os.path.basename(filepath)}")
    print(f"  サイズ   : {size_kb:.1f} KB")
    print(f"  総点数   : {len(points):,} 点")
    print(f"  有効点数  : {len(valid):,} 点")
    if nan_count > 0:
        print(f"  NaN/Inf  : {nan_count:,} 点 (戻り信号なし、表示から除外)")
    if len(valid) > 0:
        xyz = valid[:, :3]
        print(f"  X 範囲   : {xyz[:,0].min():.3f} 〜 {xyz[:,0].max():.3f} m")
        print(f"  Y 範囲   : {xyz[:,1].min():.3f} 〜 {xyz[:,1].max():.3f} m")
        print(f"  Z 範囲   : {xyz[:,2].min():.3f} 〜 {xyz[:,2].max():.3f} m")
    else:
        print("  ⚠ 有効な点がありません")
    print(f"{'─'*50}")


# ── ビューアー本体 ────────────────────────────────────────────────────────────

def view_with_open3d(points: np.ndarray, title: str = "PCD Viewer") -> None:
    """open3d を使った 3D 表示"""
    import open3d as o3d

    pts = filter_valid(points)   # NaN / Inf を除去

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))

    # Z 高さを色にマップ (グレースケール)
    if len(pts) > 0:
        z = pts[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            norm = (z - z_min) / (z_max - z_min)
        else:
            norm = np.ones(len(pts)) * 0.5
        colors = np.stack([norm, norm, norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    print("\n  操作方法:")
    print("    マウス左ドラッグ  : 回転")
    print("    マウス右ドラッグ  : 移動")
    print("    ホイール         : ズーム")
    print("    Q / Esc         : 終了")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
        width=1024,
        height=768,
    )


def view_with_matplotlib(points: np.ndarray, title: str = "PCD Viewer") -> None:
    """matplotlib を使った簡易 3D 表示 (open3d がない場合)"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = filter_valid(points)   # NaN / Inf を除去
    if len(pts) == 0:
        print("  ⚠ 有効な点がないため表示できません")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xyz = pts[:, :3]

    # 点数が多い場合はダウンサンプリング
    MAX_DISPLAY = 50_000
    if len(xyz) > MAX_DISPLAY:
        idx = np.random.choice(len(xyz), MAX_DISPLAY, replace=False)
        xyz = xyz[idx]
        print(f"  ※ 表示点数を {MAX_DISPLAY:,} 点にダウンサンプリングしました")

    ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        s=0.5, c=xyz[:, 2],
        cmap="viridis", alpha=0.7,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def view(filepath: str) -> None:
    """PCD ファイルを表示する (ライブラリが利用可能な方を自動選択)"""
    print(f"\n読み込み中: {filepath}")
    points = read_pcd(filepath)
    pcd_info(filepath, points)

    if len(points) == 0:
        print("  ⚠ 点群データが空です。")
        return

    valid = filter_valid(points)
    if len(valid) == 0:
        print("  ⚠ 有効な点が 0 点です (全点が NaN/Inf)。")
        print("    → パケットフォーマットが正しくパースできていない可能性があります。")
        return

    title = os.path.basename(filepath)

    # open3d を優先して試みる
    try:
        import open3d  # noqa: F401
        print("  open3d で表示します...")
        view_with_open3d(points, title)
        return
    except ImportError:
        pass

    # matplotlib にフォールバック
    try:
        import matplotlib  # noqa: F401
        print("  matplotlib で表示します (簡易表示)...")
        print("  ヒント: `pip install open3d` でより高品質な表示が可能です。")
        view_with_matplotlib(points, title)
        return
    except ImportError:
        pass

    print("  ⚠ 表示ライブラリが見つかりません。")
    print("    pip install open3d  または  pip install matplotlib  を実行してください。")


# ── 連番アニメーション再生 ────────────────────────────────────────────────────

def animate_sequence(
    directory: str,
    fps: float = 10.0,
    accumulate: bool = False,
) -> None:
    """
    PCD ファイル群を連番アニメーションとして再生する (open3d 必須)。

    Parameters
    ----------
    directory  : str   PCD ファイルのあるディレクトリ (再帰検索)
    fps        : float 再生フレームレート [Hz]
    accumulate : bool  True にすると点群を累積表示する
                       (差分PCDの全体像を見たいとき)。
                       False は各PCDを単独表示。

    キー操作:
        SPACE  : 再生 / 一時停止
        →      : 次のフレーム (一時停止)
        ←      : 前のフレーム (一時停止)
        =      : 再生速度 1.5 倍
        -      : 再生速度 1/1.5
        R      : 視点リセット
        C      : 累積表示クリア (accumulate モード時)
        Q / Esc: 終了
    """
    files = sorted(glob.glob(os.path.join(directory, "**", "*.pcd"), recursive=True))
    if not files:
        print(f"  ⚠ {directory} に .pcd ファイルが見つかりません")
        return

    try:
        import open3d as o3d
    except ImportError:
        print("  ⚠ open3d が必要です。`pip install open3d` を実行してください。")
        return

    mode = "累積表示" if accumulate else "単独表示"
    print(f"\n  {len(files)} ファイルを {fps:.1f} FPS で再生 ({mode})")
    print("  ─" * 30)
    print("  操作: SPACE=再生/一時停止  ←/→=前後  +/-=速度  R=視点  C=累積クリア  Q=終了")
    print("  ─" * 30)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"PCD Sequence ({len(files)} files)",
        width=1024, height=768,
    )

    state = {
        "idx": 0, "playing": True, "fps": fps,
        "last_update": time.time(),
        "accumulated_points": [],   # for accumulate mode
    }

    pcd_geom = o3d.geometry.PointCloud()

    def _colorize(pts: np.ndarray) -> np.ndarray:
        z = pts[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            norm = (z - z_min) / (z_max - z_min)
        else:
            norm = np.ones(len(pts)) * 0.5
        # viridis 風配色 (青→緑→黄)
        r = np.clip(1.5 - 4 * np.abs(norm - 0.75), 0, 1)
        g = np.clip(1.5 - 4 * np.abs(norm - 0.5),  0, 1)
        b = np.clip(1.5 - 4 * np.abs(norm - 0.25), 0, 1)
        return np.stack([r, g, b], axis=1)

    def _load(i: int) -> None:
        try:
            pts = filter_valid(read_pcd(files[i]))
        except Exception as e:  # noqa: BLE001
            print(f"\n  読み込み失敗: {files[i]} ({e})")
            return
        if accumulate:
            if len(pts) > 0:
                state["accumulated_points"].append(pts)
            if state["accumulated_points"]:
                pts = np.concatenate(state["accumulated_points"], axis=0)
        if len(pts) == 0:
            return
        pcd_geom.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        pcd_geom.colors = o3d.utility.Vector3dVector(_colorize(pts).astype(np.float64))
        fname = os.path.basename(files[i])
        print(f"\r  [{i+1:5d}/{len(files):5d}] {fname:50s} ({len(pts):>8,} pts)",
              end="", flush=True)

    # ── キーコールバック ──
    def toggle_play(_vis):
        state["playing"] = not state["playing"]
        return False

    def next_frame(_vis):
        state["playing"] = False
        if state["idx"] < len(files) - 1:
            state["idx"] += 1
            _load(state["idx"])
            _vis.update_geometry(pcd_geom)
        return False

    def prev_frame(_vis):
        state["playing"] = False
        if state["idx"] > 0:
            if accumulate and state["accumulated_points"]:
                state["accumulated_points"].pop()
            state["idx"] -= 1
            _load(state["idx"])
            _vis.update_geometry(pcd_geom)
        return False

    def speed_up(_vis):
        state["fps"] = min(120.0, state["fps"] * 1.5)
        print(f"\n  速度: {state['fps']:.1f} FPS")
        return False

    def speed_down(_vis):
        state["fps"] = max(0.5, state["fps"] / 1.5)
        print(f"\n  速度: {state['fps']:.1f} FPS")
        return False

    def reset_view(_vis):
        _vis.reset_view_point(True)
        return False

    def clear_accum(_vis):
        if accumulate:
            state["accumulated_points"].clear()
            print("\n  累積クリア")
        return False

    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(262, next_frame)   # GLFW_KEY_RIGHT
    vis.register_key_callback(263, prev_frame)   # GLFW_KEY_LEFT
    vis.register_key_callback(ord("="), speed_up)
    vis.register_key_callback(ord("-"), speed_down)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("C"), clear_accum)

    # ── 初期フレームを読み込み ──
    _load(0)
    vis.add_geometry(pcd_geom)
    vis.reset_view_point(True)

    # ── メインループ ──
    try:
        while True:
            if state["playing"]:
                now = time.time()
                if now - state["last_update"] >= 1.0 / state["fps"]:
                    if state["idx"] < len(files) - 1:
                        state["idx"] += 1
                        _load(state["idx"])
                        vis.update_geometry(pcd_geom)
                    else:
                        state["playing"] = False
                        print("\n  最終フレームに到達しました")
                    state["last_update"] = now

            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        vis.destroy_window()
        print()


# ── ファイル選択 ──────────────────────────────────────────────────────────────

def pick_file(directory: str) -> str | None:
    """ディレクトリ内の PCD ファイル一覧を表示して選択させる"""
    pattern = os.path.join(directory, "**", "*.pcd")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"  ⚠ {directory} に .pcd ファイルが見つかりません。")
        return None

    print(f"\n{directory} 内の PCD ファイル一覧:\n")
    for i, f in enumerate(files):
        size_kb = os.path.getsize(f) / 1024
        rel = os.path.relpath(f, directory)
        print(f"  [{i+1:3d}] {rel}  ({size_kb:.1f} KB)")

    print("\n  番号を入力して Enter、または 0 で終了 > ", end="")
    try:
        choice = int(input().strip())
    except (ValueError, EOFError):
        return None

    if choice == 0:
        return None
    if 1 <= choice <= len(files):
        return files[choice - 1]

    print("  無効な番号です。")
    return None


# ── エントリーポイント ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Falcon K2 LiDAR Recorder - PCD ビューアー"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join(os.path.expanduser("~"), "lidar_output"),
        help=".pcd ファイルまたはセッションディレクトリのパス",
    )
    parser.add_argument(
        "--animate", "-a",
        action="store_true",
        help="連番アニメーション再生モード (ディレクトリ指定必須、open3d必須)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="アニメーション再生フレームレート [Hz] (default: 10)",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="点群を累積表示する (差分PCDの全体像を見たいとき)",
    )
    args = parser.parse_args()

    # PowerShell が末尾の "\" を次のクォートをエスケープする扱いにする落とし穴対策:
    # "C:\path\dir\" と書くとパス末尾に " が混入するため除去する
    target = args.path.strip().rstrip('"').rstrip("'")

    if args.animate:
        if not os.path.isdir(target):
            print(f"  ⚠ --animate にはディレクトリを指定してください: {target}")
            sys.exit(1)
        animate_sequence(target, fps=args.fps, accumulate=args.accumulate)
        return

    if os.path.isfile(target):
        # ファイル直接指定
        if not target.endswith(".pcd"):
            print(f"  ⚠ {target} は .pcd ファイルではありません。")
            sys.exit(1)
        view(target)

    elif os.path.isdir(target):
        # ディレクトリ → 一覧から選択
        while True:
            filepath = pick_file(target)
            if filepath is None:
                break
            view(filepath)
            print("\n  続けて別のファイルを開きますか？ (Enter=はい / Ctrl+C=終了)")
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print(f"  ⚠ パスが見つかりません: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
