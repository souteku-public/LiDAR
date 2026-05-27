#!/usr/bin/env python3
"""
Falcon K2 LiDAR Recorder - PCD ビューアー
==========================================

保存された .pcd ファイルを 3D で表示するツール。

使い方:
    # 連番アニメーション再生 (PyQt5 + matplotlib GUI)
    python viewer.py --animate path/to/session_dir/
    python viewer.py --animate path/to/session_dir/ --fps 20
    python viewer.py --animate path/to/session_dir/ --loop
    python viewer.py --animate path/to/session_dir/ --accumulate

    # 1 ファイルを指定して表示
    python viewer.py path/to/file.pcd

    # ディレクトリを指定 → ファイル一覧から選んで表示
    python viewer.py path/to/session_dir/

    # ディレクトリ内容の統計を表示 (データ診断)
    python viewer.py --inspect path/to/session_dir/

    # 引数なしで起動 → デフォルト出力先から選択
    python viewer.py

必要ライブラリ:
    pip install PyQt5 matplotlib numpy   # アニメーションビューアー必須
    pip install open3d                   # 単ファイル表示の推奨 (なければ matplotlib)
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
    """NaN / Inf を含む点を除去して有効点のみを返す。"""
    xyz = points[:, :3]
    valid = np.isfinite(xyz).all(axis=1)
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


# ── 単ファイルビューアー ──────────────────────────────────────────────────────

def view_with_open3d(points: np.ndarray, title: str = "PCD Viewer") -> None:
    """open3d を使った 3D 表示"""
    import open3d as o3d

    pts = filter_valid(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))

    if len(pts) > 0:
        z = pts[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            norm = (z - z_min) / (z_max - z_min)
        else:
            norm = np.ones(len(pts)) * 0.5
        colors = np.stack([norm, norm, norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    print("\n  操作: マウス左=回転 / 右=移動 / ホイール=ズーム / Q=終了")
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1024, height=768)


def view_with_matplotlib(points: np.ndarray, title: str = "PCD Viewer") -> None:
    """matplotlib を使った簡易 3D 表示"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pts = filter_valid(points)
    if len(pts) == 0:
        print("  ⚠ 有効な点がないため表示できません")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    xyz = pts[:, :3]

    MAX_DISPLAY = 50_000
    if len(xyz) > MAX_DISPLAY:
        idx = np.random.choice(len(xyz), MAX_DISPLAY, replace=False)
        xyz = xyz[idx]

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               s=0.5, c=xyz[:, 2], cmap="viridis", alpha=0.7)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def view(filepath: str) -> None:
    """PCD ファイルを表示する"""
    print(f"\n読み込み中: {filepath}")
    points = read_pcd(filepath)
    pcd_info(filepath, points)

    if len(points) == 0:
        print("  ⚠ 点群データが空です。")
        return

    valid = filter_valid(points)
    if len(valid) == 0:
        print("  ⚠ 有効な点が 0 点です。")
        return

    title = os.path.basename(filepath)

    try:
        import open3d  # noqa: F401
        view_with_open3d(points, title)
        return
    except ImportError:
        pass

    try:
        import matplotlib  # noqa: F401
        view_with_matplotlib(points, title)
        return
    except ImportError:
        pass

    print("  ⚠ open3d か matplotlib をインストールしてください。")


# ── ディレクトリ一括検査 ──────────────────────────────────────────────────────

def inspect_directory(directory: str, sample_count: int = 5) -> None:
    """ディレクトリ内の全 PCD ファイルを検査して統計を表示する。"""
    files = sorted(glob.glob(os.path.join(directory, "**", "*.pcd"), recursive=True))
    if not files:
        print(f"  ⚠ {directory} に .pcd ファイルが見つかりません")
        return

    print(f"\n  対象ディレクトリ : {directory}")
    print(f"  ファイル数      : {len(files)}")
    print("  ─" * 35)

    total_pts = 0
    total_valid = 0
    total_size = 0
    all_x: list = []
    all_y: list = []
    all_z: list = []
    sample_indices = sorted(set(
        list(range(min(sample_count, len(files)))) +
        list(range(max(0, len(files) - sample_count), len(files)))
    ))

    for i, fp in enumerate(files):
        size_kb = os.path.getsize(fp) / 1024
        total_size += size_kb
        try:
            pts = read_pcd(fp)
        except Exception as e:
            print(f"  [{i+1:4d}] ⚠ 読み込みエラー: {os.path.basename(fp)} ({e})")
            continue

        valid = filter_valid(pts)
        n_pts, n_valid = len(pts), len(valid)
        total_pts += n_pts
        total_valid += n_valid

        if n_valid > 0:
            all_x.extend([valid[:, 0].min(), valid[:, 0].max()])
            all_y.extend([valid[:, 1].min(), valid[:, 1].max()])
            all_z.extend([valid[:, 2].min(), valid[:, 2].max()])

        if i in sample_indices:
            fname = os.path.basename(fp)[:48]
            if n_valid > 0:
                x_rng = f"X[{valid[:,0].min():+7.2f},{valid[:,0].max():+7.2f}]"
                y_rng = f"Y[{valid[:,1].min():+7.2f},{valid[:,1].max():+7.2f}]"
                z_rng = f"Z[{valid[:,2].min():+7.2f},{valid[:,2].max():+7.2f}]"
                print(f"  [{i+1:4d}] {size_kb:7.1f}KB  {n_pts:>7,}pts ({n_valid:>6,}有効) "
                      f"{x_rng} {y_rng} {z_rng}")
            else:
                print(f"  [{i+1:4d}] {size_kb:7.1f}KB  {n_pts:>7,}pts (有効 0)  ⚠ 全点NaN")
        elif i == sample_count and len(files) > 2 * sample_count:
            print(f"        ... ({len(files) - 2 * sample_count} ファイル省略) ...")

    print("  ─" * 35)
    print(f"  総容量        : {total_size:.1f} KB ({total_size/1024:.1f} MB)")
    print(f"  総点数        : {total_pts:,}")
    if total_pts > 0:
        print(f"  有効点数      : {total_valid:,} ({total_valid/total_pts*100:.1f}%)")
    if all_x:
        print(f"  全体 X 範囲   : {min(all_x):+8.3f} 〜 {max(all_x):+8.3f} m")
        print(f"  全体 Y 範囲   : {min(all_y):+8.3f} 〜 {max(all_y):+8.3f} m")
        print(f"  全体 Z 範囲   : {min(all_z):+8.3f} 〜 {max(all_z):+8.3f} m")
    print("  ─" * 35)

    print("\n  ◆ 診断結果:")
    if total_valid == 0:
        print("    ✗ 有効点が 1 つもありません。パーサーの XYZ フォーマット不一致の可能性。")
    elif all_x and (max(abs(min(all_x)), abs(max(all_x))) > 1000 or
                   max(abs(min(all_y)), abs(max(all_y))) > 1000):
        print("    ⚠ 座標値が異常に大きいです (>1000m)")
    elif all_x and (max(all_x) - min(all_x)) < 0.1:
        print("    ⚠ 座標範囲が極端に狭いです (<0.1m)")
    else:
        print("    ✓ データは概ね妥当に見えます。")
        if all_x:
            print(f"      測定範囲 X:{max(all_x)-min(all_x):.1f}m  "
                  f"Y:{max(all_y)-min(all_y):.1f}m  "
                  f"Z:{max(all_z)-min(all_z):.1f}m")

    if len(files) > 1:
        first_size = os.path.getsize(files[0]) / 1024
        avg_other = sum(os.path.getsize(f) for f in files[1:]) / len(files[1:]) / 1024
        ratio = first_size / avg_other if avg_other > 0 else float("inf")
        print(f"\n  ◆ フレーム境界検出:")
        print(f"    1枚目: {first_size:.1f} KB  /  2枚目以降平均: {avg_other:.1f} KB"
              f"  (比率: {ratio:.1f}倍)")


# ── PyQt5 シーケンスビューアー ────────────────────────────────────────────────

def run_sequence_viewer(
    directory: str,
    fps: float = 10.0,
    accumulate: bool = False,
    loop: bool = False,
) -> None:
    """PyQt5 + matplotlib によるシーケンスビューアーを起動する。"""

    # ── 依存ライブラリの遅延インポート ──
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget,
            QVBoxLayout, QHBoxLayout,
            QPushButton, QCheckBox, QLabel, QSlider, QProgressBar,
            QSizePolicy,
        )
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QColor, QPalette, QFont
    except ImportError:
        print("  ⚠ PyQt5 が必要です: pip install PyQt5")
        return

    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        import matplotlib
        matplotlib.use("Qt5Agg")
    except ImportError:
        print("  ⚠ matplotlib が必要です: pip install matplotlib")
        return

    files = sorted(glob.glob(os.path.join(directory, "**", "*.pcd"), recursive=True))
    if not files:
        print(f"  ⚠ {directory} に .pcd ファイルが見つかりません")
        return

    print(f"  {len(files)} ファイルをビューアーで開きます...")

    # ─────────────────────────────────────────────────────────────────────────
    class PCDViewerWindow(QMainWindow):
        """PyQt5 + matplotlib PCD シーケンスビューアー"""

        MAX_PTS = 60_000   # 描画上限 (ダウンサンプリング)

        def __init__(self):
            super().__init__()
            self._files      = files
            self._idx        = 0
            self._fps        = fps
            self._playing    = False
            self._accum_pts: list = []
            self._elev       = 20.0   # matplotlib 3D 視点 elevation
            self._azim       = -60.0  # matplotlib 3D 視点 azimuth

            self._build_ui()
            self._apply_dark_theme()
            self.setWindowTitle(
                f"PCD Viewer  —  {len(files)} ファイル  |  "
                f"{os.path.basename(directory)}"
            )
            self.resize(1100, 840)

            # タイマー
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._timer_tick)

            # 初期表示: 最大ファイルで視点を合わせてから先頭へ
            init_idx = max(range(len(files)),
                           key=lambda i: os.path.getsize(files[i]))
            self._load_frame(init_idx, reset_view=True)
            self._load_frame(0, reset_view=False)

        # ── UI 構築 ───────────────────────────────────────────────────────────

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)
            root.setContentsMargins(6, 6, 6, 6)
            root.setSpacing(4)

            # ── matplotlib 3D キャンバス ──
            self._fig = Figure(facecolor="#1a1a2e")
            self._ax  = self._fig.add_subplot(111, projection="3d")
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            root.addWidget(self._canvas, stretch=1)

            # ── シークバー ──
            self._seekbar = QProgressBar()
            self._seekbar.setRange(0, max(1, len(files) - 1))
            self._seekbar.setValue(0)
            self._seekbar.setTextVisible(False)
            self._seekbar.setFixedHeight(6)
            root.addWidget(self._seekbar)

            # ── トランスポート行 ──
            tr_widget = QWidget()
            tr = QHBoxLayout(tr_widget)
            tr.setContentsMargins(0, 2, 0, 2)
            tr.setSpacing(4)

            self._btn_first = self._mkbtn("⏮", "先頭へ (Home)",        self._on_first, 36)
            self._btn_prev  = self._mkbtn("◀",  "前のフレーム (←)",     self._on_prev,  36)
            self._btn_play  = self._mkbtn("▶",  "再生 / 一時停止 (Space)", self._on_play_toggle, 52)
            self._btn_next  = self._mkbtn("▶|", "次のフレーム (→)",     self._on_next,  36)
            self._btn_last  = self._mkbtn("⏭",  "末尾へ (End)",          self._on_last,  36)

            for b in [self._btn_first, self._btn_prev, self._btn_play,
                      self._btn_next, self._btn_last]:
                b.setFixedHeight(36)
                tr.addWidget(b)

            tr.addSpacing(12)

            lbl_fps = QLabel("FPS:")
            lbl_fps.setFixedWidth(28)
            tr.addWidget(lbl_fps)

            self._fps_slider = QSlider(Qt.Horizontal)
            self._fps_slider.setRange(1, 60)
            self._fps_slider.setValue(int(self._fps))
            self._fps_slider.setFixedWidth(120)
            self._fps_slider.setToolTip("再生速度を調整 (1〜60 FPS)")
            tr.addWidget(self._fps_slider)

            self._lbl_fps = QLabel(f"{int(self._fps)}")
            self._lbl_fps.setFixedWidth(26)
            tr.addWidget(self._lbl_fps)

            tr.addStretch()

            self._lbl_info = QLabel("")
            self._lbl_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._lbl_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            tr.addWidget(self._lbl_info)

            root.addWidget(tr_widget)

            # ── オプション行 ──
            opt_widget = QWidget()
            opt = QHBoxLayout(opt_widget)
            opt.setContentsMargins(0, 2, 0, 2)
            opt.setSpacing(12)

            self._cb_loop   = QCheckBox("↺  ループ再生")
            self._cb_filter = QCheckBox("フィルタ（有効点のみ）")
            self._cb_accum  = QCheckBox("累積表示")

            self._cb_loop.setChecked(loop)
            self._cb_filter.setChecked(True)
            self._cb_accum.setChecked(accumulate)

            self._cb_loop.setToolTip(
                "ON: 最終フレームで先頭に戻りループ再生\n"
                "OFF: 最終フレームで停止"
            )
            self._cb_filter.setToolTip(
                "ON: NaN/Inf を含む無効点を除外して表示（推奨）\n"
                "OFF: ファイル内のすべての点をそのまま表示"
            )
            self._cb_accum.setToolTip(
                "ON: フレームを重ねて累積表示（差分PCDの全体像を確認するのに便利）\n"
                "OFF: 各フレームを単独表示"
            )

            opt.addWidget(self._cb_loop)
            opt.addWidget(self._cb_filter)
            opt.addWidget(self._cb_accum)

            self._btn_clear = QPushButton("累積クリア")
            self._btn_clear.setFixedHeight(28)
            self._btn_clear.setEnabled(accumulate)
            self._btn_clear.setToolTip("累積表示をリセットして先頭フレームから再描画")
            opt.addWidget(self._btn_clear)

            opt.addStretch()

            self._btn_reset = QPushButton("視点リセット")
            self._btn_reset.setFixedHeight(28)
            self._btn_reset.setToolTip("3D ビューの視点を初期状態に戻す")
            opt.addWidget(self._btn_reset)

            root.addWidget(opt_widget)

            # ── シグナル接続 ──
            self._fps_slider.valueChanged.connect(self._on_fps_changed)
            self._cb_loop.toggled.connect(lambda _: None)   # 状態は isChecked() で参照
            self._cb_filter.toggled.connect(self._on_filter_toggled)
            self._cb_accum.toggled.connect(self._on_accum_toggled)
            self._btn_clear.clicked.connect(self._on_clear_accum)
            self._btn_reset.clicked.connect(self._on_reset_view)

        @staticmethod
        def _mkbtn(text, tip, slot, w):
            b = QPushButton(text)
            b.setFixedWidth(w)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            return b

        def _apply_dark_theme(self):
            """ダークテーマを適用する"""
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #1e1e2e;
                    color: #cdd6f4;
                }
                QPushButton {
                    background-color: #313244;
                    color: #cdd6f4;
                    border: 1px solid #45475a;
                    border-radius: 4px;
                    padding: 2px 6px;
                }
                QPushButton:hover   { background-color: #45475a; }
                QPushButton:pressed { background-color: #585b70; }
                QPushButton:disabled{ background-color: #1e1e2e; color: #6c7086; }
                QCheckBox { color: #cdd6f4; spacing: 6px; }
                QCheckBox::indicator {
                    width: 16px; height: 16px;
                    border: 1px solid #45475a;
                    border-radius: 3px;
                    background: #313244;
                }
                QCheckBox::indicator:checked {
                    background: #89b4fa;
                    border-color: #89b4fa;
                }
                QLabel  { color: #cdd6f4; }
                QSlider::groove:horizontal {
                    height: 4px;
                    background: #45475a;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background: #89b4fa;
                    width: 14px; height: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }
                QProgressBar {
                    background: #313244;
                    border: none;
                    border-radius: 3px;
                }
                QProgressBar::chunk {
                    background: #89b4fa;
                    border-radius: 3px;
                }
            """)

        # ── フレーム読み込み・描画 ─────────────────────────────────────────────

        def _load_frame(self, idx: int, reset_view: bool = False) -> None:
            if not self._files:
                return
            idx = max(0, min(idx, len(self._files) - 1))
            self._idx = idx

            try:
                raw = read_pcd(self._files[idx])
                pts = filter_valid(raw) if self._cb_filter.isChecked() else raw
            except Exception as e:
                print(f"  読み込み失敗: {self._files[idx]} ({e})")
                return

            if self._cb_accum.isChecked():
                if len(pts) > 0:
                    self._accum_pts.append(pts)
                pts = (np.concatenate(self._accum_pts, axis=0)
                       if self._accum_pts else pts)

            self._draw_pts(pts, reset_view=reset_view)

            # UI 更新
            fname = os.path.basename(self._files[idx])
            n_pts = len(pts)
            self._lbl_info.setText(
                f"[{idx+1:>5} / {len(self._files)}]  {fname}  ({n_pts:,} pts)"
            )
            self._seekbar.setValue(idx)

        def _draw_pts(self, pts: np.ndarray, reset_view: bool = False) -> None:
            """matplotlib 3D 散布図を再描画する"""
            # 視点を保存
            if not reset_view:
                try:
                    self._elev = self._ax.elev
                    self._azim = self._ax.azim
                except Exception:
                    pass

            self._ax.cla()
            self._ax.set_facecolor("#0d1117")
            self._fig.patch.set_facecolor("#1a1a2e")

            if len(pts) > 0:
                xyz = pts[:, :3]

                # ダウンサンプリング
                if len(xyz) > self.MAX_PTS:
                    rng = np.random.default_rng(42)
                    sel = rng.choice(len(xyz), self.MAX_PTS, replace=False)
                    xyz = xyz[sel]

                # Z 方向グラデーション (viridis 風)
                z = xyz[:, 2]
                z_min, z_max = z.min(), z.max()
                norm = ((z - z_min) / (z_max - z_min)
                        if z_max > z_min else np.full(len(z), 0.5))

                r = np.clip(1.5 - 4 * np.abs(norm - 0.75), 0, 1)
                g = np.clip(1.5 - 4 * np.abs(norm - 0.50), 0, 1)
                b = np.clip(1.5 - 4 * np.abs(norm - 0.25), 0, 1)
                colors = np.stack([r, g, b], axis=1)

                self._ax.scatter(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=colors, s=0.8, alpha=0.85, linewidths=0,
                )

            # 軸スタイル
            for spine in [self._ax.xaxis, self._ax.yaxis, self._ax.zaxis]:
                spine.label.set_color("#6c7086")
                spine.set_tick_params(labelcolor="#6c7086", labelsize=7)
                spine.line.set_color("#313244")
                spine.pane.set_edgecolor("#313244")
                spine.pane.fill = False

            self._ax.set_xlabel("X [m]", labelpad=2)
            self._ax.set_ylabel("Y [m]", labelpad=2)
            self._ax.set_zlabel("Z [m]", labelpad=2)
            self._ax.view_init(elev=self._elev, azim=self._azim)
            self._canvas.draw_idle()

        # ── タイマー ─────────────────────────────────────────────────────────

        def _timer_tick(self):
            if self._idx < len(self._files) - 1:
                self._load_frame(self._idx + 1)
            elif self._cb_loop.isChecked():
                if self._cb_accum.isChecked():
                    self._accum_pts.clear()
                self._load_frame(0)
            else:
                self._on_play_toggle()   # 停止

        # ── トランスポートスロット ─────────────────────────────────────────────

        def _on_play_toggle(self):
            self._playing = not self._playing
            if self._playing:
                self._btn_play.setText("⏸")
                self._timer.start(max(17, int(1000 / self._fps)))
            else:
                self._btn_play.setText("▶")
                self._timer.stop()

        def _on_first(self):
            was = self._playing
            if was: self._on_play_toggle()
            self._load_frame(0)
            if was: self._on_play_toggle()

        def _on_prev(self):
            was = self._playing
            if was: self._on_play_toggle()
            if self._cb_accum.isChecked() and self._accum_pts:
                self._accum_pts.pop()
            self._load_frame(self._idx - 1)
            if was: self._on_play_toggle()

        def _on_next(self):
            was = self._playing
            if was: self._on_play_toggle()
            self._load_frame(self._idx + 1)
            if was: self._on_play_toggle()

        def _on_last(self):
            was = self._playing
            if was: self._on_play_toggle()
            self._load_frame(len(self._files) - 1)
            if was: self._on_play_toggle()

        def _on_fps_changed(self, value: int):
            self._fps = float(value)
            self._lbl_fps.setText(str(value))
            if self._playing:
                self._timer.start(max(17, int(1000 / self._fps)))

        def _on_filter_toggled(self, _):
            self._load_frame(self._idx)

        def _on_accum_toggled(self, checked: bool):
            self._btn_clear.setEnabled(checked)
            if not checked:
                self._accum_pts.clear()
            self._load_frame(self._idx)

        def _on_clear_accum(self):
            self._accum_pts.clear()
            self._load_frame(self._idx)

        def _on_reset_view(self):
            self._elev = 20.0
            self._azim = -60.0
            self._draw_pts(
                filter_valid(read_pcd(self._files[self._idx]))
                if self._cb_filter.isChecked()
                else read_pcd(self._files[self._idx]),
                reset_view=True,
            )

        # ── キーボードショートカット ──────────────────────────────────────────

        def keyPressEvent(self, event):
            key = event.key()
            from PyQt5.QtCore import Qt as _Qt
            if   key == _Qt.Key_Space:  self._on_play_toggle()
            elif key == _Qt.Key_Left:   self._on_prev()
            elif key == _Qt.Key_Right:  self._on_next()
            elif key == _Qt.Key_Home:   self._on_first()
            elif key == _Qt.Key_End:    self._on_last()
            elif key == _Qt.Key_L:      self._cb_loop.toggle()
            elif key == _Qt.Key_F:      self._cb_filter.toggle()
            elif key == _Qt.Key_C:      self._on_clear_accum()
            elif key == _Qt.Key_R:      self._on_reset_view()
            else: super().keyPressEvent(event)

        def closeEvent(self, event):
            self._timer.stop()
            event.accept()

    # ── アプリ起動 ────────────────────────────────────────────────────────────
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    win = PCDViewerWindow()
    win.show()
    sys.exit(app.exec_())


# ── ファイル選択 ──────────────────────────────────────────────────────────────

def pick_file(directory: str) -> "str | None":
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
        default=r"C:/Users/ntlx4/OneDrive/デスクトップ/LiDAR/falcon k2",
        help=".pcd ファイルまたはセッションディレクトリのパス",
    )
    parser.add_argument(
        "--animate", "-a",
        action="store_true",
        help="PyQt5 GUI シーケンスビューアーを起動 (ディレクトリ指定必須)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="初期再生フレームレート [Hz] (default: 10)",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="点群を累積表示する (差分PCDの全体像を見るのに便利)",
    )
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        help="最終フレームで先頭に戻りループ再生する",
    )
    parser.add_argument(
        "--inspect", "-i",
        action="store_true",
        help="ディレクトリ内全PCDファイルの統計を表示する (データ診断用)",
    )
    args = parser.parse_args()

    # PowerShell の末尾 "\" エスケープ対策
    target = args.path.strip().rstrip('"').rstrip("'")

    if args.inspect:
        if not os.path.isdir(target):
            print(f"  ⚠ --inspect にはディレクトリを指定してください: {target}")
            sys.exit(1)
        inspect_directory(target)
        return

    if args.animate:
        if not os.path.isdir(target):
            print(f"  ⚠ --animate にはディレクトリを指定してください: {target}")
            sys.exit(1)
        run_sequence_viewer(target, fps=args.fps,
                            accumulate=args.accumulate, loop=args.loop)
        return

    if os.path.isfile(target):
        if not target.endswith(".pcd"):
            print(f"  ⚠ {target} は .pcd ファイルではありません。")
            sys.exit(1)
        view(target)

    elif os.path.isdir(target):
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
