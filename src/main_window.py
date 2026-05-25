"""
メインウィンドウ (PyQt5)

レイアウト概要:
  ┌─────────────────────────────────────┐
  │  接続設定   IP / Port               │
  │  録画設定   時間(秒) / 出力先        │
  │  詳細設定   ボクセルサイズ           │
  │─────────────────────────────────────│
  │  [開始]  [停止]   状態インジケーター  │
  │─────────────────────────────────────│
  │  統計情報 (受信フレーム/保存ファイル) │
  │  経過時間 / 残り時間 プログレスバー  │
  │─────────────────────────────────────│
  │  ログ (最新保存ファイル等)           │
  └─────────────────────────────────────┘
"""

import os
import sys
from datetime import timedelta
from pathlib import Path

from PyQt5.QtCore    import Qt, pyqtSlot
from PyQt5.QtGui     import QColor, QPalette, QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QStatusBar, QFrame, QSizePolicy,
)

from .recorder import Recorder, RecorderState

# ── 定数 ─────────────────────────────────────────────────────────────────────
DEFAULT_HOST      = "0.0.0.0"
DEFAULT_PORT      = 2368
DEFAULT_DURATION  = 60          # 秒
DEFAULT_VOXEL     = 0.05        # m
DEFAULT_OUTPUT    = os.path.join(os.path.expanduser("~"), "lidar_output")
MAX_LOG_LINES     = 500
MAX_DURATION_S    = 6 * 3600    # 6 時間 = 21600 秒

# ── 色定義 ────────────────────────────────────────────────────────────────────
COLOR_IDLE      = "#757575"
COLOR_RECORDING = "#4CAF50"
COLOR_ERROR     = "#F44336"
COLOR_STOPPING  = "#FF9800"


class StatusIndicator(QLabel):
    """録画状態を示す丸いインジケーターウィジェット"""

    _STATE_COLORS = {
        RecorderState.IDLE:       COLOR_IDLE,
        RecorderState.CONNECTING: COLOR_STOPPING,
        RecorderState.RECORDING:  COLOR_RECORDING,
        RecorderState.STOPPING:   COLOR_STOPPING,
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.set_state(RecorderState.IDLE)

    def set_state(self, state: str) -> None:
        color = self._STATE_COLORS.get(state, COLOR_IDLE)
        self.setStyleSheet(
            f"background-color: {color};"
            "border-radius: 10px;"
            "border: 2px solid rgba(0,0,0,0.3);"
        )
        self.setToolTip(state)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._recorder = Recorder(parent=self)
        self._connect_recorder_signals()
        self._build_ui()
        self.setWindowTitle("Falcon K2 LiDAR Recorder")
        self.resize(700, 680)

    # ── UI 構築 ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._build_connection_group())
        root.addWidget(self._build_recording_group())
        root.addWidget(self._build_advanced_group())
        root.addWidget(self._build_control_bar())
        root.addWidget(self._build_stats_group())
        root.addWidget(self._build_log_group())

        # ステータスバー
        self.statusBar().showMessage("待機中")

    # ─── 接続設定 ─────────────────────────────────────────────────────────────

    def _build_connection_group(self) -> QGroupBox:
        grp = QGroupBox("接続設定")
        layout = QGridLayout(grp)
        layout.setColumnStretch(1, 1)

        host_label = QLabel("受信 IP アドレス\n(この PC の NIC):")
        host_label.setToolTip("LiDAR ではなく、このPCのNICに割り当てられたIPアドレスを指定します")
        layout.addWidget(host_label, 0, 0)
        self._le_host = QLineEdit(DEFAULT_HOST)
        self._le_host.setPlaceholderText("0.0.0.0 (全インターフェース受信、推奨)")
        self._le_host.setToolTip(
            "【このPCのNICのIPアドレス】を入力してください。LiDARのIPではありません。\n\n"
            "  0.0.0.0  → 全NICで受信（迷ったらこれ）\n"
            "  192.168.x.x → 特定NICのみ使う場合はそのPCのIP\n\n"
            "※ LiDARのIPアドレスを入れると WinError 10049 が発生します。"
        )
        layout.addWidget(self._le_host, 0, 1)

        # ヒント表示ラベル
        hint_lbl = QLabel("⚠ LiDARのIPではなく\nこのPCのIPを指定")
        hint_lbl.setStyleSheet("color: #E65100; font-size: 10px;")
        hint_lbl.setToolTip("受信IPはこのPCのNICに付与されたIPです。迷う場合は 0.0.0.0 を使用してください。")
        layout.addWidget(hint_lbl, 0, 2)

        layout.addWidget(QLabel("UDP ポート:"), 1, 0)
        self._sb_port = QSpinBox()
        self._sb_port.setRange(1024, 65535)
        self._sb_port.setValue(DEFAULT_PORT)
        self._sb_port.setToolTip("Falcon K2 の設定に合わせてください (デフォルト: 2368)")
        layout.addWidget(self._sb_port, 1, 1)

        return grp

    # ─── 録画設定 ─────────────────────────────────────────────────────────────

    def _build_recording_group(self) -> QGroupBox:
        grp = QGroupBox("録画設定")
        layout = QGridLayout(grp)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QLabel("録画時間 (秒):"), 0, 0)
        self._sb_duration = QSpinBox()
        self._sb_duration.setRange(1, MAX_DURATION_S)
        self._sb_duration.setValue(DEFAULT_DURATION)
        self._sb_duration.setSuffix(" 秒")
        self._sb_duration.setToolTip(
            f"録画時間を秒単位で指定します (最大 {MAX_DURATION_S} 秒 = 6 時間)"
        )
        layout.addWidget(self._sb_duration, 0, 1)

        # 時間表示ラベル (参考)
        self._lbl_duration_hint = QLabel()
        self._lbl_duration_hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._lbl_duration_hint, 0, 2)
        self._sb_duration.valueChanged.connect(self._update_duration_hint)
        self._update_duration_hint(DEFAULT_DURATION)

        layout.addWidget(QLabel("出力ディレクトリ:"), 1, 0)
        dir_widget = QWidget()
        dir_layout = QHBoxLayout(dir_widget)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self._le_output = QLineEdit(DEFAULT_OUTPUT)
        self._le_output.setToolTip("PCD ファイルの保存先 (セッションごとにサブフォルダを自動作成)")
        dir_layout.addWidget(self._le_output)
        btn_browse = QPushButton("参照…")
        btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self._browse_output)
        dir_layout.addWidget(btn_browse)
        layout.addWidget(dir_widget, 1, 1, 1, 2)

        return grp

    # ─── 詳細設定 ─────────────────────────────────────────────────────────────

    def _build_advanced_group(self) -> QGroupBox:
        grp = QGroupBox("詳細設定")
        grp.setCheckable(True)
        grp.setChecked(False)
        layout = QGridLayout(grp)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QLabel("変化検出ボクセルサイズ (m):"), 0, 0)
        self._dsb_voxel = QDoubleSpinBox()
        self._dsb_voxel.setRange(0.01, 1.0)
        self._dsb_voxel.setSingleStep(0.01)
        self._dsb_voxel.setDecimals(3)
        self._dsb_voxel.setValue(DEFAULT_VOXEL)
        self._dsb_voxel.setToolTip(
            "ボクセルサイズが小さいほど細かな変化を検出します。\n"
            "大きくすると保存ファイル数が減りますがノイズ耐性も増します。"
        )
        layout.addWidget(self._dsb_voxel, 0, 1)

        return grp

    # ─── コントロールバー ─────────────────────────────────────────────────────

    def _build_control_bar(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(frame)

        self._btn_start = QPushButton("▶  録画開始")
        self._btn_start.setFixedHeight(40)
        self._btn_start.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; border-radius: 4px; font-size: 14px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #BDBDBD; }"
        )
        self._btn_start.clicked.connect(self._on_start)
        layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  停止")
        self._btn_stop.setFixedHeight(40)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; "
            "font-weight: bold; border-radius: 4px; font-size: 14px; }"
            "QPushButton:hover { background-color: #D32F2F; }"
            "QPushButton:disabled { background-color: #BDBDBD; }"
        )
        self._btn_stop.clicked.connect(self._on_stop)
        layout.addWidget(self._btn_stop)

        layout.addStretch()

        # 状態インジケーター
        self._indicator = StatusIndicator()
        layout.addWidget(self._indicator)
        self._lbl_state = QLabel("待機中")
        self._lbl_state.setMinimumWidth(80)
        layout.addWidget(self._lbl_state)

        return frame

    # ─── 統計情報 ─────────────────────────────────────────────────────────────

    def _build_stats_group(self) -> QGroupBox:
        grp = QGroupBox("統計情報")
        layout = QGridLayout(grp)

        # 数値ラベル群
        self._lbl_udp_pkts = self._stat_value("0")
        self._lbl_frames   = self._stat_value("0")
        self._lbl_saved    = self._stat_value("0")
        self._lbl_elapsed  = self._stat_value("00:00:00")
        self._lbl_remain   = self._stat_value("--:--:--")

        # 行 0: UDP パケット数 / パースされたフレーム数
        udp_label = QLabel("UDP パケット数:")
        udp_label.setToolTip(
            "受信した生 UDP パケットの総数。\n"
            "この値が増えていれば LiDAR からデータは届いています。\n"
            "「受信フレーム数」が 0 のままなら、パケット形式が\n"
            "本アプリの想定と異なります（ログで先頭バイトを確認してください）。"
        )
        layout.addWidget(udp_label,               0, 0)
        layout.addWidget(self._lbl_udp_pkts,      0, 1)
        layout.addWidget(QLabel("受信フレーム数:"), 0, 2)
        layout.addWidget(self._lbl_frames,         0, 3)

        # 行 1: 保存ファイル数 / 経過時間 / 残り時間
        layout.addWidget(QLabel("保存ファイル数:"), 1, 0)
        layout.addWidget(self._lbl_saved,          1, 1)
        layout.addWidget(QLabel("経過時間:"),       1, 2)
        layout.addWidget(self._lbl_elapsed,        1, 3)

        layout.addWidget(QLabel("残り時間:"),       2, 0)
        layout.addWidget(self._lbl_remain,         2, 1)

        # 診断ラベル (通常は非表示)
        self._lbl_diag = QLabel("")
        self._lbl_diag.setStyleSheet("color: #E65100; font-weight: bold;")
        self._lbl_diag.setWordWrap(True)
        layout.addWidget(self._lbl_diag, 3, 0, 1, 4)

        # プログレスバー
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        layout.addWidget(self._progress, 4, 0, 1, 4)

        return grp

    # ─── ログ ─────────────────────────────────────────────────────────────────

    def _build_log_group(self) -> QGroupBox:
        grp = QGroupBox("ログ")
        layout = QVBoxLayout(grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        self._log.setFont(QFont("Monospace", 9))
        layout.addWidget(self._log)
        return grp

    # ── ヘルパー ──────────────────────────────────────────────────────────────

    @staticmethod
    def _stat_value(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("", 12, QFont.Bold))
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return lbl

    def _log_message(self, msg: str) -> None:
        self._log.append(msg)
        # 行数制限
        doc = self._log.document()
        while doc.blockCount() > MAX_LOG_LINES:
            cursor = self._log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    # ── シグナル接続 ──────────────────────────────────────────────────────────

    def _connect_recorder_signals(self) -> None:
        r = self._recorder
        r.sig_state_changed.connect(self._on_state_changed)
        r.sig_raw_packet.connect(self._on_raw_packet)
        r.sig_frame_received.connect(self._on_frame_received)
        r.sig_file_saved.connect(self._on_file_saved)
        r.sig_elapsed.connect(self._on_elapsed)
        r.sig_error.connect(self._on_error)
        r.sig_finished.connect(self._on_finished)

    # ── スロット ──────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_start(self) -> None:
        host      = self._le_host.text().strip() or DEFAULT_HOST
        port      = self._sb_port.value()
        duration  = self._sb_duration.value()
        output    = self._le_output.text().strip() or DEFAULT_OUTPUT
        voxel     = self._dsb_voxel.value()

        self._recorder.start(
            host=host,
            port=port,
            duration_s=duration,
            output_dir=output,
            voxel_size=voxel,
        )

    @pyqtSlot()
    def _on_stop(self) -> None:
        self._recorder.stop()

    @pyqtSlot(str)
    def _on_state_changed(self, state: str) -> None:
        self._indicator.set_state(state)

        state_labels = {
            RecorderState.IDLE:       "待機中",
            RecorderState.CONNECTING: "接続中…",
            RecorderState.RECORDING:  "録画中",
            RecorderState.STOPPING:   "停止中…",
        }
        self._lbl_state.setText(state_labels.get(state, state))

        is_recording = (state == RecorderState.RECORDING)
        is_idle      = (state == RecorderState.IDLE)

        self._btn_start.setEnabled(is_idle)
        self._btn_stop.setEnabled(is_recording)
        self._le_host.setEnabled(is_idle)
        self._sb_port.setEnabled(is_idle)
        self._sb_duration.setEnabled(is_idle)
        self._le_output.setEnabled(is_idle)
        self._dsb_voxel.setEnabled(is_idle)

        if is_recording:
            self.statusBar().showMessage(
                f"録画中 → {self._recorder.output_dir}"
            )
        elif is_idle:
            self.statusBar().showMessage("待機中")

    @pyqtSlot(int)
    def _on_raw_packet(self, count: int) -> None:
        """生 UDP パケット受信数を更新し、フォーマット不一致の可能性を警告する"""
        self._lbl_udp_pkts.setText(f"{count:,}")

        # UDP パケットは届いているのに受信フレーム数が 0 → フォーマット不一致の疑い
        if count >= 20:
            frame_count_text = self._lbl_frames.text().replace(",", "")
            try:
                frame_count = int(frame_count_text)
            except ValueError:
                frame_count = 0
            if frame_count == 0:
                self._lbl_diag.setText(
                    f"⚠ UDP パケットは {count} 個届いていますが、フレームが1つも認識できていません。\n"
                    "パケット形式が本アプリの想定と異なる可能性があります。\n"
                    "コンソールの [診断] ログで「先頭24B」を確認してください。"
                )

    @pyqtSlot(int, int)
    def _on_frame_received(self, frame_id: int, n_points: int) -> None:
        current = self._lbl_frames.text()
        try:
            count = int(current.replace(",", "")) + 1
        except ValueError:
            count = 1
        self._lbl_frames.setText(f"{count:,}")
        # フレームが認識できたなら診断警告を消す
        self._lbl_diag.setText("")

    @pyqtSlot(str)
    def _on_file_saved(self, path: str) -> None:
        self._lbl_saved.setText(f"{self._recorder.saved_count:,}")
        filename = os.path.basename(path)
        self._log_message(f"[保存] {filename}")

    @pyqtSlot(int)
    def _on_elapsed(self, elapsed_s: int) -> None:
        duration = self._sb_duration.value()
        remain   = max(0, duration - elapsed_s)

        self._lbl_elapsed.setText(str(timedelta(seconds=elapsed_s)))
        self._lbl_remain.setText(str(timedelta(seconds=remain)))

        pct = int(elapsed_s / duration * 100) if duration > 0 else 0
        self._progress.setValue(min(pct, 100))

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._log_message(f"[エラー] {msg}")
        QMessageBox.critical(self, "エラー", msg)

    @pyqtSlot()
    def _on_finished(self) -> None:
        self._progress.setValue(100)
        saved = self._recorder.saved_count
        out   = self._recorder.output_dir
        self._log_message(f"[完了] {saved} ファイルを保存しました → {out}")
        self.statusBar().showMessage(f"録画完了 ({saved} ファイル保存)")

        # カウンターをリセット
        self._lbl_udp_pkts.setText("0")
        self._lbl_frames.setText("0")
        self._lbl_diag.setText("")
        self._progress.setValue(0)
        self._lbl_elapsed.setText("00:00:00")
        self._lbl_remain.setText("--:--:--")

    @pyqtSlot()
    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "出力ディレクトリを選択", self._le_output.text()
        )
        if d:
            self._le_output.setText(d)

    @pyqtSlot(int)
    def _update_duration_hint(self, value: int) -> None:
        td = timedelta(seconds=value)
        self._lbl_duration_hint.setText(f"({td})")

    # ── ウィンドウ終了 ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._recorder.state != RecorderState.IDLE:
            reply = QMessageBox.question(
                self, "確認",
                "録画中です。終了しますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self._recorder.stop()
        event.accept()
