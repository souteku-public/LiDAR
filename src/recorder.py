"""
録画コントローラー

UDP 受信 → 変化点検出 → PCD 保存 のパイプラインを管理する。
Qt の QObject を継承しシグナルを通じて UI を更新する。
"""

import os
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QMetaObject, Qt

from .udp_receiver import UDPReceiver
from .packet_parser import PointCloudFrame
from .change_detector import ChangeDetector
from .pcd_writer import PCDWriter

logger = logging.getLogger(__name__)


class RecorderState:
    IDLE       = "idle"
    CONNECTING = "connecting"
    RECORDING  = "recording"
    STOPPING   = "stopping"


class Recorder(QObject):
    """
    録画セッション全体を管理するコントローラー。

    Signals
    -------
    sig_state_changed(str)      : 状態変化
    sig_frame_received(int, int): (フレーム番号, 点数)
    sig_file_saved(str)         : 保存したファイルパス
    sig_elapsed(int)            : 経過秒数
    sig_error(str)              : エラーメッセージ
    sig_finished()              : 録画完了 (時間切れ or 手動停止)
    """

    sig_state_changed  = pyqtSignal(str)
    sig_frame_received = pyqtSignal(int, int)   # frame_id, n_points
    sig_raw_packet     = pyqtSignal(int)         # 生UDPパケット通算数
    sig_file_saved     = pyqtSignal(str)
    sig_elapsed        = pyqtSignal(int)
    sig_error          = pyqtSignal(str)
    sig_finished       = pyqtSignal()

    MAX_DURATION_S = 6 * 3600  # 6 時間

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._state = RecorderState.IDLE
        self._receiver  = UDPReceiver(
            on_frame=self._on_frame,
            on_error=self._on_receiver_error,
            on_raw_packet=self._on_raw_packet,
        )
        self._detector  = ChangeDetector()
        self._writer:   Optional[PCDWriter] = None
        self._output_dir: str = ""

        # 録画パラメータ
        self._duration_s: int = 0
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._saved_count:  int = 0

        # タイマースレッド
        self._timer_thread: Optional[threading.Thread] = None
        self._stop_ev = threading.Event()

    # ── パブリック API ────────────────────────────────────────────────────────

    def start(
        self,
        host: str,
        port: int,
        duration_s: int,
        output_dir: str,
        voxel_size: float = 0.05,
    ) -> None:
        """録画を開始する"""
        if self._state != RecorderState.IDLE:
            return

        duration_s = max(1, min(duration_s, self.MAX_DURATION_S))

        # セッションごとにサブディレクトリを作成
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        session_dir  = os.path.join(output_dir, session_name)

        self._duration_s  = duration_s
        self._output_dir  = session_dir
        self._frame_count = 0
        self._saved_count = 0
        self._stop_ev.clear()

        self._detector = ChangeDetector(voxel_size=voxel_size)
        self._writer   = PCDWriter(session_dir)

        self._set_state(RecorderState.CONNECTING)
        self._receiver.start(host, port)
        self._start_time = time.time()
        self._set_state(RecorderState.RECORDING)

        # タイマースレッド起動
        self._timer_thread = threading.Thread(
            target=self._timer_run,
            daemon=True,
            name="RecorderTimer",
        )
        self._timer_thread.start()

    @pyqtSlot()
    def stop(self) -> None:
        """録画を停止する (メインスレッドから呼ぶこと)"""
        if self._state == RecorderState.IDLE:
            return
        self._set_state(RecorderState.STOPPING)
        self._stop_ev.set()
        self._receiver.stop()
        self._set_state(RecorderState.IDLE)
        self.sig_finished.emit()

    @property
    def state(self) -> str:
        return self._state

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def saved_count(self) -> int:
        return self._saved_count

    # ── 内部メソッド ─────────────────────────────────────────────────────────

    def _set_state(self, state: str) -> None:
        self._state = state
        self.sig_state_changed.emit(state)

    def _on_frame(self, frame: PointCloudFrame) -> None:
        """受信スレッドから呼ばれる (非 Qt スレッド)"""
        if self._state != RecorderState.RECORDING:
            return

        self._frame_count += 1
        changed, is_first = self._detector.detect(frame.points)

        # シグナルは Qt が自動でキューイング (スレッドセーフ)
        self.sig_frame_received.emit(frame.frame_id, len(frame.points))

        if changed.shape[0] > 0 and self._writer is not None:
            path = self._writer.write(frame.frame_id, frame.timestamp, changed)
            if path:
                self._saved_count += 1
                self.sig_file_saved.emit(path)

    def _on_raw_packet(self, count: int, _head: bytes) -> None:
        """生 UDP パケット受信コールバック (受信スレッドから呼ばれる)"""
        self.sig_raw_packet.emit(count)

    def _on_receiver_error(self, msg: str) -> None:
        """受信スレッドからのエラー通知 (受信スレッドから呼ばれる)"""
        self.sig_error.emit(msg)
        # 受信スレッドから直接 stop() を呼ぶと self._thread.join() がデッドロックするため
        # Qt のイベントキューに乗せてメインスレッドで実行させる
        QMetaObject.invokeMethod(self, "stop", Qt.QueuedConnection)

    def _timer_run(self) -> None:
        """経過時間を監視し、時間切れで録画を停止するスレッド"""
        while not self._stop_ev.wait(1.0):
            elapsed = int(time.time() - self._start_time)
            self.sig_elapsed.emit(elapsed)
            if elapsed >= self._duration_s:
                logger.info("録画時間 %d 秒に達したため停止します", self._duration_s)
                # タイマースレッドからも QueuedConnection でメインスレッドに委譲
                QMetaObject.invokeMethod(self, "stop", Qt.QueuedConnection)
                return
