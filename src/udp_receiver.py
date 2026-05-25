"""
UDP 受信スレッド

別スレッドで UDP ソケットを開き、パケットを受け取るたびに
コールバック (on_frame) を呼び出す。

on_frame(frame: PointCloudFrame) はメインスレッドではなく
受信スレッドから呼ばれるため、Qt シグナルに橋渡しすること。
"""

import socket
import threading
import time
import logging
from typing import Callable, Optional

from .packet_parser import FalconK2Parser, PointCloudFrame

logger = logging.getLogger(__name__)

RECV_BUFFER_SIZE = 65535   # UDP 最大ペイロード
SOCKET_TIMEOUT   = 1.0     # [s]  ソケット受信タイムアウト (停止検知用)

# 診断ログ: 最初の N パケットのみ先頭バイトを出力する
_DIAG_DUMP_COUNT = 5


class UDPReceiver:
    """
    Falcon K2 UDP パケットを受信し、フレームが完成するたびに
    コールバックを呼ぶバックグラウンドスレッド。

    Parameters
    ----------
    on_frame : Callable[[PointCloudFrame], None]
        完全なフレームを受け取るコールバック。
    on_error : Callable[[str], None]
        エラー発生時のコールバック (エラーメッセージ文字列)。
    on_raw_packet : Callable[[int, bytes], None] | None
        生 UDP パケット受信時のコールバック (受信通算番号, 先頭バイト列)。
        UI の診断表示に使用。
    """

    def __init__(
        self,
        on_frame: Callable[[PointCloudFrame], None],
        on_error: Callable[[str], None],
        on_raw_packet: Optional[Callable[[int, bytes], None]] = None,
    ) -> None:
        self._on_frame      = on_frame
        self._on_error      = on_error
        self._on_raw_packet = on_raw_packet
        self._thread:  Optional[threading.Thread] = None
        self._stop_ev: threading.Event = threading.Event()
        self._sock:    Optional[socket.socket] = None

    # ── パブリック API ────────────────────────────────────────────────────────

    def start(self, host: str, port: int) -> None:
        """受信スレッドを開始する"""
        if self._thread and self._thread.is_alive():
            return

        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(host, port),
            daemon=True,
            name="UDPReceiver",
        )
        self._thread.start()
        logger.info("UDPReceiver started on %s:%d", host, port)

    def stop(self) -> None:
        """受信スレッドを停止する"""
        self._stop_ev.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        # 受信スレッド自身から呼ばれた場合は join しない (cannot join current thread 防止)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        logger.info("UDPReceiver stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── 内部実装 ──────────────────────────────────────────────────────────────

    def _run(self, host: str, port: int) -> None:
        parser = FalconK2Parser()
        raw_count = 0          # 生 UDP パケット受信通算数
        parsed_count = 0       # パース成功フレーム数

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(SOCKET_TIMEOUT)
            self._sock.bind((host, port))
            logger.info("UDP バインド成功: %s:%d", host, port)
        except OSError as exc:
            self._on_error(f"ソケットのバインドに失敗しました: {exc}")
            return

        while not self._stop_ev.is_set():
            try:
                data, addr = self._sock.recvfrom(RECV_BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                # ソケットが強制クローズされた場合
                break

            raw_count += 1

            # ── 診断ログ: 最初の数パケットは送信元と先頭バイトを記録 ──
            if raw_count <= _DIAG_DUMP_COUNT:
                head = data[:24].hex(" ") if len(data) >= 24 else data.hex(" ")
                logger.info(
                    "[診断] パケット#%d  送信元=%s  サイズ=%d bytes  先頭24B: %s",
                    raw_count, addr, len(data), head,
                )

            # コールバック通知 (UI カウンター更新用)
            if self._on_raw_packet is not None:
                try:
                    self._on_raw_packet(raw_count, data[:24])
                except Exception:  # noqa: BLE001
                    pass

            try:
                frame = parser.feed(data)
                if frame is not None:
                    parsed_count += 1
                    self._on_frame(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("パケット解析エラー: %s", exc)

        logger.info(
            "UDPReceiver 終了: 受信パケット=%d, パース済みフレーム=%d",
            raw_count, parsed_count,
        )
        if raw_count > 0 and parsed_count == 0:
            logger.warning(
                "【要確認】UDPパケットは %d 個届きましたがフレームが1つもパースできませんでした。\n"
                "  → Falcon K2 のパケットフォーマットが本実装の想定と異なる可能性があります。\n"
                "  → 上の [診断] ログの「先頭24B」をご確認ください。",
                raw_count,
            )

        try:
            self._sock.close()
        except OSError:
            pass
