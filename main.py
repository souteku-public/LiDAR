#!/usr/bin/env python3
"""
Falcon K2 LiDAR Recorder
=========================
Benewake Falcon K2 LiDAR から UDP で点群を受信し、
前フレームとの差分のみを PCD ファイルとして保存するアプリケーション。

使い方:
    python main.py

依存ライブラリのインストール:
    pip install -r requirements.txt
"""

import sys
import logging
import os

# src パッケージを import できるようにする
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore    import Qt

from src.main_window import MainWindow


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    _setup_logging()

    # High-DPI 対応
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("Falcon K2 LiDAR Recorder")
    app.setOrganizationName("LiDAR Tools")

    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
