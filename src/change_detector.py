"""
変化点検出モジュール

前フレームと現フレームの点群を比較し、「変化した」ポイントのみを抽出する。

アルゴリズム:
  1. 両フレームをボクセルグリッドで離散化する。
  2. 前フレームに存在しないボクセルに属する点 = 新規点 (追加)
  3. 前フレームに存在するが現フレームにないボクセルに属する点 = 消失点 (削除)
  4. 新規点 + 消失点をまとめて「変化点」とする。

距離閾値 (voxel_size) は UI から変更可能。
"""

import numpy as np
from typing import Optional, Tuple


def _points_to_voxel_keys(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """(N,3+) 点群をボクセルキー (N,3) int32 配列に変換する"""
    xyz = points[:, :3]
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    return keys


def _voxel_set(keys: np.ndarray) -> set:
    """numpy (N,3) 行列を Pythonset[tuple] に変換 (高速比較用)"""
    return {tuple(k) for k in keys}


class ChangeDetector:
    """
    連続するフレーム間の変化点を検出する。

    Parameters
    ----------
    voxel_size : float
        ボクセルサイズ [m]。この値が小さいほど検出感度が高い。
        デフォルト 0.05 m (5 cm)。
    """

    def __init__(self, voxel_size: float = 0.05) -> None:
        self.voxel_size: float = voxel_size
        self._prev_points: Optional[np.ndarray] = None   # 前フレームの生点群
        self._prev_keys:   Optional[np.ndarray] = None   # 前フレームのボクセルキー

    # ── パブリック API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """状態をリセット (録画開始時に呼ぶ)"""
        self._prev_points = None
        self._prev_keys   = None

    def detect(
        self,
        current_points: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        現フレームの点群 (N,4) を受け取り、変化した点群を返す。

        Returns
        -------
        changed : np.ndarray, shape (M, 4)
            変化した点 (新規 + 消失)。初回フレームは全点を返す。
        is_first : bool
            初回フレームなら True。
        """
        cur_keys = _points_to_voxel_keys(current_points, self.voxel_size)

        if self._prev_points is None:
            # 初回フレーム → 全点を「変化」として扱う
            self._prev_points = current_points.copy()
            self._prev_keys   = cur_keys
            return current_points.copy(), True

        prev_set = _voxel_set(self._prev_keys)
        cur_set  = _voxel_set(cur_keys)

        # 新規ボクセル (現フレームにあって前フレームにない)
        new_voxels = cur_set - prev_set
        # 消失ボクセル (前フレームにあって現フレームにない)
        gone_voxels = prev_set - cur_set

        # 新規点を抽出
        if new_voxels:
            new_voxel_arr = np.array(list(new_voxels), dtype=np.int32)
            mask_new = self._membership_mask(cur_keys, new_voxel_arr)
            new_pts = current_points[mask_new]
        else:
            new_pts = np.empty((0, 4), dtype=np.float32)

        # 消失点を抽出 (前フレームの点群から)
        if gone_voxels:
            gone_voxel_arr = np.array(list(gone_voxels), dtype=np.int32)
            mask_gone = self._membership_mask(self._prev_keys, gone_voxel_arr)
            gone_pts = self._prev_points[mask_gone]
        else:
            gone_pts = np.empty((0, 4), dtype=np.float32)

        changed = np.concatenate([new_pts, gone_pts], axis=0) if (len(new_pts) or len(gone_pts)) \
                  else np.empty((0, 4), dtype=np.float32)

        # 状態更新
        self._prev_points = current_points.copy()
        self._prev_keys   = cur_keys

        return changed, False

    # ── 内部メソッド ─────────────────────────────────────────────────────────

    @staticmethod
    def _membership_mask(
        keys: np.ndarray,       # (N, 3) int32
        target_keys: np.ndarray # (M, 3) int32
    ) -> np.ndarray:
        """
        keys[i] が target_keys に含まれるかを示す boolean マスクを返す。
        小〜中規模データ向けの実装。
        """
        # 構造化配列でセット検索するのが最速だが、
        # 実用的な点数 (<100 万) では辞書方式で十分。
        target_set = {(int(r[0]), int(r[1]), int(r[2])) for r in target_keys}
        mask = np.array(
            [(int(k[0]), int(k[1]), int(k[2])) in target_set for k in keys],
            dtype=bool,
        )
        return mask
