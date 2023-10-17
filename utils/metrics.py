import numpy as np
import matplotlib.pyplot as plt

def rectangular_graph(arr: np.ndarray, is_rate: bool=True):
    """
    帯グラフを作成する

    Args:
        arr (np.ndarray): 帯グラフにしたい配列
        is_rate (bool): 縦軸を実数ではなく割合表示にするか

    ex. [[15, 10, 20],
         [4, 20, 18],  
         [29, 10, 5],
         [18, 32, 11]]

    """

def seg_iou(y: np.ndarray[np.ndarray[int] | int],
            t: np.ndarray[np.ndarray[int] | int],
            label: int) -> float:
    """
    セグメンテーションタスクにおけるIoUを計算する
    
    Args:
        y (np.ndarray[np.ndarray[int] | int]): 推論結果の二次元or一次元配列.
        t (np.ndarray[np.ndarray[int] | int]): 正解ラベルの二次元or一次元配列.
        label (int): IoUを計算したいラベル.

    Returns:
        float: IoU
    """
    if y.shape != t.shape:
        raise ValueError('yとtのshapeは等しくしてください')
    
    if len(y.shape) > 2:
        raise ValueError('y及びtは二次元以下の配列にしてください')
    
    if len(y.shape) == 2:
        y = y.flatten()
        t = t.flatten()

    # ラベルが一致するピクセルの位置を取得
    intersection = np.logical_and(y == label, t == label)
    
    # ラベルが一致するピクセルの合計数を計算
    intersection = np.sum(intersection)
    
    # 予測と正解のラベルが一致するピクセルの位置を取得
    union = np.logical_or(y == label, t == label)
    
    # 予測と正解のラベルが一致するピクセルの合計数を計算
    union = np.sum(union)
    
    # IoUを計算
    iou = intersection / union if union > 0 else 0.0

    return iou


def mean_seg_iou(y: np.ndarray[np.ndarray[int] | int],
                 t: np.ndarray[np.ndarray[int] | int],
                 labels: list[int]=None) -> float:
    """
    セグメンテーションタスクにおけるmIoUを計算する.

    Args:
        y (np.ndarray[np.ndarray[int] | int]): 推論結果の二次元or一次元配列.
        t (np.ndarray[np.ndarray[int] | int]): 正解ラベルの二次元or一次元配列. 
        labels (list[int]): 計算するクラスラベル.

    Returns:
        float: mIoU

    ex. 
        [[0, 0, 1],     [[0, 1, 1],
    y =  [1, 0, 2],  t = [1, 2, 2],  labels = [0, 1, 2]  return (1/4 + 2/3 + 3/5) / 3
         [0, 2, 2]]      [2, 2, 2]]
    """
    if y.shape != t.shape:
        raise ValueError('yとtのshapeは等しくしてください')
    
    if len(y.shape) > 2:
        raise ValueError('y及びtは二次元以下の配列にしてください')
    
    if len(y.shape) == 2:
        y = y.flatten()
        t = t.flatten()
    
    if labels is None:
        labels = list(set(np.concatenate([y, t])))

    # クラスラベルごとのIoUを計算して記録
    ious = []
    for label in labels:
        ious.append(seg_iou(y, t, label))

    return np.mean(ious)