#-*- coding:utf-8 -*-
"""
アノテーションデータをいろいろ変換
"""
import numpy as np


def instances_to_semantic(masks: list[np.ndarray]) -> np.ndarray:
    """
    インスタンスマスクをセマンティックマスクに変換する
    ex. img_instance1, img_instance2, img_instance3, ... -> img_instance
        [[0, 0, 0, 0]  [[1, 1, 0, 0]  [[0, 0, 0, 0]         [[1, 1, 0, 0] 
         [0, 0, 0, 0]   [0, 0, 0, 0]   [0, 0, 0, 1]      ->  [0, 0, 0, 1] 
         [0, 1, 1, 0]   [0, 0, 0, 0]   [0, 0, 0, 1]          [0, 1, 1, 1] 
         [0, 1, 1, 0]]  [0, 0, 0, 0]]  [0, 0, 0, 0]]         [0, 1, 1, 0]]
    """
    return np.where(np.sum(masks, axis=0) > 0, 1, 0)