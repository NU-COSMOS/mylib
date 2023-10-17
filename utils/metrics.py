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