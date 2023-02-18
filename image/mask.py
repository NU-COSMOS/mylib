import cv2
from pathlib import Path
from typing import Literal
import numpy as np


def crop(org: str | Path, mask: str | Path, reverse: bool=False,
         fill: Literal['white', 'black', 'clear'] = 'white') -> np.ndarray:
    """
    画像をマスクで切り抜く.
    切り抜きたい画像から、マスク画像の黒部分を切り抜く. 

    Args:
        org (str | Path): 切り抜きたい画像のパス
        mask (str | Path): 切り抜きに使いたいマスク画像
        reverse (bool): マスク画像の白黒を反転させるかどうか. Trueのとき反転させる
        fill (Literal['white', 'black', 'clear']): 切り抜き部分以外の塗りつぶし方法

    Returns:
        np.ndarray: 切り抜いた画像. cv2.imwrite()で保存可能
    """
    img = cv2.imread(str(org))
    gray_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

    if img.shape[1:] != gray_mask.shape:
        print(f'画像の縦・横サイズが違います. org: {img.shape}, mask: {gray_mask.shape}')
        exit(1)

    # マスク画像を白と黒に二値化する
    _, gray_mask = cv2.threshold(gray_mask, 128, 255, cv2.THRESH_OTSU)

    # マスク画像の白部分を切り抜く場合
    if reverse:
        gray_mask = cv2.bitwise_not(gray_mask)

    # 合成のために3チャネル化
    mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)

    # 切り抜き画像を生成
    blended = cv2.addWeighted(src1=img, alpha=1, src2=mask, beta=1, gamma=0)

    # 切り抜き部分以外は黒で塗りつぶす
    if fill == 'black':
        blended[gray_mask==255] = [0, 0, 0]

    # 切り抜き部分以外は透過
    elif fill == 'clear':
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2BGRA)
        blended[gray_mask==255, 3] = 0

    return blended