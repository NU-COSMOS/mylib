import cv2
from pathlib import Path, PosixPath
from typing import Literal
import numpy as np
import math


def masking(src: str | PosixPath | cv2.Mat | np.ndarray,
            mask: str | PosixPath | cv2.Mat | np.ndarray,
            reverse: bool=False,
            fill: Literal['white', 'black', 'clear'] = 'black',
            threshold: int=128) -> np.ndarray:
    """
    画像をマスクで切り抜く.
    defaultでは、 maskの黒い部分がsrcに置き換わる

    Args:
        src (str | PosixPath | cv2.Mat | np.ndarray): 切り抜きたい画像のパス
        mask (str | PosixPath | cv2.Mat | np.ndarray): 切り抜きに使いたいマスク画像
        reverse (bool): マスク画像の白黒を反転させるかどうか. Trueのとき反転させる
        fill (Literal['white', 'black', 'clear']): 切り抜き部分以外の塗りつぶし方法
        threshold (int): 白黒に二値化するピクセル値の閾値

    Returns:
        np.ndarray: 切り抜いた画像. cv2.imwrite()で保存可能
    """
    # 画像とマスクの読み込み
    if type(src) == str or type(src) == PosixPath:
        src = cv2.imread(str(src))
    if type(mask) == str or type(mask) == PosixPath:
        mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

    src, mask = src.astype(np.uint8), mask.astype(np.uint8)

    if src.shape[:2] != mask.shape:
        print(f'画像の縦・横サイズが違います. org: {src.shape}, mask: {mask.shape}')
        exit(1)

    # マスク画像を白と黒に二値化する
    _, gray_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    # マスク画像の白部分を切り抜く場合
    if reverse:
        gray_mask = 255 - gray_mask

    # 合成のために3チャネル化
    mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)

    # 切り抜き画像を生成
    blended = cv2.addWeighted(src1=src, alpha=1, src2=mask, beta=1, gamma=0)

    # 切り抜き部分以外は黒で塗りつぶす
    if fill == 'black':
        blended[gray_mask==255] = [0, 0, 0]

    # 切り抜き部分以外は透過
    elif fill == 'clear':
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2BGRA)
        blended[gray_mask==255, 3] = 0

    return blended


def keystone_correction(src: str | Path,
                        p1: list | np.ndarray,
                        p2: list | np.ndarray,
                        p3: list | np.ndarray,
                        p4: list | np.ndarray,
                        w_ratio: float=1.0) -> cv2.Mat:
    """
    画像の一部を台形補正する

    Args:
        src (str | Path): 元画像のパス
        p1 (list | np.ndarray): 台形補正エリアの左上座標
        p2 (list | np.ndarray): 台形補正エリアの右上座標
        p3 (list | np.ndarray): 台形補正エリアの左下座標
        p4 (list | np.ndarray): 台形補正エリアの右下座標
        w_ratio (float): 台形補正時の比率調整を行うパラメータ

    Returns:
        cv2.Mat: 台形補正した画像. cv2.imwrite()で保存可能
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)
    
    # 入力画像の読み込み
    img = cv2.imread(str(src))
    
    #　幅取得
    o_width = np.linalg.norm(p2 - p1)
    o_width = math.floor(o_width * w_ratio)
    
    #　高さ取得
    o_height = np.linalg.norm(p3 - p1)
    o_height = math.floor(o_height)
    
    # 変換前の4点
    src = np.float32([p1, p2, p3, p4])
    
    # 変換後の4点
    dst = np.float32([[0, 0],[o_width, 0],[0, o_height],[o_width, o_height]])
    
    # 変換行列
    M = cv2.getPerspectiveTransform(src, dst)
    
    # 射影変換・透視変換する
    output = cv2.warpPerspective(img, M,(o_width, o_height))

    return output


def label_to_color_img(label_map: np.ndarray,
                       translate: dict[int, np.ndarray[int]]) -> np.ndarray[np.ndarray[np.ndarray[np.uint8]]]:
    """
    ラベルマップをカラー画像に変換する
    cv2.imwrite(hoge, label_to_color_img(fuga))で出力可能

    Args:
        label (np.ndarray): ラベルマップ
        translate (dict[int, np.ndarray[int]]): どのラベルを何色に変換するか

    Returns:
        np.ndarray[np.ndarray[np.ndarray[np.uint8]]]: カラー画像

    ex. 
           [[0, 1, 2],
    label = [1, 2, 2],  translate = {0: [0, 0, 255], 1: [255, 255, 255], 2: [0, 255, 0]}
            [1, 0, 0]]

           [[赤, 白, 緑]
    return  [白, 緑, 緑]
            [白, 赤, 赤]]
    """
    if len(label_map.shape) != 2:
        raise ValueError('ラベルマップは二次元配列にしてください')
    
    color_img = np.zeros((label_map.shape[0], label_map.shape[1], 3))

    # ラベルマップに存在するラベル一覧を取得
    labels = list(set(label_map.flatten()))

    # 各ラベルに対応する色でカラー画像を埋めていく
    for label in labels:
        color_img[label_map == label] = translate[label]

    return color_img.astype(np.uint8)
