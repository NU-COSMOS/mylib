import cv2
from pathlib import Path


def save_all_frames(video_path: str | Path,
                    dir_path: str | Path,
                    ext: str='png'):
    """
    指定した動画の全フレームをディレクトリに格納する

    Args:
        video_path (str | Path): 動画のパス
        dir_path (str | Path): フレームの格納先ディレクトリパス
        ext (str): 拡張子
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return

    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True)

    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(dir_path / f'{n:.07f}.{ext}', frame)
            n += 1
        else:
            return