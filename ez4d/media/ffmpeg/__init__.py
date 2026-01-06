import ffmpeg

from pathlib import Path
from typing import Optional


def export_video_to_images(
    video_path : Path,
    img_root   : Path,
    tgt_fps    : Optional[int] = None,
    ext        : str = ".jpg",
) -> None:
    video_path = Path(video_path)
    assert video_path.exists(), f'{video_path} does not exist!'
    img_root = Path(img_root)
    img_root.mkdir(parents=True, exist_ok=True)
    stream = ffmpeg.input(video_path)
    if tgt_fps:
        stream = ffmpeg.filter(stream, 'fps', fps=tgt_fps, round='down')
    stream = ffmpeg.output(
            stream,
            (img_root / f'%08d{ext}').as_posix(),
            start_number = 0,
        )
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream, quiet=True)