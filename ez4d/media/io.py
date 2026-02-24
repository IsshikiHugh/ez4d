import imageio
import numpy as np
import ffmpeg

from tqdm import tqdm
from typing import Union, List, Optional
from pathlib import Path
from glob import glob

from .edit import flex_resize_img, flex_resize_video


def load_img_meta(
    img_path : Union[str, Path],
):
    """ Read the image meta from the given path without opening image. """
    assert Path(img_path).exists(), f'Image not found: {img_path}'
    H, W = imageio.v3.improps(img_path).shape[:2]
    meta = {'w': W, 'h': H}
    return meta


def load_img(
    img_path : Union[str, Path],
    mode     : str = 'RGB',
):
    """ Read the image from the given path. """
    assert Path(img_path).exists(), f'Image not found: {img_path}'

    img = imageio.v3.imread(img_path, plugin='pillow', mode=mode)

    meta = {
            'w': img.shape[1],
            'h': img.shape[0],
        }
    return img, meta


def save_img(
    img          : np.ndarray,
    output_path  : Union[str, Path],
    resize_ratio : Union[float, None] = None,
    **kwargs,
):
    """ Save the image. """
    assert img.ndim in [2, 3], f'Invalid image shape: {img.shape}'  # 2 for grayscale, 3 for colorized

    if resize_ratio is not None:
        img = flex_resize_img(img, ratio=resize_ratio)

    imageio.v3.imwrite(output_path, img, **kwargs)

def load_video_meta(video_path: Union[str, Path]) -> dict:
    video_path = str(video_path)
    assert Path(video_path).exists(), f'Video not found: {video_path}'

    probe = ffmpeg.probe(str(video_path))
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)

    assert video_stream is not None, f"No video stream found in {video_path}"

    num, den = map(int, video_stream['r_frame_rate'].split('/'))
    fps = num / den if den != 0 else 0

    if 'nb_frames' in video_stream:
        L = int(video_stream['nb_frames'])
    else:
        L = int(float(video_stream['duration']) * fps)

    return {
        'fps'     : fps,
        'w'       : int(video_stream['width']),
        'h'       : int(video_stream['height']),
        'L'       : L,
        'sid'     : 0,
        'eid'     : L,
        'total_L' : L,
    }
def load_video(
    video_path : Union[str, Path],
    sid        : Optional[int] = None,
    eid        : Optional[int] = None,
    silent     : bool = False,
    use_decord : bool = True,
):
    """
    Read the video from the given path.
    These functions are only suggested for quick usage / small scale video processing.
    For larger scale purposes, please use `EZVideoReader` and `EZVideoWriter`.
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)

    assert video_path.exists(), f'Video not found: {video_path}'

    if video_path.is_dir():
        if not silent:
            print(f'Found {video_path} is a directory. It will be regarded as a image folder.')
        imgs_path = sorted(glob(str(video_path / '*')))
        total_L = len(imgs_path)
        sid = 0 if sid is None else max(sid, 0)
        eid = total_L if eid is None else min(eid, total_L)
        imgs_path = imgs_path[sid:eid]
        frames = []
        for img_path in tqdm(imgs_path, disable=silent):
            frames.append(imageio.imread(img_path))
        frames = np.stack(frames, axis=0) # (L, H, W, 3)
        fps = 30 # default fps
    else:
        if not silent:
            print(f'Found {video_path} is a file. It will be regarded as a video file.')

        if use_decord:
            # It's an efficient video loading tools.
            # decord: https://github.com/dmlc/decord
            try:
                import decord
                vr = decord.VideoReader(str(video_path))
                total_L = len(vr)
                sid = 0 if sid is None else max(sid, 0)
                eid = total_L if eid is None else min(eid, total_L)
                frame_indices = list(range(sid, eid))
                frames = vr.get_batch(frame_indices)
                # Convert to numpy if not already.
                if not isinstance(frames, np.ndarray):
                    frames = frames.asnumpy()
                fps = float(vr.get_avg_fps())
            except ImportError:
                raise ImportError(
                    "Decord is not installed. Please install it via `pip install decord`, "
                    "or set `use_decord=False` to use imageio backend."
                )
        else:
            reader = imageio.get_reader(video_path, format='FFMPEG')
            total_L = reader.count_frames()
            sid = 0 if sid is None else max(sid, 0)
            eid = total_L if eid is None else min(eid, total_L)
            reader.set_image_index(sid)

            frames = []
            for _ in tqdm(range(sid, eid), disable=silent):
                frames.append(reader.get_next_data())
            frames = np.stack(frames, axis=0)
            fps = reader.get_meta_data()['fps']
    meta = {
        'fps'     : fps,
        'w'       : frames.shape[2],
        'h'       : frames.shape[1],
        'L'       : frames.shape[0],
        'sid'     : sid,
        'eid'     : eid,
        'total_L' : total_L,
    }

    return frames, meta


def save_video(
    frames       : Union[np.ndarray, List[np.ndarray]],
    output_path  : Union[str, Path],
    fps          : float = 30,
    resize_ratio : Union[float, None] = None,
    quality      : Union[int, None]   = None,
    silent       : bool = False
):
    """ 
    Save the frames as a video. 
    These functions are only suggested for quick usage / small scale video processing.
    For larger scale purposes, please use `EZVideoReader` and `EZVideoWriter`."""
    if isinstance(frames, List):
        frames = np.stack(frames, axis=0)
    assert frames.ndim == 4, f'Invalid frames shape: {frames.shape}'

    if resize_ratio is None:
        resize_ratio = 1
    frames = flex_resize_video(frames, ratio=resize_ratio, kp_mod=16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(output_path, fps=fps, quality=quality)
    output_seq_name = str(output_path).split('/')[-1]
    for frame in tqdm(frames, desc=f'Saving {output_seq_name}', disable=silent):
        writer.append_data(frame)
    writer.close()