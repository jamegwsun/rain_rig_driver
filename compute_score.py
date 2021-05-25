import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from typing import List, NamedTuple

_ALLOWED_EXT = ['.jpg', '.avi']

FrameData = NamedTuple("FrameData", [
    ('rms_error', float),
    ('true_mean', float),
    ('std_dev', float)
])

_FRAME_RATE = 2  # Frames per second
_BL_FRAMES = 1  # Compare up to X frames of baseline
_C_FRAMES = 3  # Compare up to X frames of comparison footage
_RGB_RATIO = [0.114, 0.587, 0.299]  # Blue, Green, Red
_LOWER_RGB_BOUNDS = np.array([100, 0, 0])
_UPPER_RGB_BOUNDS = np.array([255, 80, 80])
_LOWER_GRAY_BOUNDS = 20
_UPPER_GRAY_BOUNDS = 255
_SHOW_FRAME = True


def convert_RGB(im: np.array) -> np.array:
    im = im.astype(float)
    ratios = _RGB_RATIO
    for i, c in enumerate(ratios):
        im[:, :, i] *= c
    return np.sum(im, axis=2) / sum(ratios)


def get_num_average(values: list) -> float:
    return sum(values) / len(values)


def get_images_from_file(imgfiles: list, is_baseline: bool = False) -> List[np.array]:
    prompt_name = 'baseline' if is_baseline else 'comparison'
    while True:
        f_index = input('\nChoose {} file index: '.format(prompt_name))
        if int(f_index) in range(0, len(imgfiles)):
            fname = imgfiles[int(f_index)]
            assert any(fname.endswith(e) for e in _ALLOWED_EXT), 'Extension for file {} not allowed'.format(fname)
            print('{} file: {}'.format(prompt_name, fname))
            if fname.endswith('.jpg'):
                return [cv2.imread(fname).astype('int')]
            else:  # if reading a video
                imgs = []
                vidcap = cv2.VideoCapture(fname)
                if is_baseline:
                    total_frames = _BL_FRAMES
                else:
                    total_frames = _C_FRAMES
                for f in range(total_frames):
                    vid_time = float(f / _FRAME_RATE)
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)
                    success, img = vidcap.read()
                    if success:
                        imgs.append(img)
                    else:
                        break
                return imgs


def filter_RGB(c_frame: np.array, bl_frame: np.array = None):
    if bl_frame:
        c_frame = c_frame - bl_frame
    show_frame(c_frame)
    mask = cv2.inRange(c_frame, _LOWER_RGB_BOUNDS, _UPPER_RGB_BOUNDS)
    res = cv2.bitwise_and(c_frame, c_frame, mask=mask)
    show_frame(mask)
    return


def filter_Greyscale(c_frame: np.array, bl_frame: np.array):
    c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
    bl_frame = cv2.cvtColor(bl_frame, cv2.COLOR_BGR2GRAY)
    show_frame(c_frame)
    diff = np.absolute(c_frame.astype(int) - bl_frame.astype(int))
    mask = cv2.inRange(diff, _LOWER_GRAY_BOUNDS, _UPPER_GRAY_BOUNDS)
    res = cv2.bitwise_and(c_frame, c_frame, mask=mask)
    show_frame(mask)
    return


def show_frame(frame: np.array, frame_name: str = None, show_frame: bool = _SHOW_FRAME) -> None:
    if show_frame:
        cv2.imshow(frame_name, frame)
        cv2.waitKey(0)
    return


if __name__ == "__main__":
    files = sorted([f for f in os.listdir() if os.path.isfile(f)])
    print('\n{0:10}{1}\n-------------------\n'.format('Index', 'File Name'))
    for i, f in enumerate(files):
        print('{0:4} ---  {1}'.format(str(i), f))

    bl_frames: List[np.array] = get_images_from_file(imgfiles=files, is_baseline=True)
    c_frames: List[np.array] = get_images_from_file(imgfiles=files)

assert bl_frames[0].shape == c_frames[0].shape, 'Image sizes are not equal'
y_size, x_size, _ = bl_frames[0].shape

frame_data = []
print('')

for fi, c_frame in enumerate(c_frames):
    print('Processing frame {} of {}'.format((fi + 1), len(c_frames)))
    im_diffs = np.array([])
    for bl_frame in bl_frames:
        _c = convert_RGB(c_frame)
        _bl = convert_RGB(bl_frame)
        # im_diffs is a 1D numpy array
        im_diffs = np.concatenate((im_diffs, (_c - _bl).flatten()), axis=None)
    frame_data.append(FrameData(
        rms_error=np.sqrt(np.sum(im_diffs ** 2) / len(im_diffs)),
        true_mean=np.sum(im_diffs) / len(im_diffs),
        std_dev=np.std(im_diffs)
    ))
    filter_Greyscale(c_frame, bl_frame)

rms_errors = []
true_means = []
std_devs = []

print('\nIndividual frame data:')
for f in frame_data:
    print(f)
    rms_errors.append(f.rms_error)
    true_means.append(f.true_mean)
    std_devs.append(f.std_dev)

print('\nFrame difference averaged across all frames:')
print('rms error: {:.2f}'.format(get_num_average(rms_errors)))
print('true mean: {:.2f}'.format(get_num_average(true_means)))
print('std dev: {:.2f}'.format(get_num_average(std_devs)))

plt.hist(im_diffs, bins=100)
plt.show()
