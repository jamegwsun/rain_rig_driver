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
_C_FRAMES = 10  # Compare up to X frames of comparison footage
_RGB_RATIO = [10, 1, 1]  # RED, GREEN, BLUE


def get_RGB_average(im: np.array) -> np.array:
    im = im.astype(int)
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


if __name__ == "__main__":
    files = sorted([f for f in os.listdir() if os.path.isfile(f)])
    print('\n{0:10}{1}\n-------------------\n'.format('Index', 'File Name'))
    for i, f in enumerate(files):
        print('{0:4} ---  {1}'.format(str(i), f))

    im_bl: List[np.array] = get_images_from_file(imgfiles=files, is_baseline=True)
    im_c: List[np.array] = get_images_from_file(imgfiles=files)

assert im_bl[0].shape == im_c[0].shape, 'Image sizes are not equal'
y_size, x_size, _ = im_bl[0].shape

frame_data = []
print('')

for fi, im in enumerate(im_c):
    print('Processing frame {} of {}'.format((fi + 1), len(im_c)))
    im_diffs = np.array([])
    for b in im_bl:
        _im = get_RGB_average(im)
        _b = get_RGB_average(b)
        # im_diffs is a 1D numpy array
        im_diffs = np.concatenate((im_diffs, (_im - _b).flatten()), axis=None)
    frame_data.append(FrameData(
        rms_error=np.sqrt(np.sum(im_diffs ** 2) / len(im_diffs)),
        true_mean=np.sum(im_diffs) / len(im_diffs),
        std_dev=np.std(im_diffs)
    ))

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

# plt.hist(im_diffs, bins=100)
# plt.show()
