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
_VID_LENGTH_S = 10  # Compare up to x seconds of footage


def get_average(values: list) -> float:
    return sum(values) / len(values)


def get_baseline_image(files: list) -> np.array:
    while True:
        bl_index = input('\nChoose baseline file index: ')
        if int(bl_index) in range(0, len(files)):
            fname = files[int(bl_index)]
            if not fname.endswith('jpg'):
                print('Chosen file is not a jpg')
            else:
                print('Baseline file: {}'.format(fname))
                return cv2.imread(fname).astype('int')


def get_comparison_image(files: list) -> List[np.array]:
    while True:
        comp_index = input('\nChoose comparison file index: ')
        if int(comp_index) in range(0, len(files)):
            fname = files[int(comp_index)]
            assert any(fname.endswith(e) for e in _ALLOWED_EXT), 'Extension for file {} not allowed'.format(fname)
            print('Comparison file: {}'.format(fname))
            if fname.endswith('.jpg'):
                return [cv2.imread(fname).astype('int')]
            else:  # if reading a video
                ims = []
                vidcap = cv2.VideoCapture(fname)
                for t in range(_VID_LENGTH_S * _FRAME_RATE):
                    vid_time = float(t / _FRAME_RATE)
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)
                    success, im = vidcap.read()
                    if success:
                        ims.append(im)
                    else:
                        break
                return ims


if __name__ == "__main__":
    files = sorted([f for f in os.listdir() if os.path.isfile(f)])
    print('\n{0:10}{1}\n-------------------\n'.format('Index', 'File Name'))
    for i, f in enumerate(files):
        print('{0:4} ---  {1}'.format(str(i), f))

    im_bl: np.array = get_baseline_image(files=files)
    im_c: List[np.array] = get_comparison_image(files=files)

assert im_bl.shape == im_c[0].shape, 'Image sizes are not equal'
y_size, x_size, _ = im_bl.shape

frame_data = []

im_bl = np.sum(im_bl, axis=2) / 3

im_diffs = np.array([])

for im in im_c:
    _im = np.sum(im, axis=2) / 3
    im_diff = (_im - im_bl).flatten()
    im_diffs = np.concatenate((im_diffs, im_diff), axis=None)

    frame_data.append(FrameData(
        rms_error=np.sqrt(np.sum(im_diff ** 2) / (x_size * y_size)),
        true_mean=np.sum(im_diff) / (x_size * y_size),
        std_dev=np.std(im_diff)
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

print('\nAverage frame data:')
print('rms error: {}'.format(get_average(rms_errors)))
print('true mean: {}'.format(get_average(true_means)))
print('std dev: {}'.format(get_average(std_devs)))

plt.hist(im_diffs, bins=100)
plt.show()
