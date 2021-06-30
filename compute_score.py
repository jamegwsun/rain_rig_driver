import numpy as np
import cv2
import os
import argparse

from typing import List, Optional


_ALLOWED_EXT = ['.avi']  # only accepts .avi files for now

_FRAMES_PER_SEC = 1  # Take X frames from each second of footage
_BASELINE_FRAME_COUNT = 5  # Merge up to X frames of baseline footage
_COMP_FRAME_COUNT = 10  # Compare up to X frames of comparison footage
_LOWER_GRAY_BOUNDS = -255
_UPPER_GRAY_BOUNDS = -60
_SHOW_FRAME = True
_X_CROP_LIMITS = [300, 1600]  # cropped pixels in X
_Y_CROP_LIMITS = [0, 950]  # cropped pixels in Y
_WHITE_AREA_FRACTION = 33/100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ns", dest="no_save_files", required=False, default=False, action='store_true',
        help="Use this flag to avoid saving baseline, comparison, diff, and mask image files.")
    return parser.parse_args()


def process_frame(frame: np.array) -> np.array:
    # expect intake: BGR frame, convert to grayscale first
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # crop frame
    y_max, x_max = frame.shape
    y_limit = [_Y_CROP_LIMITS[0], _Y_CROP_LIMITS[1]] if _Y_CROP_LIMITS else [0, y_max]
    x_limit = [_X_CROP_LIMITS[0], _X_CROP_LIMITS[1]] if _X_CROP_LIMITS else [0, x_max]
    return frame[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]]


def average_baseline_frames(baseline_frames: list) -> np.array:
    return (sum([f.astype('int') for f in baseline_frames])/len(baseline_frames)).astype('uint8')


def get_grayscale_frames_from_file(frame_files: list, is_baseline: bool = False) -> List[np.array]:
    prompt_name = 'baseline' if is_baseline else 'comparison'
    # A rough implementation of prompting user input to pick from a list of files with a few edge cases checks
    while True:
        file_index = input('\nChoose {} file index: '.format(prompt_name))
        if int(file_index) in range(0, len(frame_files)):
            fname = frame_files[int(file_index)]
            assert any(fname.endswith(e) for e in _ALLOWED_EXT), 'Extension for file {} not allowed'.format(fname)
            print('{} file: {}'.format(prompt_name, fname))

            frames = []
            vc = cv2.VideoCapture(fname)
            frame_count = _BASELINE_FRAME_COUNT if is_baseline else _COMP_FRAME_COUNT
            for f in range(frame_count):
                vid_time = float(f / _FRAMES_PER_SEC)
                vc.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)
                success, frame = vc.read()
                # read and append frames until it can't anymore...
                if success:
                    frames.append(process_frame(frame))
                else:
                    break
            return frames, fname.split('.')[0]


def filter_grayscale(comp_frame: np.array, baseline_frame: np.array, comp_name: str,
                     save_images: bool, image_index: int):
    # abs_diff is only used as saved image, non-absolute diff is used to generate mask
    abs_diff = np.absolute(comp_frame.astype(int) - baseline_frame.astype(int)).astype('uint8')
    diff = comp_frame.astype(int) - baseline_frame.astype(int)
    # thresholding for white lines that turned dark due to water droplets
    mask = cv2.inRange(diff, _LOWER_GRAY_BOUNDS, _UPPER_GRAY_BOUNDS)

    # save comparison, diff, and mask images
    if save_images:
        cv2.imwrite("{}/{}_{}_comp.png".format(comp_name, comp_name, image_index), comp_frame)
        show_frame(comp_frame)

        cv2.imwrite("{}/{}_{}_diff.png".format(comp_name, comp_name, image_index), abs_diff)
        show_frame(abs_diff)

        cv2.imwrite("{}/{}_{}_mask.png".format(comp_name, comp_name, image_index), mask)
        show_frame(mask)


    # since thresholding produces either values of 0 or 255
    # divide array sum by 255 to get the number of white pixels in mask
    return int(np.sum(mask) / 255)


def show_frame(frame: np.array, frame_name: str = None, show_frame: bool = _SHOW_FRAME) -> None:
    if show_frame:
        cv2.imshow(frame_name, frame)
        cv2.waitKey(0)
    return


if __name__ == "__main__":
    args = parse_args()

    # list all files in current directory
    files = sorted([f for f in os.listdir() if any(f.endswith(e) for e in _ALLOWED_EXT)])
    print('\n{0:10}{1}\n-------------------\n'.format('Index', 'File Name'))
    for i, f in enumerate(files):
        print('{0:4} ---  {1}'.format(str(i), f))
    print('\n')

    # get baseline and comparison grayscale frames from file via user prompt
    baseline_frames, baseline_name = get_grayscale_frames_from_file(frame_files=files, is_baseline=True)
    comp_frames, comp_name = get_grayscale_frames_from_file(frame_files=files)

    assert baseline_frames[0].shape == comp_frames[0].shape, 'Image sizes are not equal'
    y_size, x_size = baseline_frames[0].shape
    total_pixels = y_size * x_size

    # average all grayscale baseline frames into one grayscale frame
    baseline_frame = average_baseline_frames(baseline_frames=baseline_frames)

    # save baseline image
    if not args.no_save_files:
        if not os.path.exists(comp_name):
            os.makedirs(comp_name)
        cv2.imwrite("{}/{}_baseline.png".format(comp_name, comp_name), baseline_frame)

    frame_data = []
    for frame_index, comp_frame in enumerate(comp_frames):
        print('Processing frame {} of {}'.format((frame_index + 1), len(comp_frames)))
        # append the number of 1s in the mask
        frame_data.append(filter_grayscale(comp_frame=comp_frame,
                                           baseline_frame=baseline_frame,
                                           comp_name=comp_name,
                                           save_images=not(args.no_save_files),
                                           image_index=frame_index))

    print("\n" + "-" * 40)
    print("{}:\n".format(comp_name))
    print("Raw data: {}".format(frame_data))
    print("Mean: {:.1f}".format(np.average(frame_data)))
    print("Std. Dev.: {:.1f}".format(np.std(frame_data)))

    area_corrected_mean = np.average(frame_data) / _WHITE_AREA_FRACTION
    print("\nArea corrected mean: {:.1f}".format(area_corrected_mean))
    print("Area corrected droplet coverage: {:.2f}%".format(area_corrected_mean / total_pixels * 100))
    print("-" * 40)
