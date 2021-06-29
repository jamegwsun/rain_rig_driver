import numpy as np
import operator
import cv2


# show generated grid in real time
def show_frame(frame: np.array, frame_name: str = None) -> None:
    if show_frame:
        cv2.imshow(frame_name, frame)
        cv2.waitKey(0)
    return


mag = 4  # magnification factor

x_max = 914 * mag  # x size including border
y_max = 914 * mag  # y size including border

line_width = 4 * mag  # white line width
line_pitch = 22 * mag  # pitch including white line and black squares
x_repeats = 40  # number of lines repeated in x
y_repeats = 40  # number of lines repeated in y
x_grid_size = x_repeats * line_pitch + line_width
y_grid_size = y_repeats * line_pitch + line_width

assert x_grid_size < x_max, "planned x grid larger than max x size"
assert y_grid_size < y_max, "planned y grid larger than max y size"

# generate black background for the grid
grid = np.zeros((y_grid_size, x_grid_size)).astype('uint8')

# setup white lines in x and y
x_line = np.ones((line_width, x_grid_size)).astype('uint8') * 255
y_line = np.ones((y_grid_size, line_width)).astype('uint8') * 255

# populate white lines in x
for i in range(y_repeats + 1):
    line_index = i * line_pitch
    grid[line_index:line_index+line_width, :] = x_line

# populate white lines in y
for i in range(x_repeats + 1):
    line_index = i * line_pitch
    grid[:, line_index:line_index+line_width] = y_line

# generate border
full_grid = np.zeros((y_max, x_max)).astype('uint8')
x_diff, y_diff = tuple(map(operator.sub, full_grid.shape, grid.shape))
x_border = int(x_diff/2) + 1
y_border = int(y_diff/2) + 1
full_grid[y_border:y_border+y_grid_size, x_border:x_border+x_grid_size] = grid

# black space percentage
print("{:.2f}% black space".format((line_pitch-line_width)**2/(line_pitch**2)*100))
show_frame(full_grid)

