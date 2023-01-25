# -*- coding: utf-8 -*-
# created by makise, 2022/2/25

"""
apply occlusion to the input image
use interpolation to handle cases where the upper left corner of occlusion is not integer
"""

# import necessary packages
import numpy as np


# combine two type of occlusion
def occlusion(img, box, occlusion_size, occlusion_color):
    """
    Use interpolation occlusion when the box is integer since it now supports
    Use interpolation occlusion when the box is non-integer
    :param img: np array
    :param box: A 2-tuple which is treated as the upper left corner of occlusion
    :param occlusion_size: A 2-tuple which is treated as (height, width) of occlusion
    :param occlusion_color: A integer between 0 and 255 which is treated as the color of occlusion
    :return:
    """
    # check if the box is integer
    img_np = occlusion_with_interpolation(img, box, occlusion_size, occlusion_color)
    return img_np


# regular occlusion
# the occlusion has integer upper left corner and integer height and width
def regular_occlusion(img, box, occlusion_size, occlusion_color):
    """
    :param img: np array
    :param box: A 2-tuple which is treated as the upper left corner of occlusion
    :param occlusion_size: A 2-tuple which is treated as (height, width) of occlusion
    :param occlusion_color: A integer between 0 and 255 which is treated as the color of occlusion
    :return: Image after applying occlusion in np array
    """
    img_np = img
    print(img_np.shape)  # should be (height, width, 3)
    # get the height and width of the image
    height, width = img_np.shape[:2]
    # get the upper left corner of occlusion
    x, y = box
    # get the height and width of occlusion
    h, w = occlusion_size
    # get the upper left corner of occlusion in the image
    x_in_img, y_in_img = max(0, x), max(0, y)
    # get the lower right corner of occlusion in the image
    x_in_img_end, y_in_img_end = min(x + w, width), min(y + h, height)

    # apply occlusion
    img_np[y_in_img:y_in_img_end, x_in_img:x_in_img_end, :] = occlusion_color

    return img_np


# occlusion with interpolation
# the occlusion has non-integer upper left corner and integer height and width
def occlusion_with_interpolation(img, box, occlusion_size, occlusion_color):
    """
    :param img:     np array 32*32*3
    :param box:     A 2-tuple which is treated as the upper left corner of occlusion
    :param occlusion_size: A 2-tuple which is treated as (height, width) of occlusion
    :param occlusion_color: A integer between 0 and 255 which is treated as the color of occlusion
    :return:        Image after applying occlusion in nd array float32
    """
    # convert PIL Image to float numpy array
    img_np = img.astype(np.float32)
    # copy the image
    img_np_origin = img_np.copy()
    # get the height and width of the image
    height, width = img_np.shape[:2]
    # get the upper left corner of occlusion
    x, y = box
    # get the height and width of occlusion
    h, w = occlusion_size
    # get the upper left corner of occlusion in the image
    x_in_img, y_in_img = max(0, x), max(0, y)
    # get the lower right corner of occlusion in the image
    x_in_img_end, y_in_img_end = min(x + w, width), min(y + h, height)

    # apply occlusion
    # iterate through the occlusion area with while loop
    i = y_in_img
    j = x_in_img
    while i < y_in_img_end:
        while j < x_in_img_end:
            # the i, j is non-integer
            # the occlusion point affects four pixels around it
            # get four pixels that will be affected by this occlusion point
            # the four pixels are (floor(x), floor(y)), (floor(x), ceil(y)), (ceil(x), floor(y)), (ceil(x), ceil(y))
            x1 = int(np.floor(j))
            y1 = int(np.floor(i))
            x2 = int(np.ceil(j))
            y2 = int(np.ceil(i))
            # img_np has shape (height, width, 3)
            # get the four pixels' color
            pixel_x1_y1 = img_np_origin[y1, x1, :]
            pixel_x1_y2 = img_np_origin[y2, x1, :]
            pixel_x2_y1 = img_np_origin[y1, x2, :]
            pixel_x2_y2 = img_np_origin[y2, x2, :]

            # calculate the coefficients of interpolation on four pixels
            # by decompose the occlusion point color on x-axis and y-axis
            # if x1 == x2, then the coefficient of x-axis is 1
            # if y1 == y2, then the coefficient of y-axis is 1
            coefficient_x1 = 1 if x1 == x2 else (x2 - j) / (x2 - x1)
            coefficient_y1 = 1 if y1 == y2 else (y2 - i) / (y2 - y1)
            coefficient_x2 = 1 if x1 == x2 else (j - x1) / (x2 - x1)
            coefficient_y2 = 1 if y1 == y2 else (i - y1) / (y2 - y1)
            coefficient_x1_y1 = coefficient_x1 * coefficient_y1
            coefficient_x1_y2 = coefficient_x1 * coefficient_y2
            coefficient_x2_y1 = coefficient_x2 * coefficient_y1
            coefficient_x2_y2 = coefficient_x2 * coefficient_y2

            # if x1 != x2 and y1 != y2, assign both four points
            if x1 != x2 and y1 != y2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
                img_np[y1, x2] = img_np[y1, x2] - coefficient_x2_y1 * (pixel_x2_y1 - occlusion_color)
                img_np[y2, x1] = img_np[y2, x1] - coefficient_x1_y2 * (pixel_x1_y2 - occlusion_color)
                img_np[y2, x2] = img_np[y2, x2] - coefficient_x2_y2 * (pixel_x2_y2 - occlusion_color)
            elif x1 == x2 and y1 == y2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
            # if x1 == x2, only assign (y1, x1) and (y2, x1)
            elif x1 == x2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
                img_np[y2, x1] = img_np[y2, x1] - coefficient_x1_y2 * (pixel_x1_y2 - occlusion_color)
            # else if y1 == y2, only assign (y1, x1) and (y1, x2)
            elif y1 == y2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
                img_np[y1, x2] = img_np[y1, x2] - coefficient_x2_y1 * (pixel_x2_y1 - occlusion_color)
            # move to the next occlusion point
            j += 1
        # move to the next occlusion point
        i += 1
        j = x_in_img

    return img_np
