# -*- coding: utf-8 -*-
# created by makise, 2023/1/5
import numpy as np


def occlusion_v3(img, box, occlusion_size, occlusion_color):
    """
    :param img:     np array 32*32*3
    :param box:     A 2-tuple which is treated as the upper left corner of occlusion
    :param occlusion_size: A 2-tuple which is treated as (height, width) of occlusion
    :param occlusion_color: A integer between 0 and 255 which is treated as the color of occlusion
    :return:        Image after applying occlusion in nd array float32
    """
    img_np = img.astype(np.float32)
    img_np_origin = img_np.copy()
    height, width = img_np.shape[:2]
    x, y = box
    h, w = occlusion_size
    x_in_img, y_in_img = max(0, x), max(0, y)
    x_in_img_end, y_in_img_end = min(x + w, width), min(y + h, height)

    i = y_in_img
    j = x_in_img
    while i < y_in_img_end:
        while j < x_in_img_end:
            x1 = int(np.floor(j))
            y1 = int(np.floor(i))
            x2 = int(np.ceil(j))
            y2 = int(np.ceil(i))
            pixel_x1_y1 = img_np_origin[y1, x1, :]
            pixel_x1_y2 = img_np_origin[y2, x1, :]
            pixel_x2_y1 = img_np_origin[y1, x2, :]
            pixel_x2_y2 = img_np_origin[y2, x2, :]

            coefficient_x1 = 1 if x1 == x2 else (x2 - j) / (x2 - x1)
            coefficient_y1 = 1 if y1 == y2 else (y2 - i) / (y2 - y1)
            coefficient_x2 = 1 if x1 == x2 else (j - x1) / (x2 - x1)
            coefficient_y2 = 1 if y1 == y2 else (i - y1) / (y2 - y1)
            coefficient_x1_y1 = coefficient_x1 * coefficient_y1
            coefficient_x1_y2 = coefficient_x1 * coefficient_y2
            coefficient_x2_y1 = coefficient_x2 * coefficient_y1
            coefficient_x2_y2 = coefficient_x2 * coefficient_y2

            if x1 != x2 and y1 != y2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
                img_np[y1, x2] = img_np[y1, x2] - coefficient_x2_y1 * (pixel_x2_y1 - occlusion_color)
                img_np[y2, x1] = img_np[y2, x1] - coefficient_x1_y2 * (pixel_x1_y2 - occlusion_color)
                img_np[y2, x2] = img_np[y2, x2] - coefficient_x2_y2 * (pixel_x2_y2 - occlusion_color)
            elif x1 == x2 and y1 == y2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
            elif x1 == x2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
                img_np[y2, x1] = img_np[y2, x1] - coefficient_x1_y2 * (pixel_x1_y2 - occlusion_color)
            elif y1 == y2:
                img_np[y1, x1] = img_np[y1, x1] - coefficient_x1_y1 * (pixel_x1_y1 - occlusion_color)
                img_np[y1, x2] = img_np[y1, x2] - coefficient_x2_y1 * (pixel_x2_y1 - occlusion_color)
            j += 1
        i += 1
        j = x_in_img

    return img_np

