# -*- coding: utf-8 -*-
# created by makise, 2022/2/20

# customize some occlusion data_transform for GTSRB dataset
import random

from PIL import Image

"""
add single occlusion to image
"""
class OcclusionTransform:
    """
    This transform will take a PIL image and apply a 5x5 occlusion mask to it.
    """

    def __init__(self, occlusion_height, occlusion_width):
        self.occlusion_height = occlusion_height
        self.occlusion_width = occlusion_width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Randomly occluded image.
        """

        w, h = img.size
        # generate black occlusion mask
        mask = Image.new('L', (self.occlusion_width, self.occlusion_height), 0)

        # paste the mask on the image randomly
        x = random.randint(0, w - self.occlusion_width)
        y = random.randint(0, h - self.occlusion_height)
        img.paste(mask, (x, y))

        return img

"""
add multiple occlusion to image without overlap
"""
class MulOcclusionTransform:
    """
    This transform will take a PIL image and apply multiple occlusion mask to it.
    """

    def __init__(self, occlusion_height, occlusion_width, occlusion_num):
        self.occlusion_height = occlusion_height
        self.occlusion_width = occlusion_width
        self.occlusion_num = occlusion_num

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Randomly occluded image.
        """

        w, h = img.size
        # generate black occlusion mask
        mask = Image.new('L', (self.occlusion_width, self.occlusion_height), 0)
        # record the occupied area
        occupied = [False] * (w * h)
        # paste occlusion_num masks on the image randomly without overlap
        i = 0
        while i < self.occlusion_num:
            x = random.randint(0, w - self.occlusion_width)
            y = random.randint(0, h - self.occlusion_height)
            # check if the area is occupied
            is_occupied = False
            for j in range(x, x + self.occlusion_width):
                if not is_occupied:
                    for k in range(y, y + self.occlusion_height):
                        if occupied[j * w + k]:
                            is_occupied = True
                            break
            # paste the mask on the image if there is no overlap
            if not is_occupied:
                img.paste(mask, (x, y))
                for j in range(x, x + self.occlusion_width):
                    for k in range(y, y + self.occlusion_height):
                        occupied[j * w + k] = True
                i += 1

        return img