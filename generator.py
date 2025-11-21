import os
import random
import cv2
import numpy as np
from PIL import Image


class ImageGenerator:
    def __init__(self, save_dir, plates_dir, plates_dir2, nums_dir, chars_dir):
        self.save_dir = save_dir
        self.plates_dir = plates_dir
        self.plates_dir2 = plates_dir2
        self.nums_dir = nums_dir
        self.chars_dir = chars_dir

        self.numbers = self._load_images(nums_dir)
        self.chars = self._load_images(chars_dir)

        self.number_list = sorted(os.listdir(nums_dir))
        self.char_list = sorted(os.listdir(chars_dir))

        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)

    def _load_images(self, path):
        return [cv2.imread(os.path.join(path, f)) for f in sorted(os.listdir(path))]

    def add_overlay(self, background, foreground, x, y):
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255.0
            for c in range(3):
                background[
                    y : y + foreground.shape[0], x : x + foreground.shape[1], c
                ] = (
                    alpha * foreground[:, :, c]
                    + (1 - alpha)
                    * background[
                        y : y + foreground.shape[0], x : x + foreground.shape[1], c
                    ]
                )
            if background.shape[2] == 4:
                background_alpha = (
                    background[
                        y : y + foreground.shape[0], x : x + foreground.shape[1], 3
                    ]
                    / 255.0
                )
                new_alpha = alpha + (1 - alpha) * background_alpha
                background[
                    y : y + foreground.shape[0], x : x + foreground.shape[1], 3
                ] = (new_alpha * 255).astype(np.uint8)
        else:
            foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
            mask = foreground_gray < 200
            for c in range(3):
                background[y : y + foreground.shape[0], x : x + foreground.shape[1], c][
                    mask
                ] = foreground_gray[mask]

        return background

    def random_bright(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        v = np.clip(v * random.uniform(0.5, 1.5), 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def generate_type_a(self):
        plate_files = os.listdir(self.plates_dir)
        plate_file = random.choice(plate_files)
        plate_path = os.path.join(self.plates_dir, plate_file)
        plate = cv2.imread(plate_path, cv2.IMREAD_UNCHANGED)
        plate = cv2.resize(plate, (520, 110))

        first_digits = random.sample(self.numbers, 2)
        last_digits = random.sample(self.numbers, 4)
        chars = random.sample(self.chars, 1)

        positions_first = [(40, 15), (97, 15)]
        positions_last = [
            (258, 15),
            (313, 15),
            (368, 15),
            (423, 15),
        ]
        char_pos = (154, 15)

        for i, (num, pos) in enumerate(zip(first_digits, positions_first)):
            plate = self.add_overlay(plate, cv2.resize(num, (56, 83)), pos[0], pos[1])

        plate = self.add_overlay(
            plate, cv2.resize(chars[0], (60, 83)), char_pos[0], char_pos[1]
        )

        for i, (num, pos) in enumerate(zip(last_digits, positions_last)):
            plate = self.add_overlay(plate, cv2.resize(num, (56, 83)), pos[0], pos[1])

        return plate

    def generate_type_b(self):
        sft = 20
        sft2 = 42

        plate_files = os.listdir(self.plates_dir2)
        plate_file = random.choice(plate_files)
        plate_path = os.path.join(self.plates_dir2, plate_file)
        plate = cv2.imread(plate_path, cv2.IMREAD_UNCHANGED)
        plate = cv2.resize(plate, (520, 110))

        first_digits = random.sample(self.numbers, 3)
        last_digits = random.sample(self.numbers, 4)
        chars = random.sample(self.chars, 1)

        positions_first = [
            (40 + sft, 15),
            (91 + sft, 15),
            (142 + sft, 15),
        ]
        positions_last = [
            (258 + sft2, 15),
            (309 + sft2, 15),
            (360 + sft2, 15),
            (411 + sft2, 15),
        ]
        char_pos = (221, 15)

        for i, (num, pos) in enumerate(zip(first_digits, positions_first)):
            plate = self.add_overlay(plate, cv2.resize(num, (56, 83)), pos[0], pos[1])

        plate = self.add_overlay(
            plate, cv2.resize(chars[0], (60, 83)), char_pos[0], char_pos[1]
        )

        for i, (num, pos) in enumerate(zip(last_digits, positions_last)):
            plate = self.add_overlay(plate, cv2.resize(num, (56, 83)), pos[0], pos[1])

        return plate
