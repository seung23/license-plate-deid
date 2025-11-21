import os
import random
import cv2
import numpy as np


class ImageGenerator:
    def __init__(self, save_dir, plates_dir_a, plates_dir_b, nums_dir, chars_dir):
        self.save_dir = save_dir
        self.plates_dir_a = plates_dir_a
        self.plates_dir_b = plates_dir_b

        # 이미지 로드 (생성자에서 미리 로드하여 효율성 증대)
        self.numbers = self._load_images(nums_dir)
        self.chars = self._load_images(chars_dir)

    def _load_images(self, path):
        return [cv2.imread(os.path.join(path, f)) for f in sorted(os.listdir(path))]

    def add_overlay(self, background, foreground, x, y):
        """이미지 위에 다른 이미지 합성 (Alpha channel 처리 포함)"""
        # (노트북의 add 함수 로직을 여기에 구현)
        # ... (중략) ...
        return background

    def generate_type_a(self):
        """구형 번호판 생성 로직"""
        # ... (노트북의 type_a 함수 로직) ...
        pass

    def generate_type_b(self):
        """신형 번호판 생성 로직"""
        # ... (노트북의 type_b 함수 로직) ...
        pass
