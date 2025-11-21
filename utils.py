import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import config


def tensor_to_pil(tensor):
    return transforms.ToPILImage()(tensor)


def preprocess_image(image, target_size=(224, 224)):
    """모델 입력을 위한 기본 전처리"""
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0).to(config.DEVICE)


def crop_with_margin(image, bbox, margin=0.15):
    """BBox에 마진을 주어 크롭하고 리사이즈"""
    x1, y1, x2, y2 = bbox
    width, height = image.size
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)

    new_x1 = max(0, x1 - dx)
    new_y1 = max(0, y1 - dy)
    new_x2 = min(width, x2 + dx)
    new_y2 = min(height, y2 + dy)

    cropped_image = image.crop((new_x1, new_y1, new_x2, new_y2))
    cropped_image = cropped_image.resize((224, 112))

    return cropped_image, (new_x1, new_y1, new_x2, new_y2)


def apply_perspective_transform(image, coords, target_size=(540, 116)):
    """좌표를 기반으로 원근 변환 적용"""
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    points_src = np.array(coords, dtype="float32")
    points_dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(points_src, points_dst)
    transformed = cv2.warpPerspective(image_np, matrix, (width, height))
    resized = cv2.resize(transformed, target_size)

    return resized


def color_transfer(source, target):
    """Source 이미지의 컬러 분포를 Target 이미지에 적용 (Reinhard method 변형)"""
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # 통계량 계산
    (l_mean_src, a_mean_src, b_mean_src) = cv2.mean(source_lab)[:3]
    (l_mean_tar, a_mean_tar, b_mean_tar) = cv2.mean(target_lab)[:3]

    l_std_src = source_lab[..., 0].std()
    a_std_src = source_lab[..., 1].std()
    b_std_src = source_lab[..., 2].std()

    l_std_tar = target_lab[..., 0].std()
    a_std_tar = target_lab[..., 1].std()
    b_std_tar = target_lab[..., 2].std()

    # 전송
    result_lab = source_lab.copy()
    L_factor, AB_factor = 1.05, 0.6

    result_lab[..., 0] = (
        (result_lab[..., 0] - l_mean_src) * (l_std_tar / l_std_src) * L_factor
    ) + l_mean_tar
    result_lab[..., 1] = (
        (result_lab[..., 1] - a_mean_src) * (a_std_tar / a_std_src) * AB_factor
    ) + a_mean_tar
    result_lab[..., 2] = (
        (result_lab[..., 2] - b_mean_src) * (b_std_tar / b_std_src) * AB_factor
    ) + b_mean_tar

    result_lab = np.clip(result_lab, 0, 255).astype("uint8")
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


# (기타 denoise_image 등 필요한 함수들 여기에 추가)
