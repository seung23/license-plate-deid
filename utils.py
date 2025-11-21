import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import config

def tensor_to_pil(tensor):
    return transforms.ToPILImage()(tensor)

def denoise_image(style_image):
    style_image_cv = np.array(style_image)
    style_image_cv = cv2.cvtColor(style_image_cv, cv2.COLOR_RGB2BGR)

    blurred = cv2.GaussianBlur(style_image_cv, (5, 5), 0)
    bilateral = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)
    denoised = cv2.fastNlMeansDenoisingColored(bilateral, None, 10, 10, 7, 21)

    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0).to(config.DEVICE)

def preprocess_image_for_coords(image):
    transform = transforms.Compose(
        [
            transforms.Resize((112, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0).to(config.DEVICE)

def add_margin(image, margin_ratio=0.1):
    width, height = image.size
    margin_w = int(width * margin_ratio)
    margin_h = int(height * margin_ratio)
    new_width = width + 2 * margin_w
    new_height = height + 2 * margin_h

    new_image = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    new_image.paste(image, (margin_w, margin_h))
    return new_image, margin_w, margin_h

def crop_with_margin(image, bbox, margin=0.15):
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
    image_np = np.array(image)

    points_src = np.array(coords, dtype="float32")

    width, height = target_size
    points_dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(points_src, points_dst)
    transformed = cv2.warpPerspective(image_np, matrix, (width, height))

    return transformed

def color_transfer(source, target):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, a_mean_src, b_mean_src) = cv2.mean(source_lab)[:3]
    (l_mean_tar, a_mean_tar, b_mean_tar) = cv2.mean(target_lab)[:3]

    l_std_src = source_lab[..., 0].std()
    a_std_src = source_lab[..., 1].std()
    b_std_src = source_lab[..., 2].std()

    l_std_tar = target_lab[..., 0].std()
    a_std_tar = target_lab[..., 1].std()
    b_std_tar = target_lab[..., 2].std()

    result_lab = source_lab.copy()

    L_factor = 1.05
    AB_factor = 0.6

    l_std_src = l_std_src if l_std_src > 1e-5 else 1.0
    a_std_src = a_std_src if a_std_src > 1e-5 else 1.0
    b_std_src = b_std_src if b_std_src > 1e-5 else 1.0

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

def apply_soft_shadow_from_original(
    original_plate_rgb, styled_plate_rgb, blur_ksize=51, shadow_strength=0.4
):
    gray = cv2.cvtColor(original_plate_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    shading = blurred.astype(np.float32) / 255.0
    shading_resized = cv2.resize(
        shading, (styled_plate_rgb.shape[1], styled_plate_rgb.shape[0])
    )
    shadow_filter = (
        1.0 - (1.0 - shading_resized[:, :, np.newaxis]) * shadow_strength
    )
    styled_plate_float = styled_plate_rgb.astype(np.float32) / 255.0
    shadowed = styled_plate_float * shadow_filter
    shadowed = np.clip(shadowed * 255.0, 0, 255).astype(np.uint8)
    return shadowed

def _load_patched(*args, **kwargs):
    import torch

    _orig_load = torch.load
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
