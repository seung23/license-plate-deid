import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt

import config
import models
import utils
import generator
import style_transfer

import sys

from yolov5 import YOLOv5


def main():

    print("Loading models...")

    _orig_load = torch.load

    def _load_patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_patched

    yolo_model = YOLOv5(config.YOLO_MODEL_PATH)

    cls_model = models.ClassificationModel(num_classes=9).to(config.DEVICE)
    cls_model.load_state_dict(
        torch.load(config.CLS_MODEL_PATH, map_location=config.DEVICE)
    )
    cls_model.eval()

    reg_model = models.CoordinateModel(num_classes=9, num_coords=8).to(config.DEVICE)
    reg_model.load_state_dict(
        torch.load(config.REG_MODEL_PATH, map_location=config.DEVICE)
    )
    reg_model.eval()

    vgg = models.get_vgg_encoder().to(config.DEVICE)
    decoder = models.get_decoder().to(config.DEVICE)

    vgg.load_state_dict(torch.load(config.VGG_PATH, map_location=config.DEVICE))
    decoder.load_state_dict(torch.load(config.DECODER_PATH, map_location=config.DEVICE))

    vgg = torch.nn.Sequential(*list(vgg.children())[:31])

    vgg.eval()
    decoder.eval()

    img_path = "test_image.jpg"
    if not os.path.exists(img_path):
        print(f"Error: Test image not found at {img_path}")

        return

    original_image = Image.open(img_path)

    print("Detecting license plate...")
    results = yolo_model.predict(original_image)

    if results.xyxy[0].size(0) == 0:
        print("No license plate detected.")
        return

    bbox = results.xyxy[0][0][:4].detach().cpu().numpy().astype(int)
    print(f"Detected bbox: {bbox}")

    cropped_img, crop_bbox = utils.crop_with_margin(original_image, bbox)

    input_tensor_cls = utils.preprocess_image(cropped_img)

    input_tensor_reg = utils.preprocess_image_for_coords(cropped_img)

    with torch.no_grad():
        cls_output = cls_model(input_tensor_cls)
        class_label = cls_output.argmax(dim=1).item()
    print(f"Predicted class label: {class_label}")

    with torch.no_grad():
        class_tensor = torch.tensor([class_label]).to(config.DEVICE)
        coords_output = reg_model(input_tensor_reg, class_tensor)

        predicted_coords = coords_output.squeeze().cpu().numpy()
        x_coords = predicted_coords[:4]
        y_coords = predicted_coords[4:]
        predicted_coords = np.column_stack((x_coords, y_coords))
        predicted_coords = np.clip(predicted_coords, 0, 1)

        cw, ch = cropped_img.size
        predicted_coords = predicted_coords * np.array([[cw, ch]] * 4)
        predicted_coords = predicted_coords.astype(int)

    print(f"Predicted coordinates (crop base): {predicted_coords}")

    print("Generating fake plate...")
    gen = generator.ImageGenerator(
        config.RESULT_DIR,
        config.PLATES_DIR_A,
        config.PLATES_DIR_B,
        config.NUMS_DIR,
        config.CHARS_DIR,
    )

    if class_label == 3:
        plate_image_o = gen.generate_type_a()
    elif class_label == 6:
        plate_image_o = gen.generate_type_b()
    else:

        print(f"Warning: Unhandled class label {class_label}. Using Type A.")
        plate_image_o = gen.generate_type_a()

    if plate_image_o.shape[2] == 4:
        plate_alpha = plate_image_o[:, :, 3]
        plate_image_o = plate_image_o[:, :, :3]
    else:
        plate_alpha = np.ones(plate_image_o.shape[:2], dtype=np.uint8) * 255

    plate_image_rgb = cv2.cvtColor(plate_image_o, cv2.COLOR_BGR2RGB)
    plate_image_pil = Image.fromarray(plate_image_rgb)

    center = np.mean(predicted_coords, axis=0)
    expanded_coords = center + 1.05 * (predicted_coords - center)
    expanded_coords = expanded_coords.astype(np.float32)

    transformed_image_origin = utils.apply_perspective_transform(
        cropped_img, expanded_coords
    )

    style_image = Image.fromarray(transformed_image_origin)

    style_image_bgr = np.array(style_image)[:, :, ::-1]

    plate_bgr_transferred = utils.color_transfer(plate_image_o, style_image_bgr)
    plate_rgb_transferred = cv2.cvtColor(plate_bgr_transferred, cv2.COLOR_BGR2RGB)
    plate_pil_transferred = Image.fromarray(plate_rgb_transferred)

    print("Running Style Transfer...")

    tf = style_transfer.get_transform()

    content_tensor = tf(plate_image_pil).to(config.DEVICE).unsqueeze(0)

    style_tensor = tf(style_image).to(config.DEVICE).unsqueeze(0)

    with torch.no_grad():
        styled_output = style_transfer.run_style_transfer(
            vgg, decoder, content_tensor, style_tensor, alpha=1.0
        )

    styled_plate = styled_output.squeeze(0).cpu().detach().numpy()
    styled_plate = np.transpose(styled_plate, (1, 2, 0))
    styled_plate = (styled_plate * 255).astype(np.uint8)

    styled_plate_bgr = cv2.cvtColor(styled_plate, cv2.COLOR_RGB2BGR)
    styled_plate_bgr = utils.color_transfer(styled_plate_bgr, style_image_bgr)

    plate_h, plate_w = styled_plate_bgr.shape[:2]
    alpha_resized = cv2.resize(plate_alpha, (plate_w, plate_h))
    styled_plate_bgra = np.dstack([styled_plate_bgr, alpha_resized])

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    scale_x = crop_w / 224.0
    scale_y = crop_h / 112.0

    coords_original = predicted_coords.astype(float) * np.array([scale_x, scale_y])
    coords_original[:, 0] += crop_x1
    coords_original[:, 1] += crop_y1
    coords_original = coords_original.astype(int)

    print(f"Target coordinates (original): {coords_original}")

    pts_src = np.array(
        [[0, 0], [plate_w, 0], [plate_w, plate_h], [0, plate_h]], dtype="float32"
    )

    pts_dst = coords_original.astype("float32")

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    orig_img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    warped_plate = cv2.warpPerspective(
        styled_plate_bgra, matrix, (orig_img_cv.shape[1], orig_img_cv.shape[0])
    )

    mask = warped_plate[:, :, 3] / 255.0
    mask = np.expand_dims(mask, axis=2)

    result = orig_img_cv * (1 - mask) + warped_plate[:, :, :3] * mask
    result = result.astype(np.uint8)

    save_path = os.path.join(config.RESULT_DIR, "final_result.jpg")
    cv2.imwrite(save_path, result)
    print(f"Processing complete. Result saved to {save_path}")


if __name__ == "__main__":
    main()
