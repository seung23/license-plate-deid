import torch

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 가중치 경로 설정
YOLO_MODEL_PATH = "weights/yolo_best.pt"
CLS_MODEL_PATH = "weights/resnet_classification_best.pth"
REG_MODEL_PATH = "resnet_regression_best.pth"
VGG_PATH = "weights/vgg_normalised.pth"
DECODER_PATH = "weights/decoder_best.pth.tar"

# 데이터 생성 관련 경로
ASSETS_DIR = "assets"
PLATES_DIR_A = f"{ASSETS_DIR}/plates/type_a"
PLATES_DIR_B = f"{ASSETS_DIR}/plates/type_b"
NUMS_DIR = f"{ASSETS_DIR}/nums"
CHARS_DIR = f"{ASSETS_DIR}/chars"
RESULT_DIR = "results"
