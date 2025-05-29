# License Plate De-identification (Capstone Project)

🚘 YOLOv5, ResNet18, AdaIN을 활용한 차량 번호판 비식별화 딥러닝 시스템입니다.  
📚 아주대학교 전자공학과 캡스톤 디자인 프로젝트 결과물입니다.

## 🧠 주요 기술 스택
- Python, PyTorch, OpenCV
- YOLOv5: 번호판 탐지
- ResNet18: 꼭짓점 회귀
- AdaIN: 스타일 전이
- 
## 📁 파일 구성
- `main_pipeline.ipynb`: 전체 파이프라인 테스트용
- `results/`: 비식별화 전후 비교 이미지
- `test_inputs/`: 테스트용 입력 이미지

## 🚀 실행
- Google Colab 기준으로 실행됩니다.
- `main_pipeline.ipynb` 실행 시, 기본 예제 이미지로 테스트 가능
- 필요한 패키지는 셀 상단에서 설치됩니다.

## 🖼️ 예시 결과

| 입력 이미지 | 비식별화 결과 |
|-------------|----------------|
| ![](results/before.jpg) | ![](results/after.jpg) |
