import os
from ultralytics import YOLO

# 1. 경로 설정
# [주의] data.yaml 파일 내부의 경로들도 반드시 'D:/path/to/data' 처럼 슬래시(/)로 수정되어 있어야 합니다.
DATA_YAML_PATH = r"D:/Parking Detection.v1i.yolov11/high altitude/src_high/ML_Pipeline_high/data.yaml"

def train_model():
    """
    [인수인계 핵심 로직]
    목적: 고도 140m에서 매우 작게 보이는 차량을 탐지하기 위해 최적화된 학습을 수행합니다.
    특징: 슬라이싱 크기(640)와 학습 이미지 크기(imgsz)를 1:1로 일치시켜 해상도 손실을 방지합니다.
    """
    # 2. YOLO11n(Nano) 모델 로드
    model = YOLO("yolo11n.pt") 

    # 3. 학습 실행 설정
    model.train(
        data=DATA_YAML_PATH,           # 데이터셋 설정 파일 경로
        epochs=100,                    # 100회 반복
        imgsz=640,                     # 슬라이싱 크기와 동일하게 설정
        batch=16,                      # RTX 3060 12GB 환경에서 16~32 권장
        device=0,                      # GPU 사용
        project="runs/detect",         
        name="yolo11n_high_alt_140m_slicing", 
        workers=4,                     
        cache=True,                    # RAM 캐싱으로 속도 향상
        patience=20,                   # 조기 종료 설정
        mosaic=0.5,                    # 작은 객체 탐지 성능 향상
        exist_ok=True                  # 동일 이름 폴더 있을 시 덮어쓰기 허용
    )

if __name__ == '__main__':
    # 윈도우 환경에서 경로 인식 문제를 방지하기 위해 절대 경로로 변환하여 전달
    DATA_YAML_PATH = os.path.abspath(DATA_YAML_PATH)
    train_model()