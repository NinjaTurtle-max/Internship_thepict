import os
from ultralytics import YOLO

# 1. 경로 설정 (전처리 Step 4에서 생성한 640 슬라이싱 데이터셋의 설정 파일)
# [인수인계 노트] 이 YAML 파일에 데이터셋의 실제 위치가 정의되어 있습니다.
DATA_YAML_PATH = r"D:\Parking Detection.v1i.yolov11\high altitude\src_high\ML_Pipeline_high\data.yaml"

def train_model():
    """
    [인수인계 핵심 로직]
    목적: 고도 140m에서 매우 작게 보이는 차량을 탐지하기 위해 최적화된 학습을 수행합니다.
    특징: 슬라이싱 크기(640)와 학습 이미지 크기(imgsz)를 1:1로 일치시켜 해상도 손실을 방지합니다.
    """
    # 2. YOLO11n(Nano) 모델 로드 (속도와 효율이 가장 좋은 경량 모델)
    model = YOLO("yolo11n.pt") 

    # 3. 학습 실행 설정
    model.train(
        data=DATA_YAML_PATH,           # 데이터셋 설정 파일 경로
        epochs=100,                    # 충분한 학습을 위해 100회 반복
        imgsz=640,                     # [핵심] 슬라이싱 크기와 동일하게 설정하여 미세 객체 특징 유지
        batch=16,                      # GPU 메모리(VRAM) 상황에 따라 8, 16, 32 중 선택
        device=0,                      # 0번 GPU 사용 (CUDA 환경)
        project="runs/detect",         # 결과 저장 최상위 폴더
        name="yolo11n_high_alt_140m_slicing", # 모델 구분을 위한 고유 이름
        workers=4,                     # 데이터 로딩에 사용할 CPU 프로세스 수
        cache=True,                    # RAM에 데이터를 캐싱하여 학습 속도 대폭 향상
        patience=20,                   # 20회 동안 성능 향상이 없으면 조기 종료 (Overfitting 방지)
        mosaic=0.5                     # 여러 이미지를 합쳐 작은 객체 학습을 돕는 증강 기법 유지
    )

# 윈도우 환경의 멀티프로세싱 안정성을 위한 진입점 설정
if __name__ == '__main__':
    train_model()