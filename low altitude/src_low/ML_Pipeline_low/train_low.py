import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from ultralytics import YOLO

def train_model():
    """
    [인수인계 노트]
    목적: YOLOv11 모델을 불러와 사용자 데이터셋으로 파인튜닝(Fine-tuning)합니다.
    """
    # 1. 순정 YOLO11n(경량 모델) 로드
    model = YOLO("yolo11n.pt") 

    # 2. 학습 실행 설정
    model.train(
        data=r"D:\Parking Detection.v1i.yolov11\data.yaml", # 상기 yaml 파일 경로
        epochs=100,            # 전체 데이터셋 반복 횟수
        imgsz=520,             # 학습 시 입력 이미지 크기 (512 내외 설정)
        batch=16,              # 한 번에 처리할 이미지 묶음 (GPU 메모리에 따라 조절)
        device=0,              # GPU 번호 (0번 GPU 사용)
        project="runs/detect", # 결과 저장 루트 폴더
        name="yolo11n_standard",
        workers=4,             # 데이터 로딩에 사용할 프로세스 수
        cache=False            # RAM 캐시 사용 여부 (메모리 부족 시 False)
    )

# 윈도우 환경에서 멀티프로세싱(workers > 0) 에러 방지를 위한 필수 구문
if __name__ == '__main__':
    train_model()