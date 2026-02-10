from ultralytics import YOLO

def train_model():
    # 1. 순정 YOLO11n 모델 로드
    model = YOLO("yolo11n.pt") 

    # 2. 학습 실행
    model.train(
        data="data.yaml",      # 데이터셋 설정 파일 경로
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="runs/detect",
        name="yolo11n_standard",
        workers=4,             # 이제 에러 없이 작동합니다
        cache=False
    )

# 이 부분이 윈도우 멀티프로세싱 에러를 막는 핵심입니다
if __name__ == '__main__':
    train_model()