from ultralytics import YOLO
import os

# 1. 모델 로드
model_path = r"D:\Parking Detection.v1i.yolov11\runs\detect\runs\detect\yolo11n_standard2\weights\best.pt"
image_path = r"C:\Users\vrro3\Downloads\DJI_20260206154515_0001_V.jpeg"
model = YOLO(model_path)

# 2. 추론 및 자동 저장
# save=True를 사용하면 YOLO가 자체적으로 겹치지 않는 박스를 그려서 저장합니다.
results = model.predict(
    source=image_path,
    conf=0.5,
    imgsz=640,
    iou=0.1,             # 중복 박스 제거 (10% 이상 겹치면 하나로 통합)
    augment=True,
    agnostic_nms=True,   # 클래스 상관없이 겹침 제거
    save=True,           # [핵심] 수동 루프 대신 자동 저장 사용
    project=r"D:\Parking Detection.v1i.yolov11\runs\predict_filtered",
    name="inference_clean",
    exist_ok=True
)

# 3. 간단한 결과 출력
for result in results:
    print(f"🏁 검출 완료: 총 {len(result.boxes)}대의 차량이 감지되었습니다.")
    print(f"결과 확인 경로: {result.save_dir}")