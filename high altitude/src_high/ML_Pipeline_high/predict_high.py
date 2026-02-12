from ultralytics import YOLO
import os
import glob

# 1. 모델 로드
# 최신 파일 경로로 설정 (yolo11n_standard3)
model_path = r"D:\Parking Detection.v1i.yolov11\runs\detect\runs\detect\yolo11n_high_alt_140m_slicing\weights\best.pt"
model = YOLO(model_path)

# 2. 이미지 폴더 경로 설정
input_folder = r"D:\Parking Detection.v1i.yolov11\low altitude\runs\Empirical_data_test"

# 3. 폴더 내 모든 이미지 파일 리스트 생성
image_extensions = ['*.jpeg', '*.jpg', '*.png', '*.bmp']
image_list = []
for ext in image_extensions:
    image_list.extend(glob.glob(os.path.join(input_folder, ext)))

if not image_list:
    print("❌ 해당 폴더에 처리할 이미지 파일이 없습니다. 경로를 확인해주세요.")
else:
    print(f"🔎 총 {len(image_list)}개의 이미지를 발견했습니다. 추론을 시작합니다...")

    # 4. 추론 및 자동 저장
    # 주석의 의도(글자/점수 숨기기)에 맞게 불리언 값을 수정했습니다.
    results = model.predict(
        source=image_list,
        conf=0.3,           # 신뢰도 임계값
        imgsz=640,          # 이미지 크기
        iou=0.3,            # NMS 임계값
        augment=True,       # 추론 시 증강 사용 (정밀도 향상)
        agnostic_nms=True,  # 클래스 간 중복 박스 제거
        save=True,          # 결과 이미지 저장
        project=r"D:\Parking Detection.v1i.yolov11\high altitude\runs\predict_filtered",
        name="inference_specific_640",
        exist_ok=True,
        stream=False,       # 결과 요약을 출력하기 위해 리스트 형태로 반환받음

        # 시각화 옵션 설정
        line_width=1,       # 박스 선 굵기 최소화 (가늘게)
        show_labels=False,  # "car" 같은 클래스 이름 숨기기 (True -> False 수정)
        show_conf=False,    # 신뢰도 점수(0.85 등) 숨기기 (True -> False 수정)
        boxes=True          # 박스 테두리는 출력 (box -> boxes로 매개변수 명칭 확인 필요)
    )

    # 5. 결과 요약 출력
    print("\n" + "="*50)
    for result in results:
        file_name = os.path.basename(result.path)
        # 각 이미지별 검출된 객체 수 출력
        print(f"✅ {file_name}: {len(result.boxes)}대 검출")
    
    print("="*50)
    # 첫 번째 결과의 저장 경로 출력
    if len(results) > 0:
        print(f"📂 모든 결과가 다음 폴더에 저장되었습니다: {results[0].save_dir}")