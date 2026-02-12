from ultralytics import YOLO
import os
import glob

# 1. 모델 로드
model_path = r"D:\Parking Detection.v1i.yolov11\runs\detect\yolo11n_standard3\weights\best.pt"
model = YOLO(model_path)

# 2. 이미지 폴더 경로 설정
input_folder = r"D:\Parking Detection.v1i.yolov11\runs\Empirical_data_test"

# 3. 폴더 내 모든 이미지 파일 리스트 생성 (다양한 확장자 포함)
image_extensions = ['*.jpeg', '*.jpg', '*.png', '*.bmp']
image_list = []
for ext in image_extensions:
    image_list.extend(glob.glob(os.path.join(input_folder, ext)))

if not image_list:
    print("❌ 해당 폴더에 처리할 이미지 파일이 없습니다. 경로를 확인해주세요.")
else:
    print(f"🔎 총 {len(image_list)}개의 이미지를 발견했습니다. 추론을 시작합니다...")

    # 4. 추론 및 자동 저장
    # stream=True 옵션은 많은 양의 이미지를 처리할 때 메모리 부하를 줄여줍니다.
    results = model.predict(
        source=image_list,
        conf=0.3,
        imgsz=512,
        iou=0.3,
        augment=True,
        agnostic_nms=True,
        save=True,
        project=r"D:\Parking Detection.v1i.yolov11\runs\predict_filtered",
        name="inference_specific",
        exist_ok=True,
        stream=False,

        # 여기서부터 추가/수정할 옵션입니다
        line_width=1,       # 박스 선 굵기 (숫자가 작을수록 가늘어짐, 최소 1)
        show_labels=True,  # "car" 같은 글자 숨기기 (그림자 확인에 방해됨)
        show_conf=True,    # 신뢰도 점수(0.85 등) 숨기기
        box=False            # 박스 테두리만 출력 (기본값 True)
    )


    # 5. 결과 요약 출력
    print("\n" + "="*50)
    for result in results:
        file_name = os.path.basename(result.path)
        print(f"✅ {file_name}: {len(result.boxes)}대 검출")
    print("="*50)
    print(f"📂 모든 결과가 다음 폴더에 저장되었습니다: {results[0].save_dir}")