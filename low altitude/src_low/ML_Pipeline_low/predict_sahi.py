import os
import glob
import shutil
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. 모델 로드 (0.11.36 버전 호환 방식)
model_path = r"D:\Parking Detection.v1i.yolov11\runs\detect\yolo11n_standard3\weights\best.pt"

# from_pretrained가 현재 버전에서 가장 확실한 로드 방식입니다.
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0"
)

# 2. 경로 설정 및 덮어쓰기(폴더 초기화) 설정
input_folder = r"D:\Parking Detection.v1i.yolov11\runs\Empirical_data_test"
output_project_dir = r"D:\Parking Detection.v1i.yolov11\runs\predict_filtered\sahi_final_results\sahi"

# [핵심] 기존 폴더가 있다면 통째로 지우고 새로 만들어 '덮어쓰기' 효과를 줌
if os.path.exists(output_project_dir):
    shutil.rmtree(output_project_dir)
os.makedirs(output_project_dir, exist_ok=True)

# 3. 이미지 리스트 생성 (폴더나 시스템 파일을 제외하고 실제 이미지만 선별)
image_list = []
valid_extensions = ('.jpeg', '.jpg', '.png', '.bmp')

for file in os.listdir(input_folder):
    if file.lower().endswith(valid_extensions):
        full_path = os.path.join(input_folder, file)
        if os.path.isfile(full_path):
            image_list.append(full_path)

if not image_list:
    print(f"❌ '{input_folder}'에서 이미지를 찾을 수 없습니다.")
else:
    print(f"🔎 총 {len(image_list)}장의 이미지를 발견했습니다. 추론을 시작합니다...")

# 4. 슬라이싱 추론 및 시각화 저장
for image_path in image_list:
    file_name = os.path.basename(image_path)
    
    # perform_standard_prediction 등 에러 유발 인자 제거
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5
    )
    
    # 5. 결과 저장 (export_visuals로 이름 변경 및 박스 옵션 적용)
    save_name = os.path.splitext(file_name)[0]
    result.export_visuals(
        export_dir=output_project_dir, 
        file_name=save_name,
        hide_labels=False,  # 글자 숨김
        hide_conf=False,    # 확신도 숨김
        rect_th=1          # 선 굵기 최소화 (1)
    )
    
    print(f"✅ {file_name}: {len(result.object_prediction_list)}대 검출 완료")

print(f"\n📂 모든 결과가 덮어쓰기되어 저장되었습니다: {output_project_dir}")