import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import os
import glob
import shutil
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. 모델 로드
model_path = os.path.join(BASE_DIR, "best.pt")

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.2, # 0.2 정도로 낮춰서 일단 다 잡은 뒤 아래에서 필터링
    device="cuda:0"
)

# 2. 경로 설정
input_folder = os.path.join(BASE_DIR, "Empirical_data_test")
output_project_dir = os.path.join(BASE_DIR, "sahi_final_results")

if os.path.exists(output_project_dir):
    shutil.rmtree(output_project_dir)
os.makedirs(output_project_dir, exist_ok=True)

# 3. 이미지 리스트 생성
image_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
              if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))]

if not image_list:
    print(f"❌ 이미지를 찾을 수 없습니다.")
else:
    print(f"🔎 총 {len(image_list)}장의 이미지로 '더블 스캔' 추론을 시작합니다...")

# 4. 추론 루프
for image_path in image_list:
    file_name = os.path.basename(image_path)
    
    # [수정] 에러 유발 인자 제거 및 더블 스캔 활성화
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        perform_standard_pred=True,        # [핵심] 더블 스캔 활성화
        postprocess_type="GREEDYNMM",       # 박스 겹침 방지
        postprocess_match_metric="IOS",     # 밀집 주차장용 기준
        postprocess_match_threshold=0.6     # 병합 강도
    )

    # [추가] 노이즈 필터링 (눈송이나 아주 작은 티끌 제거)
    # 너무 작은 박스(예: 면적 150픽셀 이하)는 리스트에서 제외합니다.
    result.object_prediction_list = [
        obj for obj in result.object_prediction_list 
        if (obj.bbox.maxx - obj.bbox.minx) * (obj.bbox.maxy - obj.bbox.miny) > 150
    ]
    
    # 5. 결과 저장
    save_name = os.path.splitext(file_name)[0]
    result.export_visuals(
        export_dir=output_project_dir, 
        file_name=save_name,
        hide_labels=False,
        hide_conf=False,
        rect_th=1
    )
    
    print(f"✅ {file_name}: {len(result.object_prediction_list)}대 검출 (더블 스캔 완료)")

print(f"\n📂 결과 저장 완료: {output_project_dir}")