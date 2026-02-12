import os
import shutil
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. 모델 및 데이터 경로 설정 (절대 경로로 확정)
model_path = r"D:\Parking Detection.v1i.yolov11\runs\detect\yolo11n_standard3\weights\best.pt"
input_folder = r"D:\Parking Detection.v1i.yolov11\runs\Empirical_data_test"
output_project_dir = r"D:\Parking Detection.v1i.yolov11\runs\predict_filtered\sahi_final_results\Dual-scan"

# 2. SAHI 모델 로드
# YOLOv11은 yolov8 타입으로 로드하면 완벽하게 호환됩니다.
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.2, 
    device="cuda:0"
)

# 3. 출력 폴더 초기화 (기존 결과 삭제 후 생성)
if os.path.exists(output_project_dir):
    shutil.rmtree(output_project_dir)
os.makedirs(output_project_dir, exist_ok=True)

# 4. 이미지 리스트 생성
image_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
              if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))]

if not image_list:
    print(f"❌ '{input_folder}'에서 이미지를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
else:
    print(f"🔎 총 {len(image_list)}장의 이미지로 '더블 스캔' 추론을 시작합니다...")

# 5. 추론 루프
for image_path in image_list:
    file_name = os.path.basename(image_path)
    
    # SAHI 핵심 로직 실행
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        
        # [핵심] 맥락 유지를 위한 더블 스캔 활성화
        perform_standard_pred=True,        
        
        # 후처리 설정 (에러 방지를 위해 가장 안정적인 iou 조합 사용)
        postprocess_type="NMS", 
        postprocess_match_metric="IOU", 
        postprocess_match_threshold=0.5 
    )

    # [추가] 노이즈 필터링 (면적 150픽셀 이하인 작은 티끌/눈송이 제거)
    result.object_prediction_list = [
        obj for obj in result.object_prediction_list 
        if (obj.bbox.maxx - obj.bbox.minx) * (obj.bbox.maxy - obj.bbox.miny) > 150
    ]
    
    # 6. 결과 시각화 저장
    save_name = os.path.splitext(file_name)[0]
    result.export_visuals(
        export_dir=output_project_dir, 
        file_name=save_name,
        hide_labels=False,
        hide_conf=False,
        rect_th=1
    )
    
    print(f"✅ {file_name}: {len(result.object_prediction_list)}대 검출 완료")

print(f"\n📂 모든 결과가 저장되었습니다: {output_project_dir}")