import cv2
import os
import glob
import shutil

def resize_data_recursive(img_base, lbl_base, output_base):
    # 저장 경로 설정
    save_img_dir = os.path.join(output_base, 'images')
    save_lbl_dir = os.path.join(output_base, 'labels')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    # 1. 모든 이미지 파일 검색 (하위 폴더 포함)
    img_paths = []
    for ext in ('**/*.jpg', '**/*.JPG', '**/*.jpeg', '**/*.png'):
        img_paths.extend(glob.glob(os.path.join(img_base, ext), recursive=True))

    if not img_paths:
        print(f"❌ 이미지를 찾을 수 없습니다: {img_base}")
        return

    print(f"✅ [{os.path.basename(output_base)}] 총 {len(img_paths)}개 이미지 리사이징 시작 (Target: 640x360)")

    for img_path in img_paths:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 2. 이미지 리사이징 (640x360)
        resized_img = cv2.resize(img, (640, 360))
        
        # 3. 라벨 파일 찾기 및 복사
        lbl_pattern = os.path.join(lbl_base, f"**/{file_name}.txt")
        lbl_found = glob.glob(lbl_pattern, recursive=True)
        
        # 이미지 저장
        cv2.imwrite(os.path.join(save_img_dir, f"{file_name}.jpg"), resized_img)
        
        # 라벨 복사 (YOLO 좌표는 상대값이므로 내용 수정 불필요)
        if lbl_found:
            shutil.copy(lbl_found[0], os.path.join(save_lbl_dir, f"{file_name}.txt"))
        else:
            # 라벨이 없는 경우 빈 파일 생성 (배경 학습 및 평가용)
            open(os.path.join(save_lbl_dir, f"{file_name}.txt"), 'w').close()

# --- 경로 설정 및 실행 ---
root = r"D:\Parking Detection.v1i.yolov11"
output_root = os.path.join(root, "slicing_data")

# 1. Train 처리
print("\n--- Train 리사이징 시작 ---")
resize_data_recursive(
    img_base = os.path.join(root, "datasat", "train"),
    lbl_base = r"D:\Parking Detection.v1i.yolov11\datasat\Annotations\Annotations\Yolo\train",
    output_base = os.path.join(output_root, "train")
)

# 2. Valid 처리
print("\n--- Valid 리사이징 시작 ---")
resize_data_recursive(
    img_base = os.path.join(root, "datasat", "valid"),
    lbl_base = r"D:\Parking Detection.v1i.yolov11\datasat\Annotations\Annotations\Yolo\valid",
    output_base = os.path.join(output_root, "valid")
)

# 3. Test 처리 (추가됨)
print("\n--- Test 리사이징 시작 ---")
resize_data_recursive(
    img_base = r"D:\Parking Detection.v1i.yolov11\datasat\test\test",
    lbl_base = r"D:\Parking Detection.v1i.yolov11\datasat\Annotations\Annotations\Yolo\test",
    output_base = os.path.join(output_root, "test")
)

print(f"\n🏁 모든 데이터 전처리 완료! {output_root} 폴더를 확인하세요.")