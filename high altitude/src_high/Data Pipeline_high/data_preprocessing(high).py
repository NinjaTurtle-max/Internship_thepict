import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 한글 경로 지원을 위한 imwrite 대체 함수
def imwrite_korean(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='wb') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(f"저장 에러: {e}")
        return False

def slice_yolo_dataset(img_dir, label_dir, output_img_dir, output_label_dir, slice_size=640, overlap_ratio=0.2):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    stride = int(slice_size * (1 - overlap_ratio))
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
    img_files = []
    for ext in img_extensions:
        img_files.extend(list(Path(img_dir).glob(ext)))

    if not img_files:
        print(f"⚠️ 경고: {img_dir}에 이미지가 없습니다.")
        return

    saved_count = 0
    for img_path in tqdm(img_files, desc=f"Slicing {os.path.basename(img_dir)}"):
        # 한글 경로 대응 imread
        img_array = np.fromfile(str(img_path), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None: continue
        h, w, _ = image.shape
        img_name = img_path.stem
        
        label_path = Path(label_dir) / f"{img_name}.txt"
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = [line.split() for line in f.readlines()]

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + slice_size, h)
                x_end = min(x + slice_size, w)
                y_start = max(0, y_end - slice_size)
                x_start = max(0, x_end - slice_size)
                
                crop = image[y_start:y_end, x_start:x_end]
                slice_suffix = f"_{y_start}_{x_start}"
                slice_name = f"{img_name}{slice_suffix}"
                
                new_labels = []
                for label in labels:
                    cls, x_c, y_c, bw, bh = map(float, label)
                    px_c, py_c = x_c * w, y_c * h
                    pbw, pbh = bw * w, bh * h
                    
                    nx_c = px_c - x_start
                    ny_c = py_c - y_start
                    
                    if 0 <= nx_c <= slice_size and 0 <= ny_c <= slice_size:
                        new_labels.append(f"{int(cls)} {nx_c/slice_size:.6f} {ny_c/slice_size:.6f} {pbw/slice_size:.6f} {pbh/slice_size:.6f}")
                
                if new_labels:
                    img_output_path = os.path.join(output_img_dir, f"{slice_name}.jpg")
                    # 수정된 저장 함수 사용
                    if imwrite_korean(img_output_path, crop):
                        saved_count += 1
                        with open(os.path.join(output_label_dir, f"{slice_name}.txt"), 'w') as f:
                            f.write("\n".join(new_labels))

    print(f"✅ {img_dir} 처리 완료: {saved_count}개 조각 저장됨")

# --- 설정 구간 (영문 경로 권장하지만, 위 함수로 한글도 대응 가능) ---
base_path = r"D:\Parking Detection.v1i.yolov11\high altitude\140m_classified_result\high_alt_140m\random_split_dataset"
# 가능하면 C:\sliced_dataset 처럼 단순한 경로로 바꾸는 것을 추천합니다.
output_base = r"C:\Users\vrro3\OneDrive\바탕화~1-DESKTOP-NJ6COUG-23615\sliced_640_dataset"

sub_folders = ['train', 'val', 'test']

for split in sub_folders:
    slice_yolo_dataset(
        img_dir=os.path.join(base_path, split, "images"),
        label_dir=os.path.join(base_path, split, "labels"),
        output_img_dir=os.path.join(output_base, split, "images"),
        output_label_dir=os.path.join(output_base, split, "labels"),
        slice_size=640,
        overlap_ratio=0.2
    )