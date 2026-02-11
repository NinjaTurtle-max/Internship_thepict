import cv2
import os
import numpy as np
from pathlib import Path

def slice_yolo_dataset(img_dir, label_dir, output_img_dir, output_label_dir, slice_size=512, overlap_ratio=0.2):
    # 출력 폴더 생성
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 이동 간격 계산 (오버랩 고려)
    stride = int(slice_size * (1 - overlap_ratio))
    
    # 지원하는 이미지 확장자
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    img_files = []
    for ext in img_extensions:
        img_files.extend(list(Path(img_dir).glob(ext)))

    print(f"총 {len(img_files)}개의 이미지를 처리합니다... (경로: {img_dir})")

    for img_path in img_files:
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        h, w, _ = image.shape
        img_name = img_path.stem
        
        # 대응하는 라벨 파일 찾기
        label_path = Path(label_dir) / f"{img_name}.txt"
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = [line.split() for line in f.readlines()]

        # 슬라이딩 윈도우 루프
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 윈도우가 이미지 경계를 벗어나지 않도록 조정 (자투리 처리)
                y_end = min(y + slice_size, h)
                x_end = min(x + slice_size, w)
                y_start = max(0, y_end - slice_size)
                x_start = max(0, x_end - slice_size)
                
                # 이미지 크롭
                crop = image[y_start:y_end, x_start:x_end]
                slice_suffix = f"_{y_start}_{x_start}"
                slice_name = f"{img_name}{slice_suffix}"
                
                # 라벨 재계산
                new_labels = []
                for label in labels:
                    cls, x_c, y_c, bw, bh = map(float, label)
                    
                    # 1. 정규화 좌표 -> 원본 픽셀 좌표
                    px_c, py_c = x_c * w, y_c * h
                    pbw, pbh = bw * w, bh * h
                    
                    # 2. 크롭된 영역 기준의 상대 좌표로 변환
                    nx_c = px_c - x_start
                    ny_c = py_c - y_start
                    
                    # 3. 객체의 중심점이 현재 크롭 영역 안에 있는지 확인
                    if 0 <= nx_c <= slice_size and 0 <= ny_c <= slice_size:
                        # 4. 다시 512 기준으로 정규화 (0~1)
                        new_labels.append(f"{int(cls)} {nx_c/slice_size:.6f} {ny_c/slice_size:.6f} {pbw/slice_size:.6f} {pbh/slice_size:.6f}")
                
                # 객체가 있는 조각만 저장하거나, 빈 조각도 포함하려면 아래 조건문 수정
                # 여기서는 객체가 있는 조각만 저장하여 데이터 효율을 높입니다.
                if new_labels:
                    cv2.imwrite(os.path.join(output_img_dir, f"{slice_name}.jpg"), crop)
                    with open(os.path.join(output_label_dir, f"{slice_name}.txt"), 'w') as f:
                        f.write("\n".join(new_labels))

# --- 설정 구간 ---
base_path = r"D:\Parking Detection.v1i.yolov11\datasat"
output_base = r"D:\Parking Detection.v1i.yolov11\sliced_dataset"

# 폴더명 매핑 (스크린샷 기준: train_labe, valid_labe 등 오타 반영)
sub_folders = {
    'train': {'img': 'train', 'lbl': 'train_labe'},
    'valid': {'img': 'valid', 'lbl': 'valid_labe'},
    'test': {'img': 'test', 'lbl': 'test_label'}
}

for split, folders in sub_folders.items():
    print(f"\n--- {split} 데이터 처리 시작 ---")
    slice_yolo_dataset(
        img_dir=os.path.join(base_path, split, folders['img']),
        label_dir=os.path.join(base_path, split, folders['lbl']),
        output_img_dir=os.path.join(output_base, split, "images"),
        output_label_dir=os.path.join(output_base, split, "labels"),
        slice_size=512,
        overlap_ratio=0.2 # 20% 겹침
    )

print("\n작업이 완료되었습니다!")