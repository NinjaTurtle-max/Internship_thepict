import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

"""
    [인수인계 노트]
    목적: 고해상도 원본 영상을 640x640 패치로 자르고 좌표를 재계산합니다.
    수학적 변환:
        1. 원본 픽셀 좌표 복원: $px = x_{center} \times Width_{original}$
        2. 슬라이스 내 상대 좌표: $nx = px - x_{start}$
        3. 슬라이스 기준 정규화: $x'_{center} = nx / slice\_size$
    """

def slice_yolo_dataset(img_dir, label_dir, output_img_dir, output_label_dir, slice_size=640, overlap_ratio=0.2):
    # 출력 폴더 생성
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 이동 간격 계산 (오버랩 고려)
    stride = int(slice_size * (1 - overlap_ratio))
    
    # 지원하는 이미지 확장자
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
    img_files = []
    for ext in img_extensions:
        img_files.extend(list(Path(img_dir).glob(ext)))

    print(f"🚀 이미지 조각내기 시작: {len(img_files)}개 (크기: {slice_size}, 오버랩: {overlap_ratio})")

    for img_path in tqdm(img_files, desc=f"Slicing {os.path.basename(img_dir)}"):
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
                        # 4. 다시 slice_size(640) 기준으로 정규화 (0~1)
                        new_labels.append(f"{int(cls)} {nx_c/slice_size:.6f} {ny_c/slice_size:.6f} {pbw/slice_size:.6f} {pbh/slice_size:.6f}")
                
                # [지리지 포인트] 객체가 있는 조각만 저장하여 학습 효율 극대화
                if new_labels:
                    cv2.imwrite(os.path.join(output_img_dir, f"{slice_name}.jpg"), crop)
                    with open(os.path.join(output_label_dir, f"{slice_name}.txt"), 'w') as f:
                        f.write("\n".join(new_labels))

# --- 설정 구간 (현우님의 140m 랜덤 분할 경로 적용) ---
base_path = r"D:\Parking Detection.v1i.yolov11\high altitude\140m_classified_result\high_alt_140m\random_split_dataset"
output_base = r"D:\Parking Detection.v1i.yolov11\high altitude\140m_classified_result\high_alt_140m\sliced_640_dataset"

# 우리 랜덤 분할 스크립트가 만든 폴더 구조 (images / labels)
sub_folders = ['train', 'val', 'test']

for split in sub_folders:
    print(f"\n--- {split} 데이터 처리 중... ---")
    slice_yolo_dataset(
        img_dir=os.path.join(base_path, split, "images"),
        label_dir=os.path.join(base_path, split, "labels"),
        output_img_dir=os.path.join(output_base, split, "images"),
        output_label_dir=os.path.join(output_base, split, "labels"),
        slice_size=640, # 640으로 변경!
        overlap_ratio=0.2
    )

print("\n🎉 모든 데이터의 640 슬라이싱 작업이 완료되었습니다!")
print(f"📁 결과 확인: {output_base}")