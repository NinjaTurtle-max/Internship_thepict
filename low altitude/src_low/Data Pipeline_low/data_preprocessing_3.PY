import cv2
import os
import numpy as np
from pathlib import Path

def slice_yolo_dataset(img_dir, label_dir, output_img_dir, output_label_dir, slice_size=512, overlap_ratio=0.2):
    """
    [인수인계 노트]
    목적: 고해상도 이미지를 512x512 크기로 자르고, 그에 맞춰 YOLO 라벨(객체 좌표)을 재생성합니다.
    핵심 로직:
        1. 원본 정규화 좌표($x_c, y_c$)를 픽셀 좌표로 복원
        2. 크롭 영역 시작점($x_{start}, y_{start}$) 기준 상대 좌표로 변환
        3. 변환된 좌표가 크롭 영역 내에 있는지 판별
        4. 다시 512 기준 정규화 좌표로 저장
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 이동 간격 (Stride) = 슬라이스 크기 - 겹치는 영역
    stride = int(slice_size * (1 - overlap_ratio))
    
    img_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_files.extend(list(Path(img_dir).glob(ext)))

    for img_path in img_files:
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        h, w, _ = image.shape
        img_name = img_path.stem
        
        # 대응 라벨 로드
        label_path = Path(label_dir) / f"{img_name}.txt"
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = [line.split() for line in f.readlines()]

        # 슬라이딩 윈도우 루프
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 이미지 경계 처리 (자투리 영역 방지)
                y_end, x_end = min(y + slice_size, h), min(x + slice_size, w)
                y_start, x_start = max(0, y_end - slice_size), max(0, x_end - slice_size)
                
                crop = image[y_start:y_end, x_start:x_end]
                slice_name = f"{img_name}_{y_start}_{x_start}"
                
                new_labels = []
                for label in labels:
                    cls, x_c, y_c, bw, bh = map(float, label)
                    
                    # 좌표 변환 수식:
                    # $px_{pixel} = x_c \times W_{original}$
                    # $nx_{relative} = px_{pixel} - x_{start}$
                    px_c, py_c = x_c * w, y_c * h
                    nx_c, ny_c = px_c - x_start, py_c - y_start
                    
                    # 객체의 중심점이 크롭 영역 내부일 때만 새 라벨로 추가
                    if 0 <= nx_c <= slice_size and 0 <= ny_c <= slice_size:
                        new_labels.append(f"{int(cls)} {nx_c/slice_size:.6f} {ny_c/slice_size:.6f} {(bw*w)/slice_size:.6f} {(bh*h)/slice_size:.6f}")
                
                # 데이터 효율을 위해 객체가 하나라도 존재하는 조각만 저장
                if new_labels:
                    cv2.imwrite(os.path.join(output_img_dir, f"{slice_name}.jpg"), crop)
                    with open(os.path.join(output_label_dir, f"{slice_name}.txt"), 'w') as f:
                        f.write("\n".join(new_labels))