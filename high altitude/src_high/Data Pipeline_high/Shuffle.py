import os
import random
import shutil
from tqdm import tqdm

"""
    [인수인계 노트]
    목적: 분류된 140m 데이터를 모델 학습용으로 랜덤하게 섞고 분할합니다.
    핵심: random.seed(42)를 사용하여 매 실행 시 동일한 분할 결과를 보장(실험 재현성)합니다.
    """

def split_yolo_dataset_random(src_img_dir, src_label_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 1. 출력 경로 설정 (기존 폴더와 겹치지 않게 random_split 폴더 생성)
    parent_dir = os.path.dirname(src_img_dir.rstrip(os.sep))
    output_root = os.path.join(parent_dir, "random_split_dataset")
    
    sets = ['train', 'val', 'test']
    for s in sets:
        os.makedirs(os.path.join(output_root, s, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, s, 'labels'), exist_ok=True)

    # 2. 파일 목록 수집 및 랜덤 셔플
    valid_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(valid_ext)]
    
    # [지리지 포인트] 시드 고정 및 랜덤 셔플
    # seed를 고정하면 나중에 다시 돌려도 똑같이 섞여서 실험 결과 비교가 가능합니다.
    random.seed(42) 
    random.shuffle(all_images) 

    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 3. 데이터 배분 리스트 생성
    data_split = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    print(f"🎲 랜덤 분할 시작 (총 {total}세트)")
    print(f"📊 Train: {len(data_split['train'])}, Val: {len(data_split['val'])}, Test: {len(data_split['test'])}")

    # 4. 파일 복사 실행
    for split_name, image_list in data_split.items():
        for img_name in tqdm(image_list, desc=f"Copying {split_name}"):
            # 이미지/라벨 파일명 매칭
            img_base = os.path.splitext(img_name)[0]
            label_name = img_base + ".txt"

            src_img_path = os.path.join(src_img_dir, img_name)
            src_label_path = os.path.join(src_label_dir, label_name)

            if os.path.exists(src_label_path):
                # 이미지 복사
                shutil.copy2(src_img_path, os.path.join(output_root, split_name, 'images', img_name))
                # 라벨 복사
                shutil.copy2(src_label_path, os.path.join(output_root, split_name, 'labels', label_name))

    print("\n" + "="*50)
    print(f"✅ 랜덤 데이터셋 분할 완료!")
    print(f"📂 저장 경로: {output_root}")
    print("="*50)

# --- 실행부 (현우님 140m 분류 경로) ---
IMG_PATH = r"D:\Parking Detection.v1i.yolov11\high altitude\140m_classified_result\high_alt_140m\images"
LBL_PATH = r"D:\Parking Detection.v1i.yolov11\high altitude\140m_classified_result\high_alt_140m\labels"

split_yolo_dataset_random(IMG_PATH, LBL_PATH)