import os
import shutil
from tqdm import tqdm

"""
    [인수인계 노트]
    목적: 객체의 크기를 통해 고도를 역산하여 데이터를 분류합니다.
    논리: 고도가 높을수록(140m) 이미지 내 차량의 상대적 면적(YOLO 정규화 좌표 기준 w * h)은 작아집니다.
    기준: 면적 임계값 0.0005를 기준으로 고고도(140m급)와 저고도를 구분합니다.
    """

def classify_recursive_140m(img_root, label_root, threshold=0.0005):
    """
    img_root: 이미지 최상위 폴더
    label_root: 라벨 최상위 폴더
    threshold: 140m 고도(Area 0.00043)를 기준으로 한 분류 임계값
    """
    # 1. 출력 경로 설정 (이미지 경로 상위에 '140m_classified_result' 생성)
    parent_dir = os.path.dirname(img_root.rstrip(os.sep))
    output_root = os.path.join(parent_dir, "140m_classified_result")
    
    for category in ["high_alt_140m", "low_alt_under_140m"]:
        os.makedirs(os.path.join(output_root, category, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root, category, "labels"), exist_ok=True)

    print("🔍 [1/3] 하위 폴더 전수 조사 중...")
    
    # 이미지 파일 경로 매핑 (파일명: 전체경로)
    img_map = {}
    valid_img_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.bmp')
    for root, _, files in os.walk(img_root):
        for f in files:
            if f.lower().endswith(valid_img_ext):
                name = os.path.splitext(f)[0]
                img_map[name] = os.path.join(root, f)

    # 라벨 파일 경로 수집
    label_list = []
    for root, _, files in os.walk(label_root):
        for f in files:
            if f.endswith(".txt") and f != "classes.txt":
                label_list.append(os.path.join(root, f))

    if not label_list:
        print(f"❌ 라벨 파일을 찾을 수 없습니다: {label_root}")
        return

    print(f"🚀 [2/3] 총 {len(label_list)}개의 데이터 세트 분석 및 분류 시작...")
    print(f"📏 기준: 4032 해상도 / 고도 140m (Threshold: {threshold})")

    high_count = 0
    low_count = 0

    # 3. 분석 및 복사
    for lb_path in tqdm(label_list, desc="분류 진행 중"):
        lb_name = os.path.basename(lb_path)
        img_base = os.path.splitext(lb_name)[0]

        if img_base not in img_map:
            continue

        try:
            with open(lb_path, 'r') as f:
                lines = f.readlines()
                if not lines: continue
                
                areas = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        # YOLO: class x y w h
                        w, h = float(parts[3]), float(parts[4])
                        areas.append(w * h)
                
                if not areas: continue
                # 평균 면적 계산
                avg_area = sum(areas) / len(areas)
        except Exception as e:
            print(f"⚠️ {lb_name} 분석 에러: {e}")
            continue

        # [분류 결정] 140m급(0.00043) 근처면 high_alt로 분류
        target_sub = "high_alt_140m" if avg_area < threshold else "low_alt_under_140m"
        
        if target_sub == "high_alt_140m":
            high_count += 1
        else:
            low_count += 1

        # 파일 복사
        img_src_path = img_map[img_base]
        img_ext = os.path.splitext(img_src_path)[1]
        
        shutil.copy2(img_src_path, os.path.join(output_root, target_sub, "images", img_base + img_ext))
        shutil.copy2(lb_path, os.path.join(output_root, target_sub, "labels", lb_name))

    print("\n" + "="*50)
    print(f"✅ [3/3] 전수 조사 및 분류 완료!")
    print(f"📂 결과 폴더: {output_root}")
    print(f"🛰️ 고고도 (140m급): {high_count}세트")
    print(f"🚁 저고도/중고도: {low_count}세트")
    print("="*50)

# --- 실행부 ---
IMG_PATH = r"D:\Parking Detection.v1i.yolov11\high altitude\datasat"
LBL_PATH = r"D:\Parking Detection.v1i.yolov11\high altitude\datasat\Annotations\Annotations\Yolo"

# 고도 140m 실증 데이터에 최적화된 임계값 0.0005 적용
classify_recursive_140m(IMG_PATH, LBL_PATH, threshold=0.0005)