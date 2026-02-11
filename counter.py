import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import os
from PIL import Image
from collections import Counter
from tqdm import tqdm

def analyze_resolutions(root_path):
    # 조사할 이미지 확장자
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.bmp')
    resolutions = []

    print(f"🔍 경로 분석 시작: {root_path}")

    # 모든 하위 폴더를 순회하며 이미지 탐색
    image_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("❌ 이미지를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    print(f"🚀 총 {len(image_files)}개의 이미지 발견. 해상도 추출 중...")

    # 해상도 추출
    for img_path in tqdm(image_files, desc="분석 중"):
        try:
            with Image.open(img_path) as img:
                resolutions.append(img.size) # (width, height) 튜플 저장
        except Exception as e:
            print(f"⚠️ 에러 발생 ({os.path.basename(img_path)}): {e}")

    # 결과 집계
    stats = Counter(resolutions)

    print("\n" + "="*40)
    print(f"📊 {os.path.basename(root_path)} 해상도 분석 결과")
    print("="*40)
    print(f"{'해상도 (가로 x 세로)':<25} | {'개수':<10}")
    print("-" * 40)
    
    # 개수가 많은 순서대로 정렬하여 출력
    for res, count in stats.most_common():
        res_str = f"{res[0]} x {res[1]}"
        print(f"{res_str:<25} | {count:<10}개")
    print("="*40)

# --- 실행부 ---
target_path = os.path.join(BASE_DIR, "sliced_dataset")
analyze_resolutions(target_path)