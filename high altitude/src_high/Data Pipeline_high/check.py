import os
from pathlib import Path

def check_dataset_matching(root_path):
    # 1. 설정: 확인할 스플릿과 확장자
    splits = ['train', 'val', 'test']
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    root = Path(root_path)
    print(f"\n🔍 데이터셋 검증 시작: {root}")
    print("=" * 60)

    for split in splits:
        split_path = root / split
        img_dir = split_path / 'images'
        lbl_dir = split_path / 'labels'

        # 폴더 존재 여부 확인
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"⏩ [Skip] {split}: images 또는 labels 폴더가 존재하지 않습니다.")
            continue

        # 이미지와 라벨 파일 이름(확장자 제외) 추출
        img_files = {f.stem: f.suffix for f in img_dir.iterdir() if f.suffix.lower() in img_extensions}
        lbl_files = {f.stem for f in lbl_dir.iterdir() if f.suffix.lower() == '.txt'}

        img_names = set(img_files.keys())
        
        # 매칭 확인 로직
        only_in_images = img_names - lbl_files
        only_in_labels = lbl_files - img_names
        matched_count = len(img_names & lbl_files)

        print(f"[{split.upper()}] 세트 결과:")
        print(f"  - 정상 매칭: {matched_count}쌍")
        
        # 불일치 발생 시 출력
        if not only_in_images and not only_in_labels:
            print(f"  - ✅ 모든 파일이 완벽하게 매칭됩니다.")
        else:
            if only_in_images:
                print(f"  - ⚠️ 라벨 없음 (이미지만 존재): {len(only_in_images)}개")
                for name in list(only_in_images)[:5]: # 너무 많을 수 있으니 5개만 출력
                    print(f"    └ {name}{img_files[name]}")
                if len(only_in_images) > 5: print("    └ ... 외 더 있음")

            if only_in_labels:
                print(f"  - ⚠️ 이미지 없음 (라벨만 존재): {len(only_in_labels)}개")
                for name in list(only_in_labels)[:5]:
                    print(f"    └ {name}.txt")
                if len(only_in_labels) > 5: print("    └ ... 외 더 있음")
        print("-" * 60)

if __name__ == "__main__":
    # 사용자 데이터 경로
    dataset_path = r"D:\Parking Detection.v1i.yolov11\high altitude\140m_classified_result\high_alt_140m\sliced_640_dataset"
    
    check_dataset_matching(dataset_path)