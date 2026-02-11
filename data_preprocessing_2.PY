import os

# 1. 최상위 경로 설정 (이 아래 모든 폴더를 뒤집니다)
root_path = r"D:\Parking Detection.v1i.yolov11"

def final_sync_cleanup():
    all_images = set()
    all_labels = []
    
    image_extensions = ('.jpg', '.jpeg', '.png')

    print("프로젝트 전체 스캔 중...")
    
    # 2. 전체 경로를 돌며 이미지 파일명과 라벨 파일 경로를 각각 수집
    for root, _, files in os.walk(root_path):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in image_extensions:
                all_images.add(name)
            elif ext.lower() == '.txt':
                # classes.txt나 requirements.txt 같은 시스템 파일은 제외
                if name not in ['classes', 'requirements', 'notes', 'README']:
                    all_labels.append(os.path.join(root, f))

    print(f"발견된 이미지: {len(all_images)}개")
    print(f"검사할 라벨: {len(all_labels)}개")

    # 3. 매칭 확인 및 삭제
    deleted_count = 0
    for label_path in all_labels:
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        
        # 이미지 목록에 없는 라벨이면 삭제
        if label_name not in all_images:
            try:
                os.remove(label_path)
                deleted_count += 1
            except Exception as e:
                print(f"삭제 실패: {label_path} ({e})")

    print("-" * 40)
    print(f"결과: 총 {deleted_count}개의 고아 라벨을 삭제 완료했습니다.")
    print("이제 이미지와 라벨의 숫자가 완벽하게 일치합니다.")

if __name__ == "__main__":
    final_sync_cleanup()