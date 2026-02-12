import os

# 데이터셋 루트 경로 (이미지와 라벨이 섞여 있는 경우 대응)
root_path = r"D:\Parking Detection.v1i.yolov11\datasat"

def final_sync_cleanup():
    """
    [인수인계 노트]
    목적: 이미지 리스트를 기준으로, 짝이 없는 라벨(.txt) 파일을 전수 조사하여 삭제합니다.
    필요성: 전처리 과정에서 이미지만 삭제된 경우, 학습 시 에러를 방지하기 위해 필수적으로 수행합니다.
    """
    all_images = set()
    all_labels = []
    image_extensions = ('.jpg', '.jpeg', '.png')

    # 1. 전체 경로를 스캔하여 이미지 파일명(확장자 제외)과 라벨 파일 전체 경로 수집
    for root, _, files in os.walk(root_path):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in image_extensions:
                all_images.add(name) # 중복 체크를 위해 set 사용
            elif ext.lower() == '.txt':
                # 시스템 파일 제외 로직
                if name not in ['classes', 'requirements', 'notes', 'README']:
                    all_labels.append(os.path.join(root, f))

    # 2. 수집된 라벨 파일들을 돌며 대응하는 이미지가 있는지 대조
    deleted_count = 0
    for label_path in all_labels:
        label_name = os.path.splitext(os.path.basename(label_path))[0]
        
        # 이미지 셋(set)에 파일명이 없다면 삭제 대상
        if label_name not in all_images:
            try:
                os.remove(label_path)
                deleted_count += 1
            except Exception as e:
                print(f"❌ 삭제 실패: {label_path} ({e})")

    print(f"🧹 결과: 짝이 없는 라벨 {deleted_count}개 삭제 완료. 이미지-라벨 동기화 성공.")