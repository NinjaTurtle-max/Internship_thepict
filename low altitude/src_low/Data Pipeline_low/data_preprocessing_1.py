import os
from PIL import Image

# 이미지와 라벨의 루트 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_root = os.path.join(BASE_DIR, "datasat")
label_root = os.path.join(BASE_DIR, "Yolo")

# 필터링 대상: 4K 해상도 (3840x2160)
TARGET_RES = (3840, 2160)

def cleanup_high_res_data():
    """
    [인수인계 노트]
    목적: 특정 해상도(3840x2160)를 가진 이미지를 데이터셋에서 제외합니다.
    주의: 이미지 삭제 시 대응하는 라벨(.txt) 파일도 반드시 함께 삭제하여 정합성을 유지합니다.
    """
    deleted_count = 0
    
    for root, dirs, files in os.walk(image_root):
        if "Annotations" in root: continue # 주석 폴더 제외
            
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                should_delete = False
                
                try:
                    # 이미지의 헤더 정보만 읽어 해상도 확인
                    with Image.open(img_path) as img:
                        if img.size == TARGET_RES:
                            should_delete = True
                    
                    if should_delete:
                        # 1. 이미지 파일 삭제
                        os.remove(img_path)
                        
                        # 2. 매칭되는 라벨 파일명 생성 및 존재 확인 후 삭제
                        label_filename = os.path.splitext(filename)[0] + ".txt"
                        label_path = os.path.join(label_root, label_filename)
                        
                        if os.path.exists(label_path):
                            os.remove(label_path)
                            label_status = "라벨 동시 삭제"
                        else:
                            label_status = "이미지만 삭제 (라벨 없음)"
                        
                        deleted_count += 1
                        print(f"[🗑️ 삭제] {filename} ({label_status})")
                            
                except Exception as e:
                    print(f"❌ 오류 발생 ({filename}): {e}")

    print(f"✅ 최종 완료: {deleted_count} 세트의 데이터가 필터링되었습니다.")