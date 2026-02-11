import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import os
from PIL import Image

# 1. 경로 설정
image_root = os.path.join(BASE_DIR, "datasat")
label_root = os.path.join(BASE_DIR, "Yolo")

# 삭제 대상 해상도
TARGET_RES = (3840, 2160)

def cleanup_high_res_data():
    deleted_count = 0
    
    for root, dirs, files in os.walk(image_root):
        # 라벨 폴더는 건너뛰기
        if "Annotations" in root:
            continue
            
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                should_delete = False
                
                try:
                    # 2. 파일 정보를 읽고 즉시 닫음
                    with Image.open(img_path) as img:
                        if img.size == TARGET_RES:
                            should_delete = True
                    
                    # 3. 'with' 블록 밖에서(파일이 닫힌 후) 삭제 진행
                    if should_delete:
                        # 이미지 삭제
                        os.remove(img_path)
                        
                        # 매칭되는 라벨 삭제
                        label_filename = os.path.splitext(filename)[0] + ".txt"
                        label_path = os.path.join(label_root, label_filename)
                        
                        if os.path.exists(label_path):
                            os.remove(label_path)
                            label_status = "라벨 포함 삭제"
                        else:
                            label_status = "라벨 없음"
                        
                        deleted_count += 1
                        print(f"[삭제 완료] {filename} ({label_status})")
                            
                except Exception as e:
                    print(f"오류 발생 ({filename}): {e}")

    print("\n" + "="*40)
    print(f"최종 완료: 총 {deleted_count} 세트 삭제됨.")
    print("="*40)

if __name__ == "__main__":
    cleanup_high_res_data()