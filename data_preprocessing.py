import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import glob
import os

# 1. 경로 설정
base_path = os.path.join(BASE_DIR, "Yolo")

def merge_vehicle_classes(directory):
    # 모든 하위 폴더의 .txt 파일 탐색
    txt_files = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)
    
    modified_count = 0
    total_files = 0

    for file_path in txt_files:
        # classes.txt는 수정 대상에서 제외
        if os.path.basename(file_path) == 'classes.txt':
            continue
        
        total_files += 1
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        is_modified = False

        for line in lines:
            parts = line.split()
            if len(parts) > 0:
                # 클래스 ID(첫 번째 값)가 '1' 또는 '2'인 경우 '0'으로 변경
                if parts[0] in ['1', '2']:
                    parts[0] = '0'
                    new_lines.append(" ".join(parts) + "\n")
                    is_modified = True
                else:
                    new_lines.append(line)
        
        # 변경사항이 있는 경우에만 파일 다시 쓰기
        if is_modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            modified_count += 1

    print("\n" + "="*50)
    print(f"✅ 작업 완료 리포트")
    print(f"📂 대상 경로: {directory}")
    print(f"📝 탐색한 라벨 파일: {total_files}개")
    print(f"🔄 클래스 수정된 파일: {modified_count}개")
    print(f"🚀 Car, Bus, Truck -> 단일 클래스(0)로 통합 완료")
    print("="*50)

if __name__ == "__main__":
    # 실행 전 데이터 백업을 권장합니다!
    merge_vehicle_classes(base_path)