import os
import glob

# 스크립트 파일이 위치한 경로를 기준으로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(BASE_DIR, "Yolo")

def merge_vehicle_classes(directory):
    """
    [인수인계 노트]
    목적: Car(0), Bus(1), Truck(2) 등으로 나뉜 클래스를 '0'번 클래스(통합 차량)로 합칩니다.
    이유: 주차 탐지 시 차종 구분보다 '주차 여부' 자체에 집중하기 위함입니다.
    """
    # 1. 재귀적으로 모든 하위 폴더의 .txt 라벨 파일을 탐색
    txt_files = glob.glob(os.path.join(directory, "**", "*.txt"), recursive=True)
    
    modified_count = 0
    total_files = 0

    for file_path in txt_files:
        # 설정 파일인 classes.txt는 수정 대상에서 제외
        if os.path.basename(file_path) == 'classes.txt':
            continue
        
        total_files += 1
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        is_modified = False

        # 2. 라벨 파일의 각 줄(객체 1개)을 확인
        for line in lines:
            parts = line.split()
            if len(parts) > 0:
                # 클래스 ID가 '1'(Bus) 또는 '2'(Truck)인 경우 '0'(Car/Vehicle)으로 변경
                if parts[0] in ['1', '2']:
                    parts[0] = '0'
                    new_lines.append(" ".join(parts) + "\n")
                    is_modified = True
                else:
                    new_lines.append(line)
        
        # 3. 실제로 내용이 변경된 파일만 덮어쓰기 수행 (I/O 효율화)
        if is_modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            modified_count += 1

    print(f"🚀 결과: {modified_count}/{total_files} 개의 파일에서 클래스 통합 완료 (Car, Bus, Truck -> 0)")

if __name__ == "__main__":
    merge_vehicle_classes(base_path)