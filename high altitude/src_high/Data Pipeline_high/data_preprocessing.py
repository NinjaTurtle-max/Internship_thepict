import os
import glob
from tqdm import tqdm

"""
    [인수인계 노트]
    목적: Car(0), Bus(1), Truck(2) 등으로 나뉜 클래스를 '0'번 클래스(통합 차량)로 합칩니다.
    이유: 주차 탐지 시 차종 구분보다 '주차 여부' 자체에 집중하기 위함입니다.
    """


def merge_vehicle_classes(directory):
    # 하위 모든 폴더(train, val, test)의 .txt 파일 탐색
    # recursive=True를 통해 images와 혼동되지 않게 labels 폴더 깊숙이 들어갑니다.
    txt_files = glob.glob(os.path.join(directory, "**", "labels", "*.txt"), recursive=True)
    
    modified_count = 0
    total_files = 0

    if not txt_files:
        print(f"❌ 라벨 파일을 찾을 수 없습니다. 경로를 확인하세요: {directory}")
        return

    print(f"🚀 {os.path.basename(directory)} 클래스 통합 시작...")

    for file_path in tqdm(txt_files, desc="Processing Labels"):
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
                # [지리지 포인트] 클래스 ID 1(Bus), 2(Truck) 등을 0(Vehicle)으로 통일
                if parts[0] in ['1', '2', '3']: # 혹시 3번까지 있다면 추가
                    parts[0] = '0'
                    new_lines.append(" ".join(parts) + "\n")
                    is_modified = True
                else:
                    new_lines.append(line)
        
        if is_modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            modified_count += 1

    print("\n" + "="*50)
    print(f"✅ 클래스 통합 완료 리포트")
    print(f"📂 대상 경로: {directory}")
    print(f"📝 탐색한 라벨 파일: {total_files}개")
    print(f"🔄 수정된 파일: {modified_count}개")
    print(f"🚀 결과: 모든 차량 객체 -> 클래스 '0'으로 단일화")
    print("="*50)

if __name__ == "__main__":
    # 현우님이 요청하신 140m 랜덤 분할 데이터셋 경로
    target_path = r"D:\Parking Detection.v1i.yolov11\high altitude\140m_classified_result\high_alt_140m\random_split_dataset"
    merge_vehicle_classes(target_path)