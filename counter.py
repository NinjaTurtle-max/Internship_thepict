import os
from PIL import Image
from collections import Counter
from tqdm import tqdm

def analyze_resolutions(root_path):
    """
    [인수인계 노트] 
    목적: 데이터셋 내 모든 이미지의 해상도를 전수 조사합니다.
    이유: YOLO 모델은 입력 해상도에 민감하며, 특히 고해상도 항공 이미지의 경우 
         추후 SAHI(Sliced Aided Hyper Inference) 적용을 위한 슬라이싱 크기를 결정하는 기초 자료가 됩니다.
    """
    # 조사할 이미지 확장자 정의 (대소문자 구분 없이 처리하기 위해 lower()와 함께 사용)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.bmp')
    resolutions = []

    print(f"🔍 [시스템] 경로 분석 시작: {root_path}")

    # 1. 파일 목록 수집 단계
    image_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # 파일 확장자를 체크하여 이미지 파일만 선별
            if file.lower().endswith(valid_extensions):
                # 파일의 전체 경로를 생성하여 리스트에 추가
                image_files.append(os.path.join(root, file))

    # 예외 처리: 지정된 경로에 이미지가 없을 경우 프로세스 중단
    if not image_files:
        print(f"❌ [에러] 이미지를 찾을 수 없습니다. 경로를 다시 확인해주세요: {root_path}")
        return

    print(f"🚀 [진행] 총 {len(image_files)}개의 이미지 발견. 해상도 추출 중...")

    # 2. 해상도 추출 단계
    # tqdm을 사용하여 대용량 데이터 처리 시 진행 상황(Progress Bar) 시각화
    for img_path in tqdm(image_files, desc="이미지 분석 중"):
        try:
            # 파일을 실제로 열어 해상도를 읽음 (메모리 효율을 위해 open 후 바로 close 되도록 with문 사용)
            with Image.open(img_path) as img:
                # img.size는 (width, height) 형태의 튜플임
                resolutions.append(img.size) 
        except Exception as e:
            # 손상된 이미지 파일이 있을 경우 스킵하고 에러 메시지 기록
            print(f"⚠️ [주의] 파일 읽기 실패 ({os.path.basename(img_path)}): {e}")

    # 3. 데이터 통계 집계 단계
    # collections.Counter를 사용해 중복되는 해상도 조합을 빠르게 카운팅
    stats = Counter(resolutions)

    # 4. 분석 결과 출력 (가독성을 위한 포매팅)
    print("\n" + "="*50)
    print(f"📊 [데이터셋 분석 보고서]")
    print(f"📂 분석 대상: {root_path}")
    print("="*50)
    print(f"{'해상도 (가로 x 세로)':<30} | {'이미지 개수':<10}")
    print("-" * 50)
    
    # 빈도가 높은 해상도 순서대로 정렬하여 출력
    for res, count in stats.most_common():
        res_str = f"{res[0]} x {res[1]}"
        print(f"{res_str:<30} | {count:<10}개")
    print("="*50)

if __name__ == "__main__":
    # [설정] 분석하고자 하는 데이터셋의 루트 경로를 지정합니다.
    # r"..." (raw string) 형식을 사용하여 윈도우 경로의 백슬래시(\) 문자가 이스케이프 되는 것을 방지합니다.
    target_path = r"D:\Parking Detection.v1i.yolov11\high altitude\datasat" 
    
    analyze_resolutions(target_path)