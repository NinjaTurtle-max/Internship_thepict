🚁 Low Altitude (Standard) Pipeline

저고도 및 표준 고도 데이터를 정제하고 학습하기 위한 파이프라인입니다.

🚀 파이프라인 실행 순서

Step 1. 고해상도 필터링 (data_preprocessing_1.py)

역할: 학습에 부적합한 4K(3840x2160) 이미지 및 매칭 라벨 일괄 삭제.

이유: 데이터셋의 규격 통일성 및 학습 속도 향상.

Step 2. 정합성 동기화 (data_preprocessing_2.py)

역할: 이미지가 없는 '유령 라벨' 파일 검색 및 삭제.

필요성: 학습 중 발생할 수 있는 File Not Found 에러 사전 차단.

Step 3. 512px 표준 슬라이싱 (data_preprocessing_3.py)

역할: 이미지를 512x512 크기로 분할하고 좌표 재계산.

특정 사항: 저고도는 객체가 크므로 512 사이즈가 효율적입니다.

Step 4. 모델 학습 (train_low.py)

설정: data_low.yaml을 참조하여 학습 진행.

특이점: 윈도우 환경 에러 방지를 위한 if __name__ == '__main__': 구문 필수 포함.

📂 주요 경로 (D: 드라이브 기준)

원본: ...\datasat

전처리 결과: ...\low altitude\sliced_dataset

설정 파일: data_low.yaml