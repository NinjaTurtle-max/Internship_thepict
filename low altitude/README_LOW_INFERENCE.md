🔍 Model Inference & Deployment Guide

학습 완료된 모델(best.pt)을 사용하여 실제 결과를 도출하는 가이드입니다.

실행 파일               추론 방식               추천 상황
predict.py            표준 512px        빠른 결과 확인 및 일반 탐지

predict_640.py        표준 640px      중고도 데이터의 객체 인식률 향상

predict_sahi.py      SAHI 슬라이싱    작은 차량 탐지가 필요한 항공 영상

predict_dual_scan.py  더블 스캔      최상의 정확도 및 노이즈 제거 필요 시

2. 핵심 로직 설명 (Dual-scan)

predict_shai_Dual-scan.py 다음 기술을 포함합니다.

Perform Standard Pred: 전체 이미지의 맥락과 조각 이미지의 디테일을 동시 탐색.

Noise Filter: 면적 150픽셀 이하의 탐지 결과(티끌 등)를 자동으로 제거하여 신뢰도 상승.

3. 결과 확인 방법

모든 결과는 runs/predict_filtered/ 폴더 내에 시각화 이미지와 함께 저장됩니다.

터미널 로그를 통해 이미지당 검출된 차량 대수를 요약 보고합니다.
