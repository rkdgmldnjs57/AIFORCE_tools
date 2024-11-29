import cv2
import numpy as np

def match_histogram(source_path, reference_path):
    # 이미지를 읽어오기
    source = cv2.imread(source_path)
    reference = cv2.imread(reference_path)

    # BGR에서 LAB 색상 공간으로 변환
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    # L 채널 히스토그램 평활화
    source_l = source_lab[:, :, 0]
    reference_l = reference_lab[:, :, 0]

    # 히스토그램 매칭을 직접 구현
    matched_l = cv2.equalizeHist(source_l)

    # 수정된 L 채널을 원래 LAB 이미지에 적용
    source_lab[:, :, 0] = matched_l

    # LAB에서 다시 BGR로 변환
    matched = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

    return matched

# 사용 예시
source_path = "output_frames/frame_00000.jpg"
reference_path = "output_frames/frame_01372.jpg"
output = match_histogram(source_path, reference_path)

# 결과 저장
cv2.imwrite("output_frames/matched.jpg", output)
