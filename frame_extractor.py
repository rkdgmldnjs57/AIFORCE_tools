import cv2
import numpy as np
import os
import time

mtx = np.array([[624.021794, 0, 705.539195],
                [0, 624.719173, 398.307132],
                [0, 0, 1]])
dist = np.array([[-0.318379, 0.108202, -0.000758, 0.000421, -0.016728]])

def calibrate(img):
    h,  w = img.shape[:2]
    start = time.time()

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y+250:y+h, x:x+w]
    #dst = dst[250:, :]
    end = time.time()
    print(end-start)
    print(dst.shape)

    return dst

def extract_frames(video_path, output_folder):
    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)

    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 비디오 정보 가져오기
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"총 프레임 수: {frame_count}, FPS: {fps}")

    # 프레임 추출
    frame_number = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 960))

        frame = calibrate(frame)

        # 각 프레임을 이미지로 저장
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_number += 1

        if frame_number % 100 == 0:
            print(f"{frame_number}/{frame_count} 프레임 저장 완료")

    video.release()
    print("모든 프레임 추출 완료!")

# 사용 예시
video_file = "vlog_2319.avi"  # 비디오 파일 경로
output_dir = "output_frames"        # 추출된 프레임 저장 폴더
extract_frames(video_file, output_dir)
