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

# 사용 예시 - 수정된 main 부분
def process_multiple_videos(video_folder, output_base_dir):
    # 비디오 폴더에서 모든 파일 가져오기
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.avi', '.mp4', '.mkv'))]

    # 비디오별로 프레임 추출
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        
        # 비디오 이름을 기준으로 출력 폴더 생성
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_base_dir, video_name)
        
        print(f"Processing video: {video_file}")
        extract_frames(video_path, output_dir)

    print("모든 비디오 처리 완료!")

# 비디오가 저장된 폴더 경로와 프레임이 저장될 기본 폴더 경로 설정
video_folder = "real_videos_2"  # 비디오 파일이 있는 폴더
output_base_dir = "extracted_frames"  # 프레임 저장될 기본 폴더

process_multiple_videos(video_folder, output_base_dir)