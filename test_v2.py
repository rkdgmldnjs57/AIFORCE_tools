# test dataset을 가져와 추론을 실행한 후 16개의 배치로 묶어 출력하는 코드입니다.
# model_path, folder_path, output_dir을 수정해주세요.

import cv2
import os
import numpy as np
from ultralytics import YOLO, RTDETR
from matplotlib import pyplot as plt
import math


# 주어진 모델로 이미지를 처리하고 결과를 출력하는 함수
def process_images(
    model_path,
    folder_path,
    output_dir="/content/drive/MyDrive/aiforce/241123/rtdetr-alldata/cho_test",
    is_show=False,
):

    model = RTDETR(model_path)
    # model = YOLO(model_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 폴더 내의 모든 jpg 파일 경로 가져오기
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".jpg")
    ]

    batch_size = 16
    num_batches = math.ceil(len(image_paths) / batch_size)

    # 배치마다 처리
    for batch_idx in range(num_batches):
        # 현재 배치의 이미지 리스트
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
        batch_images = image_paths[start_idx:end_idx]

        # 현재 배치에 해당하는 이미지 처리
        annotated_images = []

        for image_path in batch_images:
            image = cv2.imread(image_path)

            # YOLO 추론 실행
            results = model(image, conf=0.1)

            annotated_image = results[0].plot()

            annotated_images.append(annotated_image)

        grid_size = int(math.ceil(math.sqrt(len(annotated_images))))

        # 그리드 레이아웃으로 이미지 출력
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

        if grid_size == 1:
            axs = np.array([[axs]])  # Convert to 2D array for consistency

        # 이미지를 그리드에 배치
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                if index < len(annotated_images):
                    axs[i, j].imshow(
                        cv2.cvtColor(annotated_images[index], cv2.COLOR_BGR2RGB)
                    )
                    axs[i, j].axis("off")
                else:
                    axs[i, j].axis("off")

        # 배치 출력 파일 경로 설정
        output_path = os.path.join(output_dir, f"batch_{batch_idx + 1}_output.png")
        plt.tight_layout()
        plt.savefig(output_path)
        if is_show:
            plt.show()
        plt.close()

        print(f"배치 {batch_idx + 1} 출력 이미지가 {output_path}에 저장되었습니다.")


# 모델 경로와 이미지가 포함된 폴더 경로 설정
model_path = "path/to/best.pt"  # 모델 경로 설정
folder_path = "path/to/test/images"  # 폴더 경로에 있는 .jpg 파일들을 처리

# 폴더 내 이미지를 배치로 처리
process_images(model_path, folder_path)
