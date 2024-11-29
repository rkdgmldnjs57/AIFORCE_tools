import cv2
import numpy as np
import os
import time

mtx = np.array([[624.021794, 0, 705.539195],
                [0, 624.719173, 398.307132],
                [0, 0, 1]])
dist = np.array([[-0.318379, 0.108202, -0.000758, 0.000421, -0.016728]])

roi = (0, 0, 1279, 959)

def calibrate(img):
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, mtx) #newcameramtx==mtx
    # crop the image
    x, y, w, h = roi
    dst = dst[y+250:y+h, x:x+w]

    return dst

"""
[[624.021794   0.       705.539195]
 [  0.       624.719173 398.307132]
 [  0.         0.         1.      ]] [[-0.318379  0.108202 -0.000758  0.000421 -0.016728]]
"""

# def calibrate(img):
#     h,  w = img.shape[:2]
#     start = time.time()

#     #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

#     #print(newcameramtx, roi)

#     # undistort
#     dst = cv2.undistort(img, mtx, dist, None, mtx) #newcameramtx==mtx
    
#     # crop the image
#     x, y, w, h = roi
#     dst = dst[y+250:y+h, x:x+w]
#     #dst = dst[250:, :]
#     end = time.time()
#     print(end-start)
#     print(dst.shape)

#     return dst

#input_folder/A/img..., input_folder/B/img...
def calibrate_dataset(input_folder): 
    # Input and output folder paths
    output_folder = "calibrated_" + input_folder  # 결과 저장 상위 폴더

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through all subdirectories
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # 지원되는 이미지 확장자
                input_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, input_folder)  # 상대 경로
                output_subfolder = os.path.join(output_folder, rel_path)

                # Create output subfolder if it doesn't exist
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # Load and calibrate image
                img = cv2.imread(input_path)
                if img is not None:
                    calibrated_img = calibrate(img)

                    # Save calibrated image
                    output_path = os.path.join(output_subfolder, filename)
                    cv2.imwrite(output_path, calibrated_img)
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"Failed to load image: {input_path}")

def process_single_image(input_image_path, output_image_path):
    """
    Process a single image for calibration.
    
    Parameters:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the calibrated image.
    """
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return

    calibrated_img = calibrate(img)
    cv2.imwrite(output_image_path, calibrated_img)
    print(f"Calibrated image saved to {output_image_path}")


if __name__ == "__main__":
    # Option 1: Process a single image
    single_input_image = "test.jpg"  # Replace with the path to your image
    single_output_image = "output_image1.jpg"  # Replace with the desired output path
    process_single_image(single_input_image, single_output_image)
