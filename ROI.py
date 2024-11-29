
import cv2

# 이미지를 로드
image_path = "calibrated_OneObjectSection1/a0_e0_at0_et1_b1_o0/a0_e0_at0_et1_b1_o0 (2).jpg"  # 이미지 경로
image = cv2.imread(image_path)

print(image.shape)


roi = image[250:960, :]
# OpenCV의 imshow로 이미지 표시
cv2.imshow("Image Window", image)
cv2.imshow("roi Window", roi)

# 키 입력 대기 (0은 무한 대기)
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
#710 1280