# import random
# import cv2


# def video_to_frames(video_path, skip_frames=1):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         height, width, _ = frame.shape
#         x = random.randint(0, width - 640)
#         y = random.randint(0, height - 640)
#         cropped_frame = frame[y : y + 640, x : x + 640]
#         if frame_count % skip_frames == 0:
#             cv2.imwrite(
#                 f"./dataset/frames/frame_{frame_count}.jpg",
#                 cropped_frame,
#             )
#         frame_count += 1
#         print(f"Processed frame {frame_count + 9750}")
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     video_path = "D:\\OneDrive\\Desktop\\数据标注\\0.mkv"
#     video_to_frames(video_path, skip_frames=50)
import os
import random
import cv2


def image_random_crop(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    x = random.randint(0, width - 640)
    y = random.randint(0, height - 640)
    cropped_image = image[y : y + 640, x : x + 640]
    output_path = os.path.join(
        "./dataset/frames", os.path.basename(image_path).replace(".png", ".jpg")
    )
    cv2.imwrite(output_path, cropped_image)


def process_images_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(directory_path, filename)
            image_random_crop(image_path)
            print(f"Processed image {filename}")


if __name__ == "__main__":
    input_directory = "./dataset/1"
    process_images_in_directory(input_directory)
