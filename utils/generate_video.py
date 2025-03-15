import cv2
import os

def generate_video_from_images(image_folder, output_video, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()

    if not images:
        print("No images found in the folder.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    model = "YOLO"
    image_folder = f"/home/daan/object_detection/{model}/runs/e:100_b:8_data:split_1/predict"
    output_dir = "/home/daan/object_detection/results/"
    output_video = f"{model}-e:100_b:8_data:split_1.mp4"
    generate_video_from_images(image_folder, output_dir+output_video)