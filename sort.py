
import cv2
import os

def images_to_video(input_folder, output_video_path, fps=30):
    input_folder = '/Users/uchimurataichi/SceneFlow/SceneFlow_oak_floor'
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # ファイル名から数字の部分を抽出してソート
    sorted_image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))

    # 画像の最初の1枚から動画のサイズを取得
    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # VideoWriterの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のコーデックを指定
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 画像を動画に結合
    for image_file in sorted_image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # リソースを解放
    video_writer.release()

if __name__ == "__main__":
    input_folder_path = '/Users/uchimurataichi/SceneFlow/SceneFlow_oak_floor'
    #input_folder_path = '/Users/uchimurataichi/SceneFlow/SceneFlow'
    output_video_path = '/Users/uchimurataichi/SceneFlow/SceneFlow_oak_floor/rgb_video.mp4'
    images_to_video(input_folder_path, output_video_path)