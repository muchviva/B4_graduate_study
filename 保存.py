import cv2
import os

# 動画ファイルを開く
cap = cv2.VideoCapture('scan_video.avi')

# 保存するフレームのディレクトリを指定
output_directory = 'frames'
os.makedirs(output_directory, exist_ok=True)

frame_number = 0

while True:
    # 次のフレームを読み込む
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをフォルダに保存
    frame_filename = os.path.join(output_directory, f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(frame_filename, frame)

    # 次のフレームの処理に進む
    frame_number += 1

    # 'q'キーを押したらループから抜ける
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
