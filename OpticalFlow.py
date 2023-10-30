import cv2
import numpy as np

# 動画ファイルを開く
cap = cv2.VideoCapture('scan_video.avi')

# 最初のフレームを読み込む
ret, frame1 = cap.read()

# 前のフレームと同じサイズのゼロ行列を作成
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) #読み込んだフレームをグレースケール化。グレースケールグレースケール画像の方が計算が楽。
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    # 次のフレームを読み込む
    ret, frame2 = cap.read()
    if not ret:
        break

    # グレースケールに変換
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical Flowを計算
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # オプティカルフローフィールドを表示
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Optical Flow', bgr)

    # 'q'キーを押したらループから抜ける
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 現在のフレームを前のフレームに設定
    prvs = next

# ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
