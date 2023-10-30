import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_depth_ins_params(param_file):
    '''
    read the depth intrinsic parameters file
    :param param_file: path to depth intrinsic parameters file DepthIns.txt
    :return:
    ir_camera_params_obj: a libfreenect2 IrCameraParams object
    '''
    with open(param_file, 'r') as f:
        depth_ins_params = [float(line.strip()) for line in f if line]
    ir_camera_params_obj = {
        "fx" : depth_ins_params[0],
        "fy" : depth_ins_params[1],
        "cx" : depth_ins_params[2],
        "cy" : depth_ins_params[3],
        "k1" : depth_ins_params[4],
        "k2" : depth_ins_params[5],
        "k3" : depth_ins_params[6],
        "p1" : depth_ins_params[7],
        "p2" : depth_ins_params[8]
    }
    return ir_camera_params_obj

def get_rgb_ins_params(param_file):
    '''
    read the rgb intrinsic parameters file
    :param param_file: path to depth intrinsic parameters file DepthIns.txt
    :return:
    rgb_ins_params: a libfreenect2 ColorCameraParams object
    '''
    with open(param_file, 'r') as f:
        rgb_ins_params = [float(line.strip()) for line in f if line]

    rgb_camera_params_obj = {
        "fx" : rgb_ins_params[0],
        "fy" : rgb_ins_params[1],
        "cx" : rgb_ins_params[2],
        "cy" : rgb_ins_params[3],

        "shift_d" : rgb_ins_params[4],
        "shift_m" : rgb_ins_params[5],
        "mx_x3y0" : rgb_ins_params[6],
        "mx_x0y3" : rgb_ins_params[7],
        "mx_x2y1" : rgb_ins_params[8],
        "mx_x1y2" : rgb_ins_params[9],
        "mx_x2y0" : rgb_ins_params[10],
        "mx_x0y2" : rgb_ins_params[11],
        "mx_x1y1" : rgb_ins_params[12],
        "mx_x1y0" : rgb_ins_params[13],
        "mx_x0y1" : rgb_ins_params[14],
        "mx_x0y0" : rgb_ins_params[15],

        "my_x3y0" : rgb_ins_params[16],
        "my_x0y3" : rgb_ins_params[17],
        "my_x2y1" : rgb_ins_params[18],
        "my_x1y2" : rgb_ins_params[19],
        "my_x2y0" : rgb_ins_params[20],
        "my_x0y2" : rgb_ins_params[21],
        "my_x1y1" : rgb_ins_params[22],
        "my_x1y0" : rgb_ins_params[23],
        "my_x0y1" : rgb_ins_params[24],
        "my_x0y0" : rgb_ins_params[25]
    }
    return rgb_camera_params_obj

def distort(mx, my, depth_ins):
    # see http://en.wikipedia.org/wiki/Distortion_(optics) for description
    # based on the C++ implementation in libfreenect2
    dx = (float(mx) - depth_ins['cx']) / depth_ins['fx']
    dy = (float(my) - depth_ins['cy']) / depth_ins['fy']
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    r2 = dx2 + dy2
    dxdy2 = 2 * dx * dy
    kr = 1 + ((depth_ins['k3'] * r2 + depth_ins['k2']) * r2 + depth_ins['k1']) * r2
    x = depth_ins['fx'] * (dx * kr + depth_ins['p2'] * (r2 + 2 * dx2) + depth_ins['p1'] * dxdy2) + depth_ins['cx']
    y = depth_ins['fy'] * (dy * kr + depth_ins['p1'] * (r2 + 2 * dy2) + depth_ins['p2'] * dxdy2) + depth_ins['cy']
    return x, y

def depth_to_color(mx, my, dz, depth_ins, color_ins):
    # based on the C++ implementation in libfreenect2, constants are hardcoded into sdk
    depth_q = 0.01
    color_q = 0.002199

    mx = (mx - depth_ins['cx']) * depth_q
    my = (my - depth_ins['cy']) * depth_q

    wx = (mx * mx * mx * color_ins['mx_x3y0']) + (my * my * my * color_ins['mx_x0y3']) + \
         (mx * mx * my * color_ins['mx_x2y1']) + (my * my * mx * color_ins['mx_x1y2']) + \
         (mx * mx * color_ins['mx_x2y0']) + (my * my * color_ins['mx_x0y2']) + \
         (mx * my * color_ins['mx_x1y1']) +(mx * color_ins['mx_x1y0']) + \
         (my * color_ins['mx_x0y1']) + (color_ins['mx_x0y0'])

    wy = (mx * mx * mx * color_ins['my_x3y0']) + (my * my * my * color_ins['my_x0y3']) +\
         (mx * mx * my * color_ins['my_x2y1']) + (my * my * mx * color_ins['my_x1y2']) +\
         (mx * mx * color_ins['my_x2y0']) + (my * my * color_ins['my_x0y2']) + (mx * my * color_ins['my_x1y1']) +\
         (mx * color_ins['my_x1y0']) + (my * color_ins['my_x0y1']) + color_ins['my_x0y0']

    rx = (wx / (color_ins['fx'] * color_q)) - (color_ins['shift_m'] / color_ins['shift_d'])
    ry = int((wy / color_q) + color_ins['cy'])

    rx = rx + (color_ins['shift_m'] / dz)
    rx = int(rx * color_ins['fx'] + color_ins['cx'])
    return rx, ry



def getPointXYZ(undistorted, r, c, depth_ins):
    depth_val = undistorted[r, c] #/ 1000.0  # map from mm to meters
    if np.any(np.isnan(depth_val)) or np.any(depth_val <= 0.001):
        x = 0
        y = 0
        z = 0
    else:
        x = (c + 0.5 - depth_ins['cx']) * depth_val / depth_ins['fx']
        y = (r + 0.5 - depth_ins['cy']) * depth_val / depth_ins['fy']
        z = depth_val
    point = [x, y, z]
    return point




depth_ins_params = get_depth_ins_params('DepthIns.txt')
rgb_ins_params = get_rgb_ins_params('ColorIns.txt')

# 動画ファイルを開く
cap_c = cv2.VideoCapture('scan_video.avi')
cap_d = cv2.VideoCapture('scan_video_depth.avi')
# 最初のフレームを読み込む
ret, frame1_d = cap_d.read(cv2.IMREAD_ANYDEPTH)
frame1_d = frame1_d.astype(np.float32)
if not ret:
    print("Failed to read the depth video.")

ret, frame1_c = cap_c.read()
if not ret:
    print("Failed to read the RGB video.")

point_cloud_1 = []
new_vertices_1 = []
registered = np.zeros([424, 512, 3], dtype=np.uint8)
undistorted = np.zeros([424, 512, 3], dtype=np.float32)

for y in range(424):
    for x in range(512):
        mx, my = distort(x, y, depth_ins_params)
        # distortで歪み補正したmx, myを整数値にする
        ix = int(mx + 0.5)
        iy = int(my + 0.5)
        
        point = getPointXYZ(frame1_d, y, x, depth_ins_params)
        
        if (ix >= 0 and ix < 512 and iy >= 0 and iy < 424):  # 指定ピクセルが画像内の場合
                undistorted[iy, ix] = frame1_d[y, x]
                z = frame1_d[y, x]
                if (z > 0).all and not np.isnan(z).all:
                    cx, cy = depth_to_color(x, y, z, depth_ins_params, rgb_ins_params)
                    cx, cy = int(cx[0]), int(cy[0])
                    if (cx >= 0 and cx < 1920 and cy >= 0 and cy < 1080):
                        registered[y, x, :] = frame1_c[cy, cx].flatten()
                        registered[y, x, :] = cv2.cvtColor(registered[y, x].reshape([1, 1, 3]),cv2.COLOR_BGR2RGB)
                        # [x, y, z, r, g, b]形式で各ピクセルの情報を格納
                        point_cloud_1.append([(point), registered[y, x, :]])
                        new_vertices_1.append((point[0], point[1], point[2], registered[y, x, 0], registered[y, x, 1], registered[y, x, 2]))

print('Depth&RGB対応完了')



# 前のフレームと同じサイズのゼロ行列を作成
prvs = cv2.cvtColor(frame1_c, cv2.COLOR_BGR2GRAY) #読み込んだフレームをグレースケール化。グレースケールグレースケール画像の方が計算が楽。
hsv = np.zeros_like(frame1_c)
hsv[..., 1] = 255

vectors = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    
    # 次のフレームを読み込む
    ret, frame2_c = cap_c.read()
    if not ret:
        break
    
    ret, frame2_d = cap_d.read(cv2.IMREAD_ANYDEPTH)
    frame2_d = frame2_d.astype(np.float32)
    if not ret:
        break
    
    
    point_cloud_2 = []
    new_vertices_2 = []
    registered_2 = np.zeros([424, 512, 3], dtype=np.uint8)
    undistorted_2 = np.zeros([424, 512, 3], dtype=np.float32)

    for Y in range(424):
        for X in range(512):
            mX, mY = distort(X, Y, depth_ins_params)
            # distortで歪み補正したmx, myを整数値にする
            iX = int(mX + 0.5)
            iY = int(mY + 0.5)
            
            point = getPointXYZ(frame2_d, Y, X, depth_ins_params)
            pre_point = getPointXYZ(frame1_d, Y, X, depth_ins_params)
            
            if (iX >= 0 and iX < 512 and iY >= 0 and iY < 424):  # 指定ピクセルが画像内の場合
                    undistorted_2[iY, iX] = frame2_d[Y, X]
                    Z = frame2_d[Y, X]
                    if (Z > 0).all and not np.isnan(Z).all:
                        cX, cY = depth_to_color(X, Y, Z, depth_ins_params, rgb_ins_params)
                        if (cX >= 0 and cX < 1920 and cY >= 0 and cY < 1080):
                            registered_2[Y, X, :] = frame2_c[cY, cX].flatten()
                            registered_2[Y, X, :] = cv2.cvtColor(registered_2[Y, X].reshape([1, 1, 3]),cv2.COLOR_BGR2RGB)
                            # [x, y, z, r, g, b]形式で各ピクセルの情報を格納
                            point_cloud_2.append([(point), registered_2[Y, X, :]])
                            new_vertices_2.append((point[0], point[1], point[2], registered_2[Y, X, 0], registered_2[Y, X, 1], registered_2[Y, X, 2]))
                            
                            
                            # 3D座標の点とベクトルを定義
                            plot_point = (point_cloud_2[0], point_cloud_2[1], point_cloud_2[2])
                            vector = (point[0] - pre_point[0], point[1] - pre_point[1], point[2] - pre_point[2])
                            
                            # 点をプロット
                            ax.scatter(*plot_point, color=(point_cloud_2[4], point_cloud_2[5], point_cloud_2[6]))
                            ax.quiver(*plot_point, *vector, color='blue')
                            
                            print("処理できてる")
                            
    

    
    # グレースケールに変換
    next = cv2.cvtColor(frame2_c, cv2.COLOR_BGR2GRAY)

    # Optical Flowを計算
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # オプティカルフローフィールドを表示
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    
    cv2.imshow('Scene Flow', bgr)
    
    
    ax.set_xlabel('X軸')
    ax.set_ylabel('Y軸')
    ax.set_zlabel('Z軸')
    ax.legend()
    plt.show()
    

    # 'q'キーを押したらループから抜ける
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # 現在のフレームを前のフレームに設定
    prvs = next
    frame1_c = frame2_c
    frame1_d = frame2_d
    point_cloud_1 = []
    new_vertices_1 = []
    point_cloud_1 = [(point[0], point[1], point[2], point[3], point[4], point[5]) for point in point_cloud_2]
    new_vertices_1 = [(point[0], point[1], point[2], point[3], point[4], point[5]) for point in new_vertices_2]

# ウィンドウを閉じる
cap_c.release()
cap_d.release()
cv2.destroyAllWindows()