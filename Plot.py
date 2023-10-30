import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


#======================================================================

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


def plot_points(initial_points, final_points):
    # ベクトルを計算
    vectors = final_points - initial_points

    # 3Dグラフの設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 初期座標をプロット
    #ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], c='r', marker='o', label='Initial Points')

    # 終点座標をプロット
    #ax.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2], c='g', marker='o', label='Final Points')

    # ベクトルを表示
    for i in range(len(vectors)):
        ax.quiver(initial_points[i, 0], initial_points[i, 1], initial_points[i, 2], vectors[i, 0], vectors[i, 1], vectors[i, 2],
                  color='b')

    # グラフの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # グラフを表示
    plt.show()

#======================================================================


depth_ins_params = get_depth_ins_params('DepthIns.txt')
rgb_ins_params = get_rgb_ins_params('ColorIns.txt')

# 動画ファイルを開く
cap_c = cv2.VideoCapture('scan_video.avi')
cap_d = cv2.VideoCapture('scan_video_depth.avi')
# 最初のフレームを読み込む
ret, frame1_d = cap_d.read(cv2.IMREAD_ANYDEPTH)
if not ret:
    print("Failed to read the depth video.")

frame1_d = frame1_d.astype(np.float32)

#cv2.imshow('Scene Flow', frame1_d)
#print(frame1_d)

ret, frame1_c = cap_c.read()
if not ret:
    print("Failed to read the RGB video.")

hcsize, wcsize = frame1_c.shape[:2]
hdsize, wdsize = frame1_d.shape[:2]

point_cloud_1 = []
new_vertices_1 = []
registered = np.zeros([hcsize, wcsize, 3], dtype=np.uint8)
undistorted = np.zeros([hdsize, wdsize, 3], dtype=np.float32)


for y in range(hdsize):
    for x in range(wdsize):
      mx, my = distort(x, y, depth_ins_params)
      # distortで歪み補正したmx, myを整数値にする
      ix = int(mx + 0.5)
      iy = int(my + 0.5)
      
      point = getPointXYZ(frame1_d, y, x, depth_ins_params)
      
      x_list = np.array(point[0])
      x_list = x_list[x_list != 0]
      x_mean = np.mean(x_list)
      y_list = np.array(point[1])
      y_list = y_list[y_list != 0]
      y_mean = np.mean(y_list)
      z_list = np.array(point[2])
      z_list = z_list[z_list != 0]
      z_mean = np.mean(z_list)
      
      if (ix >= 0 and ix < wdsize and iy >= 0 and iy < hdsize):  # 指定ピクセルが画像内の場合
                undistorted[iy, ix] = frame1_d[y, x]
                z = frame1_d[y, x]
                #z = z[z > 0]
                if np.any(z > 0) and not np.any(np.isnan(z)):
                    #print("こんにちは")
                    z = z[z > 0]
                    average_z = sum(z) // len(z)
                    #print(average_z)
                    cx, cy = depth_to_color(x, y, average_z, depth_ins_params, rgb_ins_params)
                    #cx, cy = int(cx[0]), int(cy[0])
                    if (cx >= 0 and cx < wcsize and cy >= 0 and cy < hcsize):
                        registered[y, x, :] = frame1_c[cy, cx].flatten()
                        registered[y, x, :] = cv2.cvtColor(registered[y, x].reshape([1, 1, 3]),cv2.COLOR_BGR2RGB)
                        # [x, y, z, r, g, b]形式で各ピクセルの情報を格納
                        point_cloud_1.append([(point), registered[y, x, :]])
                        new_vertices_1.append((x, y, x_mean, y_mean, z_mean, registered[y, x, 0], registered[y, x, 1], registered[y, x, 2]))
                        #print(point_cloud_1)

print('Depth&RGB対応完了')
"""
for point in new_vertices_1:
    # ポイントの座標とRGB情報を取得
    #x, y, z, r, g, b = point[0][0], point[0][1], point[0][2], point[1][0], point[1][1], point[1][2]
    x, y, z, r, g, b = point[2], point[3], point[4], point[5], point[6], point[7]
    X, Y = point[0], point[1]
    # ポイントの情報を表示
    print(f"x: {X}, y: {Y}, X: {x}, Y: {y}, Z: {z}, R: {r}, G: {g}, B: {b}")
"""




# 前のフレームと同じサイズのゼロ行列を作成
prvs = cv2.cvtColor(frame1_c, cv2.COLOR_BGR2GRAY) #読み込んだフレームをグレースケール化。グレースケールグレースケール画像の方が計算が楽。
hsv = np.zeros_like(frame1_c)
hsv[..., 1] = 255

vectors = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

while True:
    
    # 次のフレームを読み込む
    ret, frame2_c = cap_c.read()
    if not ret:
        break
    #print(frame2_c)


    ret, frame2_d = cap_d.read(cv2.IMREAD_ANYDEPTH)
    frame2_d = frame2_d.astype(np.float32)
    if not ret:
        break
    #print(frame2_d)
    #cv2.imshow('Scene Flow', frame2_d)
    
    hcsize, wcsize = frame2_c.shape[:2]
    hdsize, wdsize = frame2_d.shape[:2]
    
    #print(hdsize)
    #print(wdsize)
    
    point_cloud_2 = []
    new_vertices_2 = []
    registered_2 = np.zeros([hdsize, wdsize, 3], dtype=np.uint8)
    undistorted_2 = np.zeros([hdsize, wdsize, 3], dtype=np.float32)
    
    count = 0
    final_points = np.array([[0,0,0]])
    initial_points = np.array([[0,0,0]])

    for Y in range(hdsize):
        #print("繰り返し")
        for X in range(wdsize):
            mX, mY = distort(X, Y, depth_ins_params)
            # distortで歪み補正したmx, myを整数値にする
            iX = int(mX + 0.5)
            iY = int(mY + 0.5)
            
            point = getPointXYZ(frame2_d, Y, X, depth_ins_params)
            pre_point = getPointXYZ(frame1_d, Y, X, depth_ins_params)
            
            x_list = np.array(point[0])
            x_list = x_list[x_list != 0]
            x_mean = np.mean(x_list)
            y_list = np.array(point[1])
            y_list = y_list[y_list != 0]
            y_mean = np.mean(y_list)
            z_list = np.array(point[2])
            z_list = z_list[z_list != 0]
            z_mean = np.mean(z_list)
            
            pre_x_list = np.array(pre_point[0])
            pre_x_list = pre_x_list[pre_x_list != 0]
            pre_x_mean = np.mean(pre_x_list)
            pre_y_list = np.array(pre_point[1])
            pre_y_list = pre_y_list[pre_y_list != 0]
            pre_y_mean = np.mean(pre_y_list)
            pre_z_list = np.array(pre_point[2])
            pre_z_list = pre_z_list[pre_z_list != 0]
            pre_z_mean = np.mean(pre_z_list)
            
            distance = math.sqrt((x_mean - pre_x_mean)**2 + (y_mean - pre_y_mean)**2 + (z_mean - pre_z_mean)**2)
            
            

            if (distance >= 10 and distance <= 1000 ):
                count = count + 1
                new_row = np.array([x_mean, y_mean, z_mean])
                initial_points = np.vstack((initial_points, new_row))
                new_row = np.array([pre_x_mean, pre_y_mean, pre_z_mean])
                final_points = np.vstack((final_points, new_row))
            #print(final_points)
            #final_points = np.array([[4, 5, 6], [6, 7, 8], [7, 8, 9]])
            
            if (iX >= 0 and iX < wdsize and iY >= 0 and iY < hdsize):  # 指定ピクセルが画像内の場合
                    undistorted_2[iY, iX] = frame2_d[Y, X]
                    Z = frame2_d[Y, X]
                    #print("ここまで来てる２")
                    if np.any(Z > 0) and not np.any(np.isnan(Z)):
                        Z = Z[Z > 0]
                        average_Z = sum(Z) // len(Z)
                        cX, cY = depth_to_color(X, Y, average_Z, depth_ins_params, rgb_ins_params)
                        #print("ここまで来てる３")
                        if (cX >= 0 and cX < wcsize and cY >= 0 and cY < hcsize):
                            registered_2[Y, X, :] = frame2_c[cY, cX].flatten()
                            registered_2[Y, X, :] = cv2.cvtColor(registered_2[Y, X].reshape([1, 1, 3]),cv2.COLOR_BGR2RGB)
                            #print("ここまで来てる4")
                            # [x, y, z, r, g, b]形式で各ピクセルの情報を格納
                            point_cloud_2.append([(point), registered_2[Y, X, :]])
                            new_vertices_2.append((X, Y, x_mean, y_mean, z_mean, registered_2[Y, X, 0], registered_2[Y, X, 1], registered_2[Y, X, 2]))
                            
                            
                            # 3D座標の点とベクトルを定義
                            #plot_point = (point[0], point[1], point[2])
                            #vector = (point[0] - pre_point[0], point[1] - pre_point[1], point[2] - pre_point[2])
                            #print(plot_point)
                            # 点をプロット
                            #c_values = registered_2[Y, X, :3]  # RGB値を抽出
                            #c_values = c_values / 255.0  # 255で割って正規化
                            #print(c_values)
                            #ax.scatter(*plot_point, c='blue', s=5)  # カラーマップを無効にするために色を指定
                            #ax.scatter(*plot_point, c=[c_values], cmap='viridis')  # カラーマップを指定
                            #ax.scatter(*plot_point, color=(registered_2[Y, X, 0], registered_2[Y, X, 1], registered_2[Y, X, 2]))
                            

                            #ax.quiver(*plot_point, *vector, color='blue')
                            
                            #print("処理できてる")
                            #plt.show()
                            #plt.close()
                            #ax.legend()
        
                            
    #plt.show()
    plot_points(initial_points, final_points)
    plt.close()
    print(count)
    
    '''
    """
    ax.set_xlabel('X軸')
    ax.set_ylabel('Y軸')
    ax.set_zlabel('Z軸')
    ax.legend()
    plt.show()
    """
    for point in new_vertices_2:
        # ポイントの座標とRGB情報を取得
        #x, y, z, r, g, b = point[0][0], point[0][1], point[0][2], point[1][0], point[1][1], point[1][2]
        x, y, z, r, g, b = point[2], point[3], point[4], point[5], point[6], point[7]
        X, Y = point[0], point[1]
        # ポイントの情報を表示
        distance = math.sqrt((x_mean - pre_x_mean)**2 + (y_mean - pre_y_mean)**2 + (z_mean - pre_z_mean)**2)
        
        if (distance >= 0.00001):
            count = count + 1
        #print(f"x: {X}, y: {Y}, X: {x}, Y: {y}, Z: {z}, R: {r}, G: {g}, B: {b}")
        
        
    print(count)
    '''

    # 'q'キーを押したらループから抜ける
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # 現在のフレームを前のフレームに設定
    prvs = next
    frame1_c = frame2_c
    frame1_d = frame2_d
    point_cloud_1 = []
    new_vertices_1 = []
    #point_cloud_1 = [(point[0], point[1], point[2], point[3], point[4], point[5]) for point in point_cloud_2]
    #new_vertices_1 = [(point[0], point[1], point[2], point[3], point[4], point[5]) for point in new_vertices_2]
    #plt.clf()

# ウィンドウを閉じる
cap_c.release()
cap_d.release()
cv2.destroyAllWindows()