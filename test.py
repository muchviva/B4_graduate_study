import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
from PIL import Image, ImageFilter


#======================================================================

#フォルダー"depth"から深度画像を読み込む
def read_depth_image(i):
  #フォルダーのi番目の画像を持ってくる
  number = str(i).zfill(3)  # 3桁にゼロ埋め
  image_path = f'/Users/uchimurataichi/SceneFlow/depth/000{number}.png'
  try:
      #image = Image.open(image_path)
      image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
      return image
  except FileNotFoundError:
      print(f"File not found: {image_path}")
      return None


#深度カメラの内部パラメータが記録されたファイル（引数）を読み取り、そのパラメータを返すもの
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


#RGBカメラの内部パラメータが記録されたファイルを読み取り、それらのパラメータを返すもの
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


#レンズ歪みの影響を受けた2D画像上の座標（mx, my）を、歪みが補正された座標（x, y）に変換
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
    #return x, y
    return int(x), int(y)


#深度画像上の座標（mx, my）と対応する深度値（dz）を受け取り、それをカラー画像上の座標に変換
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


# 歪みが補正された深度画像上の特定のピクセル（r, c）に対応する3Dポイント（x, y, z）を取得
# もし深度値が無効な場合や非常に小さい場合（0.001未満）は、座標を原点に設定
# それ以外の場合は、指定された内部パラメータを使用して深度座標に変換し、3Dポイントを作成して返す
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


# 3D座標（rx、ry、dz）を受け取り、その座標がカラーカメラの画像上のどの位置に投影されるかを計算
# 具体的には、Y座標（ry）は整数値に切り捨てられ、X座標（rx）はカラーカメラの内部パラメータ（fx、cx、shift_m、dz）を使用して計算
# 計算結果のcx、cyは、カラーカメラ画像上の特定のピクセル座標を表示
def apply(dz, rx, ry, rgb_ins_params):
    cy = int(ry)
    rx = rx + (rgb_ins_params['shift_m'] / dz)
    cx = int(rx * rgb_ins_params['fx'] + rgb_ins_params['cx'])
    return cx, cy


# 座標やベクトルを3Dグラフに表示
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

def flow_vector(flow, spacing, margin, minlength):
    """Parameters:
    input
    flow: motion vectors 3D-array
    spacing: pixel spacing of the flow
    margin: pixel margins of the flow
    minlength: minimum pixels to leave as flow
    output
    x: x coord 1D-array
    y: y coord 1D-array
    u: x direction flow vector 2D-array
    v: y direction flow vector 2D-array
    """
    h, w, _ = flow.shape

    x = np.arange(margin, w - margin, spacing, dtype=np.int64)
    y = np.arange(margin, h - margin, spacing, dtype=np.int64)

    mesh_flow = flow[np.ix_(y, x)]
    mag, _ = cv2.cartToPolar(mesh_flow[..., 0], mesh_flow[..., 1])
    mesh_flow[mag < minlength] = np.nan  # minlength以下をnanに置換

    u = mesh_flow[..., 0]
    v = mesh_flow[..., 1]

    return x, y, u, v

def cv2_vector_display(img, color, x, y, u, v):
    """Parameters:
    input
    img: the first image used to calculate the flow
    color: 3 channels 0 to 255 color
    x, y, u, v: `flow_vector` function reuslts
    x: x coord 1D-array
    y: y coord 1D-array
    u: x direction flow vector 2D-array
    v: y direction flow vector 2D-array
    output
    flow_img: image with vector added to input `img`
    description
    Create an image with a vector added to the input `img` using cv2.
    e.g. u[j, i] is the vector of flow in the x direction at (x[i], y[j]) coord.
    """
    flow_img = np.copy(img)
    for i in range(len(x)):
        for j in range(len(y)):
            if np.isnan(u[j, i]) or np.isnan(v[j, i]):
                continue
            pts = np.array([[x[i], y[j]], [x[i]+u[j, i], y[j]+v[j, i]]], np.int64)
            cv2.arrowedLine(flow_img, pts[0], pts[1], color, thickness=1, tipLength=0.5)
    return flow_img


def interpolate_missing_pixels(image):
    height, width, channels = image.shape
    #print(f"height = {height}, width = {width}")

    for y in range(height):
        if np.all(image[y, :] == [0, 0, 0]):
            image[y, :, 0] = 1  # 赤成分
            image[y, :, 1] = 1  # 緑成分
            image[y, :, 2] = 1  # 青成分
        for x in range(width):
            #print(f"x = {x}, y = {y}")
            # 欠落したピクセルの場合
            if np.all(image[y, -x] == [0, 0, 0]):
                # 上下左右のピクセルのRGB情報を取得
                y_top = y
                y_under = y
                x_right = -x
                x_left = -x
                
                tr = tg = tb = 0
                ur = ug = ub = 0
                rr = rg = rb = 0
                lr = lg = lb = 0
                r = g = b = 0
                
                count = 0
                
                #上
                while (y_top >= 0):
                  y_top = y_top - 1
                  
                  if y_top < 0:
                    break
                  
                  if not np.all(image[y_top, x] == [0, 0, 0]):
                    (tr, tg, tb) = image[y_top, x]
                    #r = r + tr
                    #g = g + tg
                    #b = b + tb
                    count = count + 1
                    break
                
                #下
                while (y_under < height):
                  y_under = y_under + 1
                  
                  if y_under >= height:
                    break
                  
                  if not np.all(image[y_under, x] == [0, 0, 0]):
                    (ur, ug, ub) = image[y_under, x]
                    #r = r + ur
                    #g = g + ug
                    #b = b + ub
                    count = count + 1
                    break
                
                #右
                while (x_right <= 0):
                  x_right = x_right + 1
                  
                  if x_right > 0:
                    break
                  
                  if not np.all(image[y, x_right] == [0, 0, 0]):
                    (rr, rg, rb) = image[y, x_right]
                    #r = r + rr
                    #g = g + rg
                    #b = b + rb
                    count = count + 1
                    break
                
                #左
                while (x_left < -(width - 1)):
                  x_left = x_left - 1
                  
                  if x_left < -(width - 1):
                    break
                  
                  if not np.all(image[y, x_left] == [0, 0, 0]):
                    (lr, lg, lb) = image[y, x_left]
                    #r = r + lr
                    #g = g + lg
                    #b = b + lb
                    count = count + 1
                    break
                
                # RGB情報の平均を計算
                if count > 0:
                    r = (tr + ur + rr + lr)
                    r = r / count
                    g = (tg + ug + rg + lg)
                    g = g / count
                    b = (tb + ub + rb + lb)
                    b = b / count
                    
                    image[y, -x, 0] = r  # 赤成分
                    image[y, -x, 1] = g  # 緑成分
                    image[y, -x, 2] = b  # 青成分
                #print("ウェイ")



    return image



#======================================================================



#深度カメラのパラメータをdepth_ins_paramsに格納
depth_ins_params = get_depth_ins_params('DepthIns.txt')
#RGBカメラのパラメータをrgb_ins_paramsに格納
rgb_ins_params = get_rgb_ins_params('ColorIns.txt')

#RGB videoから最初のフレームを読み込む
cap_c = cv2.VideoCapture('scan_video.avi')
ret, rgb_img = cap_c.read()
if not ret:
    print("Failed to read the RGB video.")

#depthフォルダから最初のimgを持ってくる
i = 0
depth_img = read_depth_image(i)
#depth_img = cv2.imread(d_image, cv2.IMREAD_ANYDEPTH).astype(np.float32)

rgb_img = cv2.flip(rgb_img, 1)
depth_img = cv2.flip(depth_img, 1)

hcsize, wcsize = rgb_img.shape[:2]
hdsize, wdsize = depth_img.shape[:2]

# 保存するフレームのディレクトリを指定
output_directory = 'Gene_rgb'
os.makedirs(output_directory, exist_ok=True)

frame_number = 0

new_vertices = []
#registered = np.zeros([hdsize, wdsize, 3], dtype=np.uint8)

# 画像データの構造を用意
image = np.zeros((hdsize, wdsize, 3), dtype=np.uint8)  # 3チャンネルのRGB画像

for y in range(hdsize):
    for x in range(wdsize):
        #distortの出力を整数値にしたのでこのままでOK
        ix, iy = distort(x, y, depth_ins_params)
        point = getPointXYZ(depth_img, y, x, depth_ins_params)
        #歪み補正した2D座標が画像座標上にあるなら
        if (ix >= 0 and ix < wdsize and iy >= 0 and iy < hdsize): #wdsize = 512, hdsize = 424
          z = depth_img[iy, ix]
          #zが０以外の数なら
          if z > 0 and not np.isnan(z):
            cx, cy = depth_to_color(x, y, z, depth_ins_params, rgb_ins_params)
            if (cx >= 0 and cx < wcsize and cy >= 0 and cy < hcsize): #wcsize = 1920, hcsize = 1080
              #ここの処理で画素に色が入る
              
              cb, cg, cr = rgb_img[cy, cx].flatten()
              #registered[y, x, :] = (cr, cg, cb)
              #new_vertices.append((-point[0], -point[1], -point[2], cr, cg, cb))
              new_vertices.append((point[0], -point[1], point[2], cr, cg, cb))
              image[y, -x, 0] = cr  # 赤成分
              image[y, -x, 1] = cg  # 緑成分
              image[y, -x, 2] = cb  # 青成分
#'''
            else:
              image[y, -x, 0] = 0  # 赤成分
              image[y, -x, 1] = 255  # 緑成分
              image[y, -x, 2] = 0  # 青成分
          
          else:
              image[y, -x, 0] = 0  # 赤成分
              image[y, -x, 1] = 0  # 緑成分
              image[y, -x, 2] = 255  # 青成分
#'''

#image = interpolate_missing_pixels(image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cv2ではBGRで表示されるため

frame_filename = os.path.join(output_directory, f'rgb_{frame_number:04d}.png')
cv2.imwrite(frame_filename, image_rgb)

frame_number += 1

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

while True:
    # 次のフレームを読み込む
    ret, rgb_img2 = cap_c.read()
    if not ret:
        print("Failed to read the RGB video.")
        break
    
    #depthフォルダから次のimgを持ってくる
    i += 1
    depth_img2 = read_depth_image(i)
    
    rgb_img2 = cv2.flip(rgb_img2, 1)
    depth_img2 = cv2.flip(depth_img2, 1)
    
    new_vertices = []
    #registered = np.zeros([hdsize, wdsize, 3], dtype=np.uint8)
    
    flow = np.zeros((hdsize, wdsize, 2))
    
    next_image = np.zeros((hdsize, wdsize, 3), dtype=np.uint8)  # 3チャンネルのRGB画像
    
    for y in range(hdsize):
        for x in range(wdsize):
            #distortの出力を整数値にしたのでこのままでOK
            ix, iy = distort(x, y, depth_ins_params)
            point = getPointXYZ(depth_img2, y, x, depth_ins_params)
            #歪み補正した2D座標が画像座標上にあるなら
            if (ix >= 0 and ix < wdsize and iy >= 0 and iy < hdsize): #wdsize = 512, hdsize = 424
              z = depth_img2[iy, ix]
              #zが０以外の数なら
              if z > 0 and not np.isnan(z):
                cx, cy = depth_to_color(x, y, z, depth_ins_params, rgb_ins_params)
                if (cx >= 0 and cx < wcsize and cy >= 0 and cy < hcsize): #wcsize = 1920, hcsize = 1080
                  cb, cg, cr = rgb_img2[cy, cx].flatten()
                  #registered[y, x, :] = (cr, cg, cb)
                  #new_vertices.append((-point[0], -point[1], -point[2], cr, cg, cb))
                  new_vertices.append((point[0], -point[1], point[2], cr, cg, cb))
                  next_image[y, -x, 0] = cr  # 赤成分
                  next_image[y, -x, 1] = cg  # 緑成分
                  next_image[y, -x, 2] = cb  # 青成分
                
                else:
                  next_image[y, -x, 0] = 0  # 赤成分
                  next_image[y, -x, 1] = 255  # 緑成分
                  next_image[y, -x, 2] = 0  # 青成分
              
              else:
                  next_image[y, -x, 0] = 0  # 赤成分
                  next_image[y, -x, 1] = 0  # 緑成分
                  next_image[y, -x, 2] = 255  # 青成分
    
    #next_image = interpolate_missing_pixels(next_image)
    next_image_rgb = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
    
    frame_filename = os.path.join(output_directory, f'rgb_{frame_number:04d}.png')
    cv2.imwrite(frame_filename, next_image_rgb)

    
    
    # グレースケールに変換
    gray1 = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(next_image_rgb, cv2.COLOR_BGR2GRAY)
    
    # Optical Flowを計算
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('prev_frame and prev2nextflow vector')
    ax.imshow(image_rgb)
    x, y, u, v = flow_vector(flow=flow, spacing=10, margin=0, minlength=1)
    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color=[0.0, 1.0, 0.0])
    flow_img_vector = cv2_vector_display(image_rgb, (0, 255, 0), x, y, u, v)

    #plt.show()
    #frame_filename = os.path.join(output_directory, f'flow_img_vector_{frame_number:04d}.png')
    #cv2.imwrite(frame_filename, flow_img_vector)
    
    '''
    # フラットな形式に変換
    flat_flow = image.reshape(-1, 2)  # 2チャンネルのデータをフラットな形式に変換
    
    # 2. ファイルへの書き出し
    with open('flow_data.txt', 'w') as file:
        for pixel in flat_flow:
            u, v = pixel
            file.write(f'u = {u}, v = {v}\n')
    print("txt切り替え")
    
    
    # 光流アルゴリズムの設定
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # 初期の対応点をランダムに選択
    p0 = cv2.goodFeaturesToTrack(gray1, mask = None, **feature_params)
    
    
    # 光流を計算
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    
    # 有効な対応点の選択
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # 対応点と移動ベクトルを表示
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        #print(f"(a = {a}, b = {b})")
        pt1 = (int(a), int(b))
        pt2 = (int(c), int(d))
        cv2.line(next_image_rgb, pt1, pt2, (0, 0, 255), 2)
        cv2.circle(next_image_rgb, pt1, 5, (0, 0, 255), -1)
    
    # 結果の表示
    #cv2.imshow('Optical Flow', next_image_rgb)
    
    # 'q'キーを押したらループから抜ける
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    
    # フレームをフォルダに保存
    frame_filename = os.path.join(output_directory, f'generate_rgb_{frame_number:04d}.jpg')
    cv2.imwrite(frame_filename, next_image_rgb)
    '''

    # 次のフレームの処理に進む
    frame_number += 1
    
    image_rgb = next_image_rgb.copy()







# 画像を表示
cv2.imshow('Generated Image', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
# 1. imageのデータをフラットな形式に変換
flat_image = image.reshape(-1, 3)  # 3チャンネルのデータをフラットな形式に変換

# 2. ファイルへの書き出し
with open('image_data.txt', 'w') as file:
    for pixel in flat_image:
        r, g, b = pixel
        file.write(f'{r} {g} {b}\n')

'''


cap_c.release()
