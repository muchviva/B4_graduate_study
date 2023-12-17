import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
from PIL import Image, ImageFilter

#=================================================================

#フォルダー"depth"から深度画像を読み込む
def read_depth_image(i):
  #フォルダーのi番目の画像を持ってくる
  number = str(i).zfill(3)  # 3桁にゼロ埋め
  image_path = f'/Users/uchimurataichi/SceneFlow/depth_Kallax_Shelf_Drawer/000{number}.png'
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

def get_relative_depth(img, min_depth_val=0.0, max_depth_val = 4500, colormap='jet'):
    '''
    Convert the depth image to relative depth for better visualization. uses fixed minimum and maximum distances
    to avoid flickering
    :param img: depth image
           min_depth_val: minimum depth in mm (default 50cm)
           max_depth_val: maximum depth in mm ( default 10m )
    :return:
    relative_depth_frame: relative depth converted into cv2 GBR
    '''

    relative_depth_frame = cv2.convertScaleAbs(img, alpha=(255.0/max_depth_val),
                                               beta=-255.0*min_depth_val/max_depth_val)
    relative_depth_frame = cv2.cvtColor(relative_depth_frame, cv2.COLOR_GRAY2BGR)
    # relative_depth_frame = cv2.applyColorMap(relative_depth_frame, cv2.COLORMAP_JET)
    return relative_depth_frame

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
def getPointXYZ(depth_val, r, c, depth_ins):
    #depth_val = undistorted[r, c] #/ 1000.0  # map from mm to meters
    if (np.isnan(depth_val)) or (depth_val <= 0.001):
        x = 0
        y = 0
        z = 0
    else:
        x = (c + 0.5 - depth_ins['cx']) * depth_val / depth_ins['fx']
        y = (r + 0.5 - depth_ins['cy']) * depth_val / depth_ins['fy']
        z = depth_val
    point = [x, y, z]
    return point

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

def flow_vector1(flow, x, y, z, depth_img2):

    u = flow[y, x, 0]
    v = flow[y, x, 1]
    
    point = getPointXYZ(z, y, x, depth_ins_params)
    
    if(int(y+v)>423 or int(x+u)>511):
        w = 0
    else:
        z2 = depth_img2[int(y+v), int(x+u)]
        w = z2 - z

    x_v = [x]
    y_v = [y]
    z_v = [z]
    x_v2 = [point[0]]
    y_v2 = [point[1]]
    z_v2 = [point[2]]
    u_v = [u]
    v_v = [v]
    w_v = [w]
    
    
    return x_v, y_v, z_v, u_v, v_v, w_v, x_v2, y_v2, z_v2

"""
def flow_vector2(flow, x, y, z, depth_img2, x_v, y_v, z_v, u_v, v_v, w_v):
    x_v.append(x)
    y_v.append(y)
    z_v.append(z)


    u = flow[y, x, 0]
    v = flow[y, x, 1]
    
    if(int(y+v)>423 or int(x+u)>511):
        w = 0
    else:
        z2 = depth_img2[int(y+v), int(x+u)]
        w = z2 - z

    #x_v2 = x_v.extend([x])
    u_v.append(u)
    v_v.append(v)
    w_v.append(w)
        #print(f"x_v={x_v}, y_v={y_v}")


    return x_v, y_v, z_v, u_v, v_v, w_v

"""
def flow_vector2(flow, x, y, z, depth_img2, x_v, y_v, z_v, u_v, v_v, w_v, x_v2, y_v2, z_v2):
    point = getPointXYZ(z, y, x, depth_ins_params)
    
    x_v.append(x)
    y_v.append(y)
    z_v.append(z)
    x_v2.append(point[0])
    y_v2.append(point[1])
    z_v2.append(point[2])


    u = flow[y, x, 0]
    v = flow[y, x, 1]
    mag = np.sqrt(u**2 + v**2)
    if(mag >= 0.5):
        #print(f"2Dベクトルの大きさは{mag}")
        #print(f"int(y+v)={int(y+v)}, int(x+u)={int(x+u)}")
        if(int(y+v)>423 or int(x+u)>511):
            w = 0
        else:
            z2 = depth_img2[int(y+v), int(x+u)]
            w = z2 - z
        
        #x_v2 = x_v.extend([x])
        u_v.append(u)
        v_v.append(v)
        w_v.append(w)
        #print(f"x_v={x_v}, y_v={y_v}")
    else:
        u_v.append(0)
        v_v.append(0)
        w_v.append(0)


    return x_v, y_v, z_v, u_v, v_v, w_v, x_v2, y_v2, z_v2
#"""




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



#==================================================================

#深度カメラのパラメータをdepth_ins_paramsに格納
depth_ins_params = get_depth_ins_params('DepthIns.txt')
#RGBカメラのパラメータをrgb_ins_paramsに格納
rgb_ins_params = get_rgb_ins_params('ColorIns.txt')

#RGB videoから最初のフレームを読み込む
cap_c = cv2.VideoCapture('scan_video.avi')
ret, rgb_img = cap_c.read()
if not ret:
    print("Failed to read the RGB video.")

frame_number = 0

#depthフォルダから最初のimgを持ってくる
i = 0
depth_img = read_depth_image(i)


# 保存するフレームのディレクトリを指定
output_directory = 'SceneFlow_oak_floor'
os.makedirs(output_directory, exist_ok=True)

hcsize, wcsize = rgb_img.shape[:2]
hdsize, wdsize = depth_img.shape[:2]

window_size = 3

vertices = np.zeros((hdsize, wdsize, 1))

# 画像データの構造を用意
image = np.zeros((hdsize, wdsize, 3), dtype=np.uint8)  # 3チャンネルのRGB画像

loss = np.zeros((hdsize, wdsize, 1)) #画素抜けしてるピクセルの座標を保存

count = 0

while count < 3:
    count += 1
    for y in range(1, 423):
        for x in range(1, 511):
          if(depth_img[y, x] == 0):
              roi = depth_img[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1]
              if roi.size > 0:
                max_value = np.max(roi)
              else:
                max_value = 0
              depth_img[y,x] = max_value


rgb_img = cv2.flip(rgb_img, 1)
depth_img = cv2.flip(depth_img, 1)

for y in range(hdsize):
    for x in range(wdsize):
        #distortの出力を整数値にしたのでこのままでOK
        ix, iy = distort(x, y, depth_ins_params)
        #point = getPointXYZ(depth_img, y, x, depth_ins_params)
        #歪み補正した2D座標が画像座標上にあるなら
        if (ix >= 0 and ix < wdsize and iy >= 0 and iy < hdsize): #wdsize = 512, hdsize = 424
          z = depth_img[iy, ix]
          point = getPointXYZ(z, y, x, depth_ins_params)
          #zが０以外の数なら
          if z > 0 and not np.isnan(z):
            cx, cy = depth_to_color(x, y, z, depth_ins_params, rgb_ins_params)
            if (cx >= 0 and cx < wcsize and cy >= 0 and cy < hcsize): #wcsize = 1920, hcsize = 1080
              #ここの処理で画素に色が入る
              
              cb, cg, cr = rgb_img[cy, cx].flatten()
              #registered[y, x, :] = (cr, cg, cb)
              #new_vertices.append((-point[0], -point[1], -point[2], cr, cg, cb))
              vertices[y, -x, 0] = z
              image[y, -x, 0] = cr  # 赤成分
              image[y, -x, 1] = cg  # 緑成分
              image[y, -x, 2] = cb  # 青成分

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cv2ではBGRで表示されるため



while True:
    # 次のフレームを読み込む
    ret, rgb_img2 = cap_c.read()
    if not ret:
        print("Failed to read the RGB video.")
        break
    
    #depthフォルダから次のimgを持ってくる
    i += 1
    depth_img2 = read_depth_image(i)
    
    count = 0

    while count < 3:
        count += 1
        for y in range(1, 423):
            for x in range(1, 511):
              if(depth_img2[y, x] == 0):
                  roi = depth_img2[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1]
                  if roi.size > 0:
                    max_value = np.max(roi)
                  else:
                    max_value = 0
                  depth_img2[y,x] = max_value
    
    rgb_img2 = cv2.flip(rgb_img2, 1)
    depth_img2 = cv2.flip(depth_img2, 1)
    
    new_vertices = np.zeros((hdsize, wdsize, 1))
    #registered = np.zeros([hdsize, wdsize, 3], dtype=np.uint8)
    
    flow = np.zeros((hdsize, wdsize, 2))
    
    next_image = np.zeros((hdsize, wdsize, 3), dtype=np.uint8)  # 3チャンネルのRGB画像
    
    for y in range(hdsize):
        for x in range(wdsize):
            #distortの出力を整数値にしたのでこのままでOK
            ix, iy = distort(x, y, depth_ins_params)
            #歪み補正した2D座標が画像座標上にあるなら
            if (ix >= 0 and ix < wdsize and iy >= 0 and iy < hdsize): #wdsize = 512, hdsize = 424
              z = depth_img2[iy, ix]
              point = getPointXYZ(z, y, x, depth_ins_params)
              #zが０以外の数なら
              if z > 0 and not np.isnan(z):
                cx, cy = depth_to_color(x, y, z, depth_ins_params, rgb_ins_params)
                if (cx >= 0 and cx < wcsize and cy >= 0 and cy < hcsize): #wcsize = 1920, hcsize = 1080
                  cb, cg, cr = rgb_img2[cy, cx].flatten()
                  #registered[y, x, :] = (cr, cg, cb)
                  #new_vertices.append((-point[0], -point[1], -point[2], cr, cg, cb))
                  new_vertices[y, -x, 0] = z
                  next_image[y, -x, 0] = cr  # 赤成分
                  next_image[y, -x, 1] = cg  # 緑成分
                  next_image[y, -x, 2] = cb  # 青成分
    
    
    #next_image = interpolate_missing_pixels(next_image)
    next_image_rgb = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
    

    
    #"""
    # グレースケールに変換
    gray1 = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(next_image_rgb, cv2.COLOR_BGR2GRAY)
    
    # Optical Flowを計算
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    
    for y in range(0, 424, 5):
        for x in range(0, 512, 5):
            z = depth_img[y, x]
            if(x==0 and y==0):
                x_v, y_v, z_v, u_v, v_v, w_v, x_v2, y_v2, z_v2 = flow_vector1(flow, x, y, z, depth_img2)
                #print(f"x_v={x_v}, y_v={y_v}, z_v={z_v}, u_v={u_v}, v_v={v_v}, w_v={w_v}")
            else:
                #print(f"x_v={x_v}, y_v={y_v}, z_v={z_v}, u_v={u_v}, v_v={v_v}, w_v={w_v}")
                x_v, y_v, z_v, u_v, v_v, w_v, x_v2, y_v2, z_v2 = flow_vector2(flow, x, y, z, depth_img2, x_v, y_v, z_v, u_v, v_v, w_v, x_v2, y_v2, z_v2)
                
    
    # 各ベクトルを抽出
    #x, y, z, u, v, w = arrange[:, 0], arrange[:, 1], arrange[:, 2], arrange[:, 3], arrange[:, 4], arrange[:, 5]
    
    # 3Dプロットを作成
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    lw = 1.0
    l = 0.25
    alr = 0.1
    
    # ベクトルをプロット
    #ax.quiver(x_v, y_v, z_v, u_v, v_v, w_v, normalize=True)
    #ax.quiver(x_v, y_v, z_v, u_v, v_v, w_v, normalize=True, color=[0.0, 1.0, 0.0])
    #ax.quiver(x_v, z_v, y_v, u_v, w_v, v_v, linewidth=lw, length=l, arrow_length_ratio=alr)
    #ax.quiver(x_v, z_v, y_v, u_v, w_v, v_v, normalize=True, color=[0.0, 1.0, 0.0])
    
    colors = [(next_image[int(y), int(-x), :] / 255) for x, y in zip(x_v, y_v)]
    #ax.scatter(x_v, z_v, y_v, c=[next_image[y_v, x_v, 0], next_image[y_v, x_v, 1], next_image[y_v, x_v, 2]], marker='o')
    ax.scatter(x_v2, z_v2, y_v2, c=colors, marker='o', s=1)
    ax.quiver(x_v2, z_v2, y_v2, u_v, w_v, v_v, linewidth=lw, length=l, arrow_length_ratio=alr, color=[0.0, 1.0, 0.0])
    
    #max_value = max(z_v)
    #print("最大値:", max_value)
    
    # ラベルの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    # 値域を設定
    ax.set_xlim(max(x_v2), min(x_v2))
    ax.set_ylim(0, 4500)
    ax.set_zlim(max(y_v2), min(y_v2))
    
    plt.show()
    #plt.savefig(f'/Users/uchimurataichi/SceneFlow/SceneFlow_oak_floor/000{i}.png')
    

    print("現在のフレーム:", frame_number)
    
    # 次のフレームの処理に進む
    frame_number += 1
    
    depth_img = depth_img2.copy()
    image_rgb = next_image_rgb.copy()
    vertices = new_vertices.copy()

"""
# 画像を表示
cv2.imshow('Generated Image', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
#"""
cap_c.release()
