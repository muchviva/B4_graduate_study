import os

#指定されたディレクトリ内に存在するサブディレクトリのリストを取得
def get_subdirs(input_path):
    '''
    get a list of subdirectories in input_path directory
    :param input_path: parent directory (in which to get the subdirectories)
    :return:
    subdirs: list of subdirectories in input_path
    '''
    subdirs = [os.path.join(input_path, dir_i) for dir_i in os.listdir(input_path)
              if os.path.isdir(os.path.join(input_path, dir_i))]
    subdirs.sort()
    return subdirs

#指定されたディレクトリ内のANU IKEA Datasetからカテゴリのパス、スキャンのパス、RGB画像、深度画像、法線画像のパス、各デバイスに対するRGBおよび深度の内部パラメータファイルのパスを返す
def get_scan_list(input_path, devices='all'):
    '''
    get_scan_list retreieves all of the subdirectories under the dataset main directories:
    :param input_path: path to ANU IKEA Dataset directory

    :return:
    scan_path_list: path to all available scans
    category_path_list: path to all available categories
    '''

    category_path_list = get_subdirs(input_path)
    scan_path_list = []
    for category in category_path_list:
        if os.path.basename(category) != 'Calibration':
            category_scans = get_subdirs(category)
            for category_scan in category_scans:
                scan_path_list.append(category_scan)

    rgb_path_list = []
    depth_path_list = []
    normals_path_list = []
    rgb_params_files = []
    depth_params_files = []
    for scan in scan_path_list:
        device_list = get_subdirs(scan)
        for device in device_list:
            if os.path.basename(device) in devices:
                rgb_path = os.path.join(device, 'images')
                depth_path = os.path.join(device, 'depth')
                normals_path = os.path.join(device, 'normals')
                if os.path.exists(rgb_path):
                    rgb_path_list.append(rgb_path)
                    rgb_params_files.append(os.path.join(device, 'ColorIns.txt'))
                if os.path.exists(depth_path):
                    if 'dev3' in device:  # remove redundant depths - remove this line for full 3 views
                        depth_path_list.append(depth_path)
                        depth_params_files.append(os.path.join(device, 'DepthIns.txt'))
                if os.path.exists(normals_path):
                    normals_path_list.append(normals_path)

    return category_path_list, scan_path_list, rgb_path_list, depth_path_list, depth_params_files, rgb_params_files, normals_path_list
