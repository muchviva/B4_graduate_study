def get_rc_from_xyz(x, y, z, depth_ins):
    '''
    project a 3d point back to the image row and column indices
    :param point: xyz 3D point
    :param depth_ins: depth camera intrinsic parameters
    :return:
    '''
    c = int((x * depth_ins['fx'] / z) + depth_ins['cx'])
    r = int((y * depth_ins['fy'] / z) + depth_ins['cy'])
    return c, r