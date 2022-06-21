import os
import re
import pickle
import shutil

import torch
import SimpleITK as itk
import nibabel as nib
import numpy as np
import logging
import cv2
import logging
import scipy.ndimage
import matplotlib.pyplot as plt
import imageio




# from cv.Resample import resample_image

def match_str(m_str, key_str=None):
    res = m_str.rsplit(key_str, 1)
    return res


def is_file_path(data):
    """
    判断是否为文件
    :param data:
    :return:
    """
    return os.path.isfile(data)


def is_dir(data):
    """
    判断是否为文件夹
    :param data:
    :return:
    """
    return os.path.isdir(data)


def is_mkdir(dir):
    """
    如果不存在创建文件夹
    :param dir:
    :return:
    """
    if os.path.exists(dir) == False:
        os.mkdir(dir)
    return 0


def normalization_spacing(src_dir, dst_dir):
    statistic_nii_gz(src_dir, dst_dir)


def write_file(data, fname, root=""):
    '''
    将data写入文件中
    :param data:写入内容
    :param fname:文件名
    :param root:根目录
    :return:
    '''
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))

    return 0


def get_file_list(dir):
    """
    获得当前目录文件列表
    :param dir:
    :return:
    """
    fl = os.listdir(dir)
    fl = [f for f in fl]
    return fl


def get_file_path(dir, format):
    """
    递归获得当下目录下所有文件列表
    :param dir:
    :param format:
    :return:
    """
    file_paths = []
    file_dirs = []
    file_names = []
    for dirpath, dirname, filename in os.walk(dir):
        for file in filename:

            # if os.path.splitext(file)[-1] == format:
            if format in file:
                all_path = os.path.join(dirpath, file)
                file_paths.append(all_path)
                file_dirs.append(dirpath)
                file_names.append(file)
                # print("DIR-->:\t",dirpath)
    return file_paths, file_dirs, file_names


def nib_load(file_name):
    """
    使用nible加载nii文件
    :param file_name:
    :return:
    """
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()

    return data

def nii_gz_load(path):
    original_img = itk.ReadImage(path)
    return original_img
def nib_load_pro(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    data=proxy.get_data()

    print("data.shape:\n", data.shape)
    print(f'With the unique values: {np.unique(data)}')
    # proxy.uncache()

    return proxy, data


def savepkl(data, path):
    '''
    保存pkl文件
    :param data:数据
    :param path:保存路径
    :return:
    '''
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_label(paths):
    # x y z -> z, y, z
    labels = [nib.load(path).get_fdata().astype(np.int8).transpose(2, 1, 0).clip(0, 1) for path in paths]
    label = np.bitwise_or.reduce(labels, axis=0)
    return label


def copy_file(srcfile, dstpath):
    if not os.path.exists(srcfile):
        print("{} not exist!".format(srcfile))
        return -1
    else:
        fp, fn = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, os.path.join(dstpath, fn))
        logging.info("Copy File: {} To Dir:{}".format(srcfile, dstpath))
        return 0


modalities = ('flair', 't1ce', 't1', 't2')


def nii_to_dcm(nii_path, dcm_dir):
    if not os.path.isdir(dcm_dir):
        msg = "Folder " + dcm_dir + " No Exist And Needs To Be Created"
        logging.warn(msg)
        os.mkdir(dcm_dir)
    img = itk.ReadImage(nii_path)
    img = itk.GetArrayFromImage(img)
    for i in range(img.shape[0]):
        select_img = itk.GetImageFromArray(img[i])
        itk.WriteImage(select_img, dcm_dir + "/series_" + str(img.shape[0] - i) + '.dcm')
        msg = "Saved:" + dcm_dir + "/series_" + str(img.shape[0] - i) + '.dcm'
        logging.info(msg)
    msg = "nii.gz To dcm Successful"
    logging.info(msg)
    return 0


def dcm_to_nii(dcm_dir, nii_path):
    '''
    :param dcm_dir:dicom存放文件夹
    :param nii_path:
    :return:
    '''

    series_IDs = itk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir)
    print(dcm_dir, series_IDs)
    series_file_names = itk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir)
    series_reader = itk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    image3D = series_reader.Execute()
    itk.WriteImage(image3D, nii_path)
    msg = "{},To {} Successful".format(dcm_dir, nii_path)
    logging.info(msg)
    return 0


def adjust_window_level_width(img, window_center, window_width):
    '''
    针对CT影响，调整窗位窗宽
    :param img:[512,512,320]
    :param window_center:窗位     CT为90   85
    :param window_width:窗宽         35   40
    :return:
    '''

    win_min = (2 * window_center - window_width) / 2.0 + 0.5
    win_max = (2 * window_center + window_width) / 2.0 + 0.5
    for i in range(img.shape[-1]):
        img[..., i] = 255.0 * (img[..., i] - win_min) / (win_max - win_min)

    return img


def look_slice_label_Z(mask, Z_number=128):
    if mask.shape[-1] == Z_number:
        Z = (0, mask.shape[-1])
    else:
        log_read = []
        for i in range(mask.shape[-1]):
            if mask[..., i].max() > 0:
                # num=np.unique(data)
                log_read.append(i)
                # print("第{0:} Slice".format(i + 1))

        start = log_read[0]
        end = log_read[-1]
        strat_Z = start - 2
        end_Z = end + 2
        # strat_Z, end_Z = calculate_index(start, end, Z_number)
        print("Z Start:", strat_Z, " End:", end_Z, " Number:", end_Z - strat_Z)
        Z = (strat_Z, end_Z)
        return Z


def calculate_index(start, end, number_slice):
    '''
    根据筛选个数计算索引位置
    :param start:
    :param end:
    :param number_slice:
    :return:
    '''
    if number_slice > (end - start):

        residue_ = number_slice - (end - start)
        start_ = start - (residue_ // 2)
        if start_<0:
            start_=start
            end_=start+number_slice
        else:
            end_ = end + (residue_ // 2) + (residue_ % 2)
    else:
        start_ = start
        end_ = start + number_slice
    return start_, end_
def find_index(lengh,label_ind, number_slice):
    '''
    根据筛选个数计算索引位置
    :param start:
    :param end:
    :param number_slice:
    :return:
    '''

    assert lengh>=number_slice,"索引个数{}大于总长度{}".format(number_slice,lengh)

    start,end=label_ind
    residue_ = abs(number_slice - (end - start))
    if number_slice > (end - start):
        start_ = start - (residue_ // 2)
        end_ = end + (residue_ // 2) + (residue_ % 2)
        cha=lengh-number_slice

    else:
        start_ = start+ (residue_ // 2)
        end_ = start - ((residue_ // 2) + (residue_ % 2))
    return start_, end_

def get_mask_center_index(mask,crop_size=(256,256,128)):
    if torch.is_tensor(mask):
        mask=mask.numpy()
    for i in range(len(crop_size)):
        assert mask.shape[i]>=crop_size[i], "mask.shape:{},crop_size:{}".format(mask.shape,crop_size)


    where=np.where(mask>0)
    where_label= [np.unique(np.sort(where[i])) for i in range(len(where))]

    index_x=(where_label[0][0],where_label[0][-1])
    index_y=(where_label[1][0],where_label[1][-1])
    index_z=(where_label[2][0],where_label[2][-1])

    # x_ind=calculate_index(index_x[0],index_x[1],crop_size[0])
    # y_ind=calculate_index(index_y[0],index_y[1],crop_size[1])
    # z_ind=calculate_index(index_z[0],index_z[1],crop_size[2])

    x_ind = find_index(mask.shape[0], index_x, crop_size[0])
    y_ind = find_index(mask.shape[1], index_y, crop_size[1])
    z_ind = find_index(mask.shape[2], index_z, crop_size[2])

    return x_ind,y_ind,z_ind

def look_slice_label_X_Y_Z(mask, X_number=256, Y_number=128, Z_number=128):
    '''
    筛选出X,Y,Z不同轴上存在肿瘤的切片，
    :param data:label数据
    :param X_number:个数
    :param Y_number:
    :param Z_number:
    :return:
    '''

    log_read = []
    for i in range(mask.shape[0]):
        if mask[i, ...].max() > 0:
            log_read.append(i)
    start = log_read[0]
    end = log_read[-1]
    # print("There are {} slices that have labels".format(str(end - start)))
    strat_X, end_X = calculate_index(start, end, X_number)
    depth = mask.shape[0]
    if strat_X < 0:  # 解决当索引的一边超出原图边界时
        strat_X = 0
        end_X = X_number
    if strat_X > depth:
        strat_X = strat_X - end_X + depth
        end_X = depth
    # print("X Start:", strat_X, "End:", end_X)

    X = (strat_X, end_X)

    log_read = []
    for i in range(mask.shape[1]):
        if mask[:, i, ...].max() > 0:
            log_read.append(i)
            # print("第{0:} Slice".format(i + 1))
            # print(num)
    start = log_read[0]
    end = log_read[-1]
    strat_Y, end_Y = calculate_index(start, end, Y_number)
    depth = mask.shape[1]
    if strat_Y < 0:  # 解决当索引的一边超出原图边界时
        strat_Y = 0
        end_Y = Y_number
    if end_Y > depth:
        strat_Y = strat_Y - end_Y + depth
        end_Y = depth
    # print("Y Start:", strat_Y, "End:", end_Y)

    Y = (strat_Y, end_Y)

    log_read = []
    for i in range(mask.shape[-1]):
        if mask[..., i].max() > 0:
            # num=np.unique(data)
            log_read.append(i)
            # print("第{0:} Slice".format(i + 1))

    start = log_read[0]
    end = log_read[-1]
    strat_Z, end_Z = calculate_index(start, end, Z_number)
    depth = mask.shape[-1]
    # print("strat_Z:{}, end_Z:{}".format(strat_Z, end_Z))
    if strat_Z < 0:  # 解决当索引的一边超出原图边界时
        strat_Z = 0
        end_Z = Z_number
    if end_Z > depth:
        strat_Z = strat_Z - end_Z + depth
        end_Z = depth
    # print("Z Start:", strat_Z, "End:", end_Z)

    Z = (strat_Z, end_Z)

    return [X, Y, Z]


def set_nii_info(raw_nii, dst_nii):
    dst_nii.SetSpacing(raw_nii.GetSpacing())
    dst_nii.SetDirection(raw_nii.GetDirection())
    dst_nii.SetOrigin(raw_nii.GetOrigin())
    # dst_nii.SetPixel(raw_nii.GetPixelIDValue())
    return dst_nii


def statistic_nii_gz(f_dir, save_dir, out_spacing=(0.42899999022483826, 0.42899999022483826, 0.3000030517578125)):
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    file_paths, file_dirs, file_names = get_file_path(f_dir, format=".gz")
    for fp, fd, fn in zip(file_paths, file_dirs, file_names):

        if "-label." not in fp:
            label_path = fp.split(".nii.gz")[0] + "-label.nii.gz"

            p_nm = fp.split(".nii.gz")[0].split("/")[-1]
            raw2dst_path = os.path.join(save_dir, p_nm)
            if os.path.exists(raw2dst_path) == False:
                os.mkdir(raw2dst_path)
            img2dst_path = os.path.join(raw2dst_path, p_nm + ".nii.gz")
            lab2dst_path = os.path.join(raw2dst_path, p_nm + "-label.nii.gz")

            original_img = nii_gz_load(fp)
            img_arr = itk.GetArrayFromImage(original_img)

            original_lab = nii_gz_load(label_path)
            lab_arr = itk.GetArrayFromImage(original_lab)

            if original_img.GetSize()[-1] > 320:
                img_arr = img_arr[original_img.GetSize()[-1] - 450:, ...]
                print(img_arr.shape)
                sitk_img = itk.GetImageFromArray(img_arr)
                sitk_img_info = set_nii_info(original_img, sitk_img)
                # sitk_img_info=resample_image(sitk_img_info,out_spacing)
                print(sitk_img.GetSize())
                lab_arr = lab_arr[original_lab.GetSize()[-1] - 450:, ...]

                sitk_lab = itk.GetImageFromArray(lab_arr)
                sitk_lab_info = set_nii_info(original_lab, sitk_lab)
                # sitk_lab_info= resample_image(sitk_lab_info,out_spacing)
                # sitk_lab.SetPixel()
                itk.WriteImage(sitk_img_info, img2dst_path)
                itk.WriteImage(sitk_lab_info, lab2dst_path)

            else:
                # sitk_lab_info=resample_image(original_lab,out_spacing)
                # sitk_img_info=resample_image(original_img,out_spacing)
                shutil.copy(label_path, lab2dst_path)
                shutil.copy(fp, img2dst_path)
            # print('Image 转换后 {} 图像的Spacing：{}'.format(fp, sitk_img_info.GetSpacing()))
            # print('Image 转换后 {} 图像的Size：{}'.format(fp, sitk_img_info.GetSize()))
            # print('lable 转换后 {} 图像的Spacing：{}'.format(fp, sitk_lab_info.GetSpacing()))
            # print('lable 转换后 {} 图像的Size：{}'.format(fp, sitk_lab_info.GetSize()))


def save_slice_img(data, save_path):
    """
    保存3D中的每个切片的图片
    :param data:
    :param save_path:
    :return:
    """
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    for i in range(data.shape[-1]):
        img = data[..., i]
        img_path = os.path.join(save_path, str(i) + ".png")
        cv2.imwrite(img_path, img)
        msg = "Save " + img_path + " Successful"
        logging.info(msg)
    return 0


def save_pred(pred, labels, save_dir, dataset, idx_batch):
    """
    保存预测的图像快照
    :param pred:预测数据
    :param labels:真实标签
    :param save_dir:保存路径
    :param dataset:数据加载类，其中有对应的路径
    :param idx_batch:batch的索引
    :return:
    """

    name = dataset.paths[idx_batch].split("/")[-1].split(".")[0] + ".png"
    # print(name)
    #
    tmp = np.zeros((512, 512, 3))
    pred = pred.cpu().data.numpy()
    pred = pred[0, 0, :, :] * 255
    pred = pred.T
    pred = cv2.resize(pred, (512, 512))
    # cv2.imwrite(r"{}/{}".format(save_dir, name), np.uint8(pred))
    tmp[:, :, 1] = pred
    #
    labels = labels.cpu().data.numpy()
    labels = labels[0, :, :] * 255
    labels = labels.T
    labels = cv2.resize(labels, (512, 512))
    tmp[:, :, 2] = labels  # red :lable
    # tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"{}/{}".format(save_dir, name), tmp)


def input_mask_output_gpu_2_cpu(inputs, masks, outputs):
    """
    将放入gpu的数据转换成cpu，用于可视化
    :param inputs:[batch_size,channel,H,W,D]
    :param masks:[batch_size,H,W,D]
    :param outputs:[batch_size,num_class,H,W,D],默认为2分类，若是多分类情况需要另行处理
    :return:inputs_cpu, masks_cpu, outputs_cpu
    """
    inputs_cpu = inputs.cpu().data.numpy()
    masks_cpu = masks.cpu().data.numpy().astype(np.float64)
    masks_cpu = np.expand_dims(masks_cpu, 1)
    outputs_cpu = torch.sigmoid(outputs).cpu().data.numpy().astype(np.float64)
    if outputs.dim()==5:
        # outputs_cpu = outputs.cpu().data.numpy().astype(np.float64)
        outputs_cpu = np.expand_dims(outputs_cpu[:, 1, :, :, :], 1)
    elif outputs.dim()==4:
        outputs_cpu = np.expand_dims(outputs_cpu[:, 1, :, :], 1)

    return inputs_cpu, masks_cpu, outputs_cpu

def input_mask_output_to_cpu(inputs, masks, outputs):
    """
        将放入gpu的数据转换成cpu中的numpy类型
        :param inputs:[batch_size,channel,H,W,D]
        :param masks:[batch_size,H,W,D]
        :param outputs:[batch_size,num_class,H,W,D],默认为2分类，若是多分类情况需要另行处理
        :return:inputs_cpu, masks_cpu, outputs_cpu
        """
    inputs_cpu = inputs.cpu().data.numpy()
    masks_cpu = masks.cpu().data.numpy().astype(np.float64)
    # masks_cpu = np.expand_dims(masks_cpu, 1)
    outputs_cpu = torch.sigmoid(outputs).cpu().data.numpy().astype(np.float64)
    return inputs_cpu, masks_cpu, outputs_cpu

def draw_img_true_pred(raw_img, raw_lab, pre_lab, save_path=""):
    """
    :param raw_img:原图，兼容原图矩阵输入和路径输入
    :param raw_lab:真实标签，兼容原图矩阵输入和路径输入
    :param pre_lab:预测标签
    :param save_path:保存路径
    :return:
    """

    if os.path.isfile(raw_img) and os.path.isfile(raw_lab) and os.path.isfile(pre_lab):
        raw_img = cv2.imread(raw_img)
        raw_lab = cv2.imread(raw_lab)
        pre_lab = cv2.imread(pre_lab)

    raw_im = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    raw_lb = cv2.cvtColor(raw_lab, cv2.COLOR_BGR2GRAY)
    pre_lb = cv2.cvtColor(pre_lab, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(raw_im, 127, 255, cv2.THRESH_BINARY)
    _, raw_lb_threshold = cv2.threshold(raw_lb, 127, 255, cv2.THRESH_BINARY)
    _, pre_lb_threshold = cv2.threshold(pre_lb, 127, 255, cv2.THRESH_BINARY)

    raw_contours, hierarchy = cv2.findContours(raw_lb_threshold, 3, 2)
    cnt = raw_contours[0]
    img_color1 = cv2.cvtColor(raw_im, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color1, [cnt], 0, (0, 255, 255), 1)

    pre_contours, hierarchy = cv2.findContours(pre_lb_threshold, 3, 2)
    cnt_pre = pre_contours[0]
    cv2.drawContours(img_color1, [cnt_pre], 0, (255, 0, 255), 1)

    plt.imshow(img_color1)
    plt.show()

    if os.path.isfile(save_path):
        cv2.imwrite(save_path, img_color1)


def image_to_video(img_path, video_path, format=".png"):
    '''
    img_path:image path format:png,jpg,jpeg...
    video_path:save video path format:avi,mp3
    '''
    fps = 24  # 视频每秒24帧
    size = (512, 512)  # 需要转为视频的图片的尺寸
    # 可以使用cv2.resize()进行修改
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    file_paths, file_dirs, file_names = get_file_path(img_path, format)
    for fp in file_paths:
        img = cv2.imread(fp)
        video.write(img)
    video.release()
    # cv2.destroyAllWindows()


def image_to_video_of_name(img_path, video_dir, format=".png"):
    """
    将序列图片转换为视频
    :param img_path:
    :param video_dir:
    :param format:
    :return:
    """
    fps = 24  # 视频每秒24帧
    size = (512, 512)  # 需要转为视频的图片的尺寸, 可以使用cv2.resize()进行修改

    # 可以使用cv2.resize()进行修改
    file_paths, file_dirs, file_names = get_file_path(img_path, format)

    res_l = []
    tmp = ""
    for fp in file_paths:
        # print(fp)
        fp_split = match_str(fp, "_")
        res_l.append(fp_split)
    # print(res_l)
    res = []

    for i in range(len(res_l)):
        res_tmp = []
        flag = 0
        for j in range(len(res_l) - 1):
            if res_l[i][0] == res_l[j + 1][0]:
                if flag == 0:
                    name = res_l[i][0].rsplit("/", 1)[-1] + ".avi"
                    save_path = os.path.join(video_dir, name)
                    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
                    flag += 1
                path = res_l[i][0] + "_" + res_l[i][1]
                img = cv2.imread(path)
                video.write(img)
                res_tmp.append(res_l[i][0] + "_" + res_l[i][1])

        video.release()
        res.append(res_tmp)


def add_dir_mv_file(src_path, video_dir):
    """
    将序列图像转换为视频
    :param src_path:
    :param video_dir:
    :return:
    """
    file_paths, file_dirs, file_names = get_file_path(src_path, "png")

    split_path = []
    split_dict = {}
    for fp in file_paths:
        # print(fp)
        fp_split = match_str(fp, "_")
        number = fp_split[1].split(".")[0]
        format_key = fp_split[1].split(".")[1]
        if fp_split[0] in split_dict:
            split_dict[fp_split[0]] += " " + number
        else:
            split_dict[fp_split[0]] = number

    fps = 24  # 视频每秒24帧
    size = (512, 512)  # 需要转为视频的图片的尺寸
    # 可以使用cv2.resize()进行修改

    for k in split_dict.keys():
        v = split_dict[k]
        # v=v.split(" ")
        v = list(map(int, v.split(" ")))
        v = sorted(v)
        split_dict[k] = v

        name = k.rsplit("/", 1)[-1] + ".avi"

        video_path = os.path.join(video_dir, name)
        print("*" * 15, video_path, "*" * 15)
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        for i in v:
            img_path = k + "_" + str(i) + "." + format_key

            img = cv2.imread(img_path)
            video.write(img)
        video.release()

    print(split_dict)


if __name__ == "__main__":
    path="/home/dell/jing/Coding/DATA/FromIA/train/C1xD240xH356xW356_3D/huangxiuping_C40W120/huangxiuping_C40W120_data_f32.pkl"
