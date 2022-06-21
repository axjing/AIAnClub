import os
import random
from common import is_mkdir


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)
def split(img_path, save_path, val_rate=0.5):
    is_mkdir(save_path)
    assert os.path.exists(img_path), "path: '{}' does not exist.".format(img_path)

    files_name = sorted([file.split(".")[0] for file in os.listdir(img_path)])
    print(files_name)
    all_files = []
    for n in range(len(files_name)):
        nm_path = os.path.join(img_path, files_name[n])
        for f in os.listdir(nm_path):
            all_files.append(os.path.join(files_name[n], f.split(".")[0]))
    print(all_files)
    files_num = len(all_files)
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(all_files):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)
    print(save_path)
    try:
        train_f = open(os.path.join(save_path, "train.txt"), "x")
        eval_f = open(os.path.join(save_path, "val.txt"), "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    # main()
    # img_path = "/home/jing/Coding/DATA/IAobj/JPEGImages"
    img_path = "/home/jing/Coding/DATA/IAobj/Annotations"
    save_path = "/home/jing/Coding/DATA/IAobj/ImageSets/Main"

    split(img_path, save_path, 0.3)

if __name__ == '__main__':
    main()
