import pickle

import math
import os
import json
import sys
import copy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import yaml
import shutil
from tqdm import tqdm

# from matplotlib import pyplot as plt

exist = lambda target_path: os.path.exists(target_path)
list_files = lambda x: list_dir_files(x, with_dir_path=True, list_flag=1)
list_dirs = lambda x: list_dir_files(x, with_dir_path=True, list_flag=2)
list_files_with_suffix = lambda x, suffix: list_with_suffix(x, suffix, with_dir_path=True)


def list_dir_files(dir_path, with_dir_path=False, list_flag=0):
    # 0代表全部需要（文件夹+文件）
    # 1代表只需要文件
    # 2代表只需要文件夹
    assert isinstance(list_flag, int) and 0 <= list_flag < 3
    if list_flag == 0:
        file_list = os.listdir(dir_path)  # 过滤掉文件夹
    elif list_flag == 1:
        file_list = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    else:
        file_list = [file for file in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, file))]
    if with_dir_path:
        file_list = [os.path.join(dir_path, file) for file in file_list]
    return file_list


def list_with_suffix(dir_path, suffix_list=None, with_dir_path=True):
    file_list = list_dir_files(dir_path, with_dir_path)
    if isinstance(suffix_list, str):
        suffix_list = [suffix_list]

    if suffix_list is not None:
        suffix_list = set(map(lambda x: x.lower().split(".")[-1], suffix_list))  # 只保留suffix，不要'.'+去重
        new_file_list = []
        for file_path in file_list:
            file_suffix = file_path.split(".")[-1].lower()
            if file_suffix in suffix_list:
                new_file_list.append(file_path)
        return new_file_list
    return file_list


def get_filename(file_path, need_suffix=False):
    if need_suffix:
        return os.path.basename(file_path)
    else:
        return os.path.basename(file_path).split(".")[0]


def get_suffix(file_path):
    return file_path.split(".")[-1]


def read_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()


def copy_once(source_path, target_dir_path):
    if not os.path.exists(source_path):
        print(f"file: {source_path}  not exist")
        return
        # print("{}/{}".format(idx + 1, len(source_path_list)))
    target_path = connect_path(target_dir_path, os.path.basename(source_path))
    shutil.copy(source_path, target_path)


def mv_once(source_path, target_dir_path):
    if not os.path.exists(source_path):
        print(f"file: {source_path}  not exist")
        return
        # print("{}/{}".format(idx + 1, len(source_path_list)))
    target_path = connect_path(target_dir_path, os.path.basename(source_path))
    shutil.move(source_path, target_path)


def mv2dir(source_path_list, target_dir_path):
    create_not_exist(target_dir_path)
    pool = ThreadPool(8)
    results = pool.imap(lambda x: mv_once(*x),
                        zip(source_path_list, repeat(target_dir_path)))
    pbar = tqdm(results, total=len(source_path_list))
    for _ in pbar:
        pbar.desc = "copy..."
    pbar.close()
    #
    pool.close()


def copy2dir(source_path_list, target_dir_path):
    create_not_exist(target_dir_path)
    pool = ThreadPool(8)
    results = pool.imap(lambda x: copy_once(*x),
                        zip(source_path_list, repeat(target_dir_path)))
    pbar = tqdm(results, total=len(source_path_list))
    for _ in pbar:
        pbar.desc = "copy..."
    pbar.close()
    #
    pool.close()


def add_suffix(path, suffix, filename=None):
    if filename is None:
        res = path + "." + suffix
    else:
        res = os.path.join(path, get_filename(filename)) + "." + suffix
    return res


def create_not_exist(target_path_list):
    if isinstance(target_path_list, str):
        target_path_list = [target_path_list]
    for target_path in target_path_list:
        if not os.path.exists(target_path):
            os.makedirs(target_path)


def connect_path(*paths):
    return os.path.join(*paths)


def del_file_list(file_list):
    for file in file_list:
        os.remove(file)


def show_process(title, cur, total):
    print("{} {}/{}".format(title, cur, total))


def show_process2(title, cur, total_list):
    print("{} {}/{}".format(title, cur + 1, len(total_list)))


def delete_file(file_list):
    if isinstance(file_list, list):
        for file_path in file_list:
            os.remove(file_path)
    else:
        os.remove(file_list)


def delete_file_exist(target_file_path):
    if os.path.exists(target_file_path) and os.path.isfile(target_file_path):
        os.remove(target_file_path)


def delete_dir_exist(target_dir_path):
    if os.path.exists(target_dir_path) and os.path.isdir(target_dir_path):
        shutil.rmtree(target_dir_path)


def reset_dir(target_dir_path):
    delete_dir_exist(target_dir_path)
    create_not_exist(target_dir_path)


def write_txt_file(txt, txt_path):
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(txt)


def load_config(config_path):
    assert os.path.exists(config_path), "Config file: {} is not exists".format(config_path)

    if config_path.endswith(".yaml"):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    elif config_path.endswith(".json"):
        # ISO-8859-1
        with open(config_path, "r", encoding="ISO-8859-1") as config_file:
            config = json.loads(config_file.read(), encoding="ISO-8859-1")
    else:
        raise ValueError("the config type is not support")
    return config


def load_json(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    return data
def is_path_creatable(pathname: str) -> bool:
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def cal_center(box):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.
    y_center = (y_min + y_max) / 2.
    return x_center, y_center


def cal_dist(center1, center2):
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def cal_area(box):
    x_min, y_min, x_max, y_max = box

    return abs((x_max - x_min) * (y_max - y_min))
