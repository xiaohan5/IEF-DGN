import pandas as pd
import gc
import random
import re
import json
import time
import numpy as np
from datetime import date
import math

import tarfile
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nowplaying', help='tmall,nowplaying,tafeng,dunnhumby,instacart,valuedshopper')
args = parser.parse_args()

curr_dir = os.curdir()

def Tmall_process(curr_dir):
    df = pd.read_csv(curr_dir+"dataset15.csv")
    data = {}
    item_dict = {}
    list2 = []
    session = 0
    sess_num = 0
    num = 1
    max_length = 0
    length = 0
    for row in df.iterrows():
        list1 = row[1][0].split('\t')
        del list1[3]
        data_list = list(map(int, list1))
        if data_list[0] not in data:
            session = data_list[2]
            data[data_list[0]] = []
            max_length = max(max_length, length)
            length = 1
            if data_list[1] not in item_dict:
                item_dict[data_list[1]] = num
                num += 1
            sess_num += 1
            list2.append(item_dict[data_list[1]])
        else:
            if session != data_list[2]:
                sess_num += 1
                session = data_list[2]
                data[data_list[0]].append(list2)
                list2 = []
            if data_list[1] not in item_dict:
                item_dict[data_list[1]] = num
                num += 1
            length += 1
            list2.append(item_dict[data_list[1]])
    return data, len(item_dict), max_length, sess_num

def NowPlaying_process(curr_dir):
    df = pd.read_csv(curr_dir+"nowplaying.csv")
    data = {}
    item_dict = {}
    list2 = []
    session = 0
    sess_num = 0
    num = 1
    max_length = 0
    length = 0
    for row in df.iterrows():
        list1 = row[1][0].split(' ')
        del list1[4]
        del list1[3]
        data_list = list(map(int, list1))
        if data_list[0] not in data:
            session = data_list[1]
            data[data_list[0]] = []
            max_length = max(max_length, length)
            length = 1
            if data_list[2] not in item_dict:
                item_dict[data_list[2]] = num
                num += 1
            sess_num += 1
            list2.append(item_dict[data_list[2]])
        else:
            if session != data_list[1]:
                sess_num += 1
                session = data_list[1]
                data[data_list[0]].append(list2)
                list2 = []
            if data_list[2] not in item_dict:
                item_dict[data_list[2]] = num
                num += 1
            length += 1
            list2.append(item_dict[data_list[2]])
    print(data)
    return data, len(item_dict), max_length, sess_num

if __name__ == '__main__':
    if args.dataset == "tmall":
        data, item_num, max_length, sess_num = Tmall_process(curr_dir)
    elif args.dataset == "nowplaying":
        data, item_num, max_length, sess_num = NowPlaying_process(curr_dir)

    file = open('tmall.txt', 'w') 
    file.write(str(data))
    print(item_num,max_length, sess_num)
    data, item_num, max_length, sess_num = NowPlaying_process(curr_dir)
    file.close()
    file = open('nowplaying.txt', 'w') 
    file.write(str(data))
    print(item_num,max_length, sess_num)
    file.close()
