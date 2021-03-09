import os
import re
import numpy as np
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from paddleocr import PaddleOCR


ocr = PaddleOCR(use_angle_cls=True, det_model_dir='./weights/ocr/det/best',
                rec_model_dir='./weights/ocr/rec/best')


def pre_processing():
    # 删除并重新创建medium文件夹，存放经过Yolov5裁剪后的鞋标图片
    medium_dir = './dataset/medium_images/'
    shutil.rmtree(medium_dir)
    os.mkdir(medium_dir)


def clip_img(img, points, img_dir='./dataset/medium_images/', cor_img=True):
    p1, p2, p3, p4 = points

    x1, y1 = p1 - (p3 / 2), p2 - (p4 / 2)
    x4, y4 = p1 + (p3 / 2), p2 + (p4 / 2)
    img = Image.open(img)
    height, width = img.size
    cropped = img.crop((width * x1, height * y1, width * x4, height * y4))  # (left, upper, right, lower) 左上，右下
    # cropped.save('../dataset/new_2003/' + img_name)
    # print('YOLOv5 Complete...')
    # if cor_img:
    #     print('correct image...')
    #     return correct_img(cropped)
    return cropped


def correct_img(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if torch.cuda.is_available():
        CNN = torch.load('./weights/epoch30.pkl')
    else:
        CNN = torch.load('./weights/epoch30.pkl', map_location=torch.device('cpu'))

    cur_img = img
    init_img = cur_img[:]
    cur_img = cv2.resize(cur_img, (320, 320))
    cur_img = transform(cur_img)
    cur_img = torch.unsqueeze(cur_img, 0)
    if torch.cuda.is_available():
        cur_img = cur_img.cuda()
    img_res = torch.max(CNN(cur_img).cpu(), 1)[1].data.numpy()
    cor_img = np.rot90(init_img, k=-img_res[0])
    return cor_img


def get_ocr(img):
    result = ocr.ocr(img, cls=True)
    if not result:
        return
    ocr_text = ''
    for line in result:
        ocr_text += line[1][0] + ';'
    print(ocr_text)
    ocr_res = parse_1987_ocr(ocr_text)
    return ocr_res


def parse_ocr_for_all(img):
    result = ocr.ocr(img, cls=True)
    if not result:
        print('No output for ocr')
        return
    text = ''
    for line in result:
        text += line[1][0] + ';'
    print(text)
    # check if NIKE
    test = text[:]
    test = test.replace('i', '1', 30)
    test = test.replace('l', '1', 30)
    test = test.replace('O', '0', 30)
    date = re.findall('\d{2}/\d{2}/\d{2}', test, flags=0)
    if date:
        return parse_1987_ocr(text)

    return parse_ocr(text)


def parse_1987_ocr(text):
    type_list = {'XC', 'XH', 'XB', 'YS', 'VY', 'VH', 'VP', 'VYM', 'VW1', 'SQ', 'IY', 'LNM', 'LN4', 'MD', 'TT', 'Y3',
                 'QD', 'SZ', 'VF', 'VO2', 'JJS', 'VW', 'XG', 'LN3', 'VJ', 'LU1', 'VE', 'LN2', 'KW', 'IW', 'VT', 'JJ2',
                 'VTM', 'VT', 'VT2', 'JX', ''}
    ocr_list = []

    # Search origin
    for t in type_list:
        result = re.findall(';' + t + ';', text.upper(), flags=0)
        if result:
            ocr_list.append(t)
            break
        elif text.upper().split(';')[0] in type_list:
            ocr_list.append(text.upper().split(';')[0])

    if len(ocr_list) != 1:
        print('Origin not found...')
        return None

    # Search time
    test = text[:]
    test = test.replace('i', '1', 30)
    test = test.replace('l', '1', 30)
    test = test.replace('O', '0', 30)
    date = re.findall('\d{2}/\d{2}/\d{2}', test, flags=0)
    if len(date) >= 2:
        ocr_list.append(compare_date(date[0], date[1]))
    else:
        print('Time not found...')
        return None

    # Search type
    type = re.findall('......-...', text, flags=0)
    if type:
        ocr_list.append(type[0])
        return '_'.join(ocr_list)
    print('Type not found...')
    return None


def compare_date(date1, date2):
    m1, d1, y1 = [int(k) for k in date1.split('/')]
    m2, d2, y2 = [int(k) for k in date2.split('/')]
    if y1 > y2 or (y1 == y2 and m1 > m2) or (y1 == y2 and m1 == m2 and d1 > d2):
        return str(y2).zfill(2) + str(m2).zfill(2)
    else:
        return str(y1).zfill(2) + str(m1).zfill(2)


def parse_ocr(text):
    type_list = ['APC', 'APE', 'EVA', 'EVH', 'EVM', 'EVN', 'CLU', 'YYA', 'YYJ', 'PYV', 'PYA', 'PGD', 'PWI',
                 'LVL', 'PCI', 'PRB', 'PVN', 'PGS', 'VID', 'JUM', 'SHW', 'LHV', 'APY', 'APH', 'HWA', 'LHG']

    ocr_list = []
    ocr_res = ''

    # Search origin
    for t in type_list:
        result = re.findall(t, text.upper(), flags=0)
        if result:
            ocr_list.append(t)
            break
    if len(ocr_list) != 1:
        inner_dic = {'791004': 'EVH', '779001': 'APE', '791001': 'EVN',
                     '600001': 'CLU', '011001': 'APC', '702001': 'PYV',
                     '606001': 'YYA', '606004': 'YYJ', '046001': 'PYA',
                     '789006': 'PGD', '001001': 'PWI', '029002': 'LVL',
                     '789002': 'PCI', '698001': 'PRB', '059503': 'PVN',
                     '789005': 'PGS', '791005': 'EVA', '2J2001': 'VID',
                     '714001': 'JUM', '004001': 'EVM', '675001': 'SHW',
                     '029005': 'LHV', '779007': 'APY', '28I001': 'APH',
                     '281001': 'APH', '1Y3001': 'HWA', '029003': 'LHG',
                     '698007': 'PBB'}
        for key, value in inner_dic.items():
            if re.search(key, text):
                ocr_list.append(value)
                break
    if len(ocr_list) != 1:
        print('Origin Not Found...')
        return None

    # Search time
    test = text[:]
    test = test.replace('i', '1', 30)
    test = test.replace('l', '1', 30)
    test = test.replace('O', '0', 30)
    date = re.findall('\d{2}/\d{2}', test, flags=0)
    if date:
        if 0 < int(date[0].split('/')[0]) < 13 and 13 < int(date[0].split('/')[1]) < 21:
            dat = date[0].replace('/', '')
            ocr_list.append(dat)
    if len(ocr_list) != 2:
        inner_list = test.split(';')
        for s in inner_list:
            if len(s) > 3 and s[-4:].isdigit() and ('0' <= s[-4] <= '9'):
                if 0 < int(s[-4:-2]) < 13 and 13 < int(s[-2:]) < 22:
                    ocr_list.append(s[-4:-2] + s[-2:])
                    break
    if len(ocr_list) != 2:
        print('Time Not Found...')
        return None

    # Search type
    type = re.findall('ART.*?;', text, flags=0)
    if not type:
        type = re.findall('ABT.*?;', text, flags=0)
    if type:
        if len(type[0][3:-1].strip()) == 6 and '/' not in type[0][3:-1].strip():
            ocr_list.append(type[0][3:-1].strip())

    if len(ocr_list) != 3:
        print('Type Not Found...')
        return None
    return '_'.join(ocr_list)
