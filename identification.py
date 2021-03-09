import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import my_utils
import detect
import shutil
import argparse
from Network import Net


def detect_ocr(image_path: str):
    # 利用yolo模型检测鞋标位置point
    img_points = detect.detect(source=image_path)
    # 利用鞋标定位point裁剪鞋标
    detect_img = my_utils.clip_img(image_path, img_points)
    # 临时性保存鞋标
    detect_img.save('./test-image.jpg')
    # ocr检测
    ocr_res = my_utils.parse_ocr_for_all('./test-image.jpg')
    print('ocr res: ', ocr_res)
    return detect_img, ocr_res


img = './test-image/7af6ee448e18d64b922126386639e2a4.jpg'
img = './test-image/7c9cda8d260920df2abbdb7a2cbf9236.jpg'
detect_ocr(img)



def process(source):

    img_points = detect.detect(source='')

    os.mkdir(res_dir)
    origin_dir, img_list = my_utils.get_img_list(source)

    success_cnt = 0
    for idx, img_name in enumerate(img_list):
        if not img_name.endswith('.jpg'):
            continue
        my_utils.pre_processing()
        img_points = detect.detect(source=origin_dir + img_name)
        if img_points:
            print(img_points)
        else:
            print('Shoe Tag Not Found!')
            continue
        my_utils.clip_img(origin_dir + img_name, img_points)
        ocr_res = my_utils.get_ocr(medium_dir + img_name)
        print(ocr_res)
        if ocr_res:
            if not os.path.exists(res_dir + ocr_res):
                os.mkdir(res_dir + ocr_res)
            shutil.copyfile(medium_dir + img_name, res_dir + ocr_res + '/' + img_name)
            success_cnt += 1
        print('\n\n\n\n%d / %d complete...' % (idx+1, len(img_list)))
        print('%d success in %d samples, trans_rate: %.2f' % (success_cnt, idx+1, success_cnt/(idx+1)))


# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', type=str, default='./data', help='source')  # file/folder, 0 for webcam
#     opt = parser.parse_args()
#     print(opt)
#     process(source=opt.source)
