# This file will use to score your implementations.
# You should not change this file

import os
import pandas as pd
import time
import sys
import cv2

from simple_ocr import HoadonOCR

if __name__ == "__main__":
    input_folder = sys.argv[1]                      #sampledata- file anh test

    #tên file nhãn là tham số thứ 2
    label_file = sys.argv[2]                        #file chua nhan file .csv

    #đọc file csv
    df_labels = pd.read_csv(label_file)                     #đọc dữ liệu file .csv
    img_name = df_labels['img_name'].values.tolist()        #cột img_name
    img_labels = df_labels['label'].values.tolist()         #cột label
    img_to_label = dict([(img_name[i], img_labels[i]) for i in range(len(img_name))])

    '''
    gia tri cua img_to_labels:
    {'img1.jpeg': 'highlands', 'img2.jpeg': 'highlands', 'img3.jpeg': 'highlands', 'img4.jpeg': 'phuclong', 'img5.jpeg': 'phuclong', 
    'img6.jpeg': 'starbucks', 'img7.jpeg': 'starbucks', 'img8.jpeg': 'starbucks', 'img9.jpeg': 'others', 'img10.jpeg': 'others'}
    '''

    #kiểm tra thời gian chạy
    start_time = time.time()
    model = HoadonOCR()                             #khoi tao doi tuong
    init_time = time.time() - start_time
    #print("Run time in: %.2f s" % init_time)        #thời gian khởi tạo mô hình

    list_files = os.listdir(input_folder)           #file sampledata ở tham số truyền vào
    '''
    print("Gia tri cua list_file: \n", list_files)
    ['img2.jpeg', 'img5.jpeg', 'img10.jpeg', 'img4.jpeg', 'img7.jpeg', 'img6.jpeg', 'img8.jpeg', 'img1.jpeg', 'img3.jpeg', 'img9.jpeg']
    '''
    print("Total test images: ", len(list_files))   #tổng số ảnh test (có trong input_forder)
    fail_process = 0
    cnt_predict = 0


    nameOfImageFail = ''


    start_time = time.time()
    for filename in list_files:                     # tên các file ảnh test, 10 lần lặp
        img = cv2.imread(os.path.join(input_folder, filename))  #đọc từng ảnh test một,  img là ảnh màu
        #print(img.shape)                            # kích thước của từng ảnh test
        try:
            label = model.find_label(img)           #hàm này CẦN CÀI ĐẶT - label: Hghtland, Starbuck, Phúc long hoặc other
        except:
            label = -1

        if img_to_label[filename] == label:         #nếu cùng tên thư mục
            cnt_predict += 1
        elif label == -1:
            fail_process += 1
        elif img_to_label[filename] != label:
            print("Anh",filename ,"sai: ", label, "Ten dung:", img_to_label[filename])
    run_time = time.time() - start_time             #thời gian chạy

    print("Ket qua dung: %i/%i" % (cnt_predict, len(list_files)))
    print("Lỗi: %i" % fail_process)
    print("Score = %.2f" % (10.*cnt_predict/len(list_files)))
    print("Run time in: %.2f s" % run_time)
