# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:01:06 2020

@author: wesley yam
"""



import os
import requests



HOME_PATH = "D:/"
RELATIVE_PATH = "AI4ImpactProject2/"

FILE_DIR = HOME_PATH + RELATIVE_PATH


def create_directory(file_dir):
    try:
        # Create target Directory
        os.mkdir(file_dir)
        print("file_dir " , file_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , file_dir ,  " already exists")
        
def download_one_csv_file(CSV_URL,data_name,is_training_set):
    if is_training_set:
        STORE_DATA_DIR = FILE_DIR + "Train/"
    else:
        STORE_DATA_DIR = FILE_DIR + "Test/"
    with open(STORE_DATA_DIR+data_name+".csv", 'wb+') as f, \
            requests.get(CSV_URL, stream=True) as r:  
                f.write(r.content)




#create home dir test dir and train dir
create_directory(FILE_DIR)
create_directory(FILE_DIR+"Test/")
create_directory(FILE_DIR+"Train/")



#url setup
TRAIN_URL = "https://ai4impact.org/P003/"
TEST_URL= "https://ai4impact.org/P003/historical/"


DATA_NAME = "guitrancourt,lieusaint,lvs-pussay,parc-du-gatinais,arville,boissy-la-riviere,angerville-1,angerville-2"
DATA_NAME = DATA_NAME.split(',')


#download data
for data_name in DATA_NAME:
    download_one_csv_file(TRAIN_URL+data_name+".csv",data_name,True)

for data_name in DATA_NAME:
    download_one_csv_file(TEST_URL+data_name+".csv",data_name,False)

