# Generate annotation for keras
# https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

from scipy import io as spio
from datetime import datetime

import numpy as np
import os
import shutil

if(os.path.exists("./dataset/imdb_crop/")):
    dataset_dir = ""
else:
    dataset_dir = "/Volumes/TB4/Keras/"

output_folder = "agegender_imdb"

imdb_dir = dataset_dir+"dataset/imdb_crop/"
gender_dir = dataset_dir+"dataset/"+output_folder+"/annotations/gender/"
age_dir = dataset_dir+"dataset/"+output_folder+"/annotations/age/"
age101_dir = dataset_dir+"dataset/"+output_folder+"/annotations/age101/"

if(not os.path.exists(dataset_dir+"dataset/"+output_folder+"/")):
    os.mkdir(dataset_dir+"dataset/"+output_folder)
if(not os.path.exists(dataset_dir+"dataset/"+output_folder+"/annotations/")):
    os.mkdir(dataset_dir+"dataset/"+output_folder+"/annotations")

if(not os.path.exists(gender_dir)):
    os.mkdir(gender_dir)
    os.mkdir(gender_dir+"train")
    os.mkdir(gender_dir+"train/f")
    os.mkdir(gender_dir+"train/m")
    os.mkdir(gender_dir+"validation")
    os.mkdir(gender_dir+"validation/f")
    os.mkdir(gender_dir+"validation/m")

if(not os.path.exists(age_dir)):
    os.mkdir(age_dir)
    os.mkdir(age_dir+"train")
    os.mkdir(age_dir+"train/0-2")
    os.mkdir(age_dir+"train/4-6")
    os.mkdir(age_dir+"train/8-13")
    os.mkdir(age_dir+"train/15-20")
    os.mkdir(age_dir+"train/25-32")
    os.mkdir(age_dir+"train/38-43")
    os.mkdir(age_dir+"train/48-53")
    os.mkdir(age_dir+"train/60-")
    os.mkdir(age_dir+"validation")
    os.mkdir(age_dir+"validation/0-2")
    os.mkdir(age_dir+"validation/4-6")
    os.mkdir(age_dir+"validation/8-13")
    os.mkdir(age_dir+"validation/15-20")
    os.mkdir(age_dir+"validation/25-32")
    os.mkdir(age_dir+"validation/38-43")
    os.mkdir(age_dir+"validation/48-53")
    os.mkdir(age_dir+"validation/60-")

if(not os.path.exists(age101_dir)):
    os.mkdir(age101_dir)
    os.mkdir(age101_dir+"train")
    os.mkdir(age101_dir+"validation")
    for i in range(0, 101):
        os.mkdir(age101_dir+"train/"+format(i, '03d'))
        os.mkdir(age101_dir+"validation/"+format(i, '03d'))


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def is_valid(face_score, second_face_score, age, gender):
    if face_score < 1.0:
        return False
    if (~np.isnan(second_face_score)) and second_face_score > 0.0:
        return False
    if ~(0 <= age <= 100):
        return False
    if np.isnan(gender):
        return False
    return True


def get_gender_dir(gender):
    if(gender == 0.0):
        return "f"
    return "m"


def get_age_dir(age):
    if(age >= 0 and age <= 3):
        return "0-2"
    if(age >= 4 and age <= 7):
        return "4-6"
    if(age >= 8 and age <= 14):
        return "8-13"
    if(age >= 15 and age <= 24):
        return "15-20"
    if(age >= 25 and age <= 37):
        return "25-32"
    if(age >= 38 and age <= 47):
        return "38-43"
    if(age >= 48 and age <= 59):
        return "48-53"
    return "60-"


meta = spio.loadmat(imdb_dir+"imdb.mat")

db = "imdb"

full_path = meta[db][0, 0]["full_path"][0]
dob = meta[db][0, 0]["dob"][0]
gender = meta[db][0, 0]["gender"][0]
photo_taken = meta[db][0, 0]["photo_taken"][0]
face_score = meta[db][0, 0]["face_score"][0]
second_face_score = meta[db][0, 0]["second_face_score"][0]
age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

for i in range(len(full_path)):
    if(not is_valid(face_score[i], second_face_score[i], age[i], gender[i])):
        continue
    print("path:"+str(full_path[i])+" gender:"+str(gender[i]))
    print(""+get_age_dir(age[i])+" "+get_gender_dir(gender[i]))
    train_or_validation = "train"
    if(i % 4 == 0):
        train_or_validation = "validation"
    src_img = imdb_dir+full_path[i][0]
    shutil.copyfile(src_img, age_dir+train_or_validation +
                    "/"+get_age_dir(age[i])+"/"+str(i)+".jpg")
    shutil.copyfile(src_img, gender_dir+train_or_validation +
                    "/"+get_gender_dir(gender[i])+"/"+str(i)+".jpg")
    shutil.copyfile(src_img, age101_dir+train_or_validation +
                    "/"+format(age[i], "03d")+"/"+str(i)+".jpg")
