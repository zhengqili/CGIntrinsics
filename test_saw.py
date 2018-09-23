import time
import torch
import numpy as np
from options.train_options import TrainOptions
import sys, traceback
import h5py
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.data_loader import CreateDataLoader_TEST
from data.data_loader import CreateDataLoaderIIWTest

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


root = "/home/zl548/phoenix24/"
full_root = root +'/phoenix/S6/zl548/'

model = create_model(opt)

total_steps = 0
best_loss = 100


def test_SAW(model):
    # parameters for SAW 
    pixel_labels_dir = full_root + '/CGIntrinsics/SAW/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
    splits_dir = full_root + '/CGIntrinsics/SAW/saw_splits/'
    img_dir = full_root + "/CGIntrinsics/SAW/saw_images_512/"
    dataset_split = 'E'
    class_weights = [1, 1, 2]
    bl_filter_size = 10

    print("============================= Validation ON SAW============================")
    model.switch_to_eval()
    AP = model.compute_pr(pixel_labels_dir, splits_dir,
                dataset_split, class_weights, bl_filter_size, img_dir)

    print("Current AP: %f"%AP)
    model.switch_to_train()
    return AP


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch =0




# current_loss = validation(model, dataset, dataset_size)
# print("best loss %f, current loss %f",best_loss, current_loss)
# sys.exit()
print("WE ARE IN TESTING PHASE!!!!")
# test(model, dataset_L,dataset_size_L)
test_SAW(model)
print("WE ARE DONE TESTING!!!")


print("We are done")
