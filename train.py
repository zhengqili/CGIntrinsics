import time
import torch
import sys
from scipy import misc
import h5py
import numpy as np

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoaderCGIntrinsics
from data.data_loader import CreateDataLoaderIIW
from data.data_loader import CreateDataLoaderIIWTest
from data.data_loader import CreateDataLoaderRender
from data.data_loader import CreateDataLoaderSAW

from models.models import create_model
import torch
import math

batch_size = 16

train_on_IIW = False
train_on_SAW = False


opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
root = "/home/zl548/phoenix24/"
full_root = root +'/phoenix/S6/zl548/'

train_list_CGIntrinsics = full_root + '/CGIntrinsics/intrinsics_final/train_list/'
data_loader_S = CreateDataLoaderCGIntrinsics(full_root, train_list_CGIntrinsics, batch_size)

train_list_Render = full_root + '/CGIntrinsics/intrinsics_final/render_list/'
data_loader_Render = CreateDataLoaderRender(full_root, train_list_Render, batch_size)

if train_on_IIW:
    train_list_IIW = full_root + '/CGIntrinsics/IIW/train_list/'
    data_loader_IIW = CreateDataLoaderIIW(full_root, train_list_IIW, 0, batch_size)
    dataset_IIW = data_loader_IIW.load_data()
    dataset_size_IIW = len(data_loader_IIW)
    print('#train_list_IIW images = %d' % dataset_size_IIW)
    iterator_IIW = iter(dataset_IIW)

if train_on_SAW:
    train_list_SAW = full_root + '/CGIntrinsics/SAW/train_list/'
    data_loader_SAW = CreateDataLoaderSAW(full_root, train_list_SAW, 0, batch_size)
    dataset_SAW = data_loader_SAW.load_data()
    dataset_size_SAW = len(data_loader_SAW)
    print('#train_list_SAW images = %d' % dataset_size_SAW)
    iterator_SAW = iter(dataset_SAW)

dataset_CGIntrinsics = data_loader_S.load_data()
dataset_Render = data_loader_Render.load_data()

dataset_size_CGIntrinsics = len(data_loader_S)


print("#train_list CGIntrinsics Intrinsics ", dataset_size_CGIntrinsics)


num_iterations = dataset_size_CGIntrinsics/batch_size
model = create_model(opt)
model.switch_to_train()


total_steps = 0
best_loss = 100


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch =0


print(best_loss)
print("WE ARE GOOD")

validation_t = int(round(num_iterations/10))

print("validation_interval = %d"%validation_t)

iterator_Render = iter(dataset_Render)



os_t = 0
count = 0 
count_saw =0 
count_iiw = 0
num_orientation = 5


for epoch in range(0, 50):
    if epoch > 0 and epoch % 16 ==0:
        model.scaled_learning_rate(rate=2.)

    for i, data in enumerate(dataset_CGIntrinsics):
        print('CGIntrinsics Intrinsics: epoch %d, iteration %d, best_loss %f num_iterations %d best_epoch %d' % (epoch, i, best_loss, num_iterations, best_epoch) )
        stacked_img = data['img_1']
        targets = data['target_1']

        data_set_name = 'CGIntrinsics'
        model.set_input(stacked_img, targets)
        model.optimize_intrinsics(epoch, data_set_name)        


        if train_on_IIW:
        # optimize for IIW or IIW
        # if count%2 == 0:
            print('IIW Intrinsics: %d epoch %d, iteration %d, best_loss %f num_iterations %d best_epoch %d' % (count_iiw%num_orientation, epoch, i, best_loss, num_iterations, best_epoch) )
            data_IIW = next(iterator_IIW, None)

            if data_IIW is None:
                count_iiw +=1
                data_loader_IIW= CreateDataLoaderIIW(full_root, train_list_IIW, count_iiw%num_orientation)
                dataset_IIW = data_loader_IIW.load_data()
                iterator_IIW = iter(dataset_IIW)
                data_IIW= next(iterator_IIW, None)

            data_set_name = "IIW"
            stacked_img = data_IIW['img_1']
            targets = data_IIW['target_1']

            model.set_input(stacked_img, targets)
            model.optimize_intrinsics(epoch, data_set_name)

        if train_on_SAW:
            print('SAW Intrinsics: %d epoch %d, iteration %d, best_loss %f num_iterations %d best_epoch %d' % (count_saw%num_orientation, epoch, i, best_loss, num_iterations, best_epoch) )
            data_SAW = next(iterator_SAW, None)

            if data_SAW is None:
                count_saw +=1
                data_loader_SAW= CreateDataLoaderSAW(full_root, train_list_SAW, count_saw%num_orientation)
                dataset_SAW = data_loader_SAW.load_data()
                iterator_SAW = iter(dataset_SAW)
                data_SAW = next(iterator_SAW, None)

            data_set_name = "SAW"
            stacked_img = data_SAW['img_1']
            targets = data_SAW['target_1']

            model.set_input(stacked_img, targets)
            model.optimize_intrinsics(epoch, data_set_name)

        # Optimize for small number of super high quality rendered images
        os_t += 1
        if os_t % 10 == 0:
            # START Opedsurface
            data_R = next(iterator_Render, None)
            # end of one epoch 
            if data_R is None:
                iterator_Render = iter(dataset_Render)
                data_R = next(iterator_Render, None)

            stacked_img_OS= data_R['img_1']
            targets_OS = data_R['target_1']
            data_set_name = 'Render'

            model.set_input(stacked_img_OS, targets_OS)
            model.optimize_intrinsics(epoch, data_set_name)

            print('Render: epoch %d, iteration %d, best_loss %f num_iterations %d best_epoch %d' % (epoch, i, best_loss, num_iterations, best_epoch) )
            #  END Rendering 


print("we are done!!!!!")
