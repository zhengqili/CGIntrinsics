import numpy as np
import torch
import os
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import sys, traceback
import h5py
import os.path
import scipy.misc
import torch.nn as nn
import json
from . import saw_utils
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage.measurements import label
from skimage.transform import resize


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Intrinsics_Model(BaseModel):
    def name(self):
        return 'Intrinsics_Model'

    def __init__(self, opt):

        which_model_netG = "pix2pix"#"pix2pix"

        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)

        # define tensors
        print("LOAD Unet pix2pix version")
        output_nc = 3
        model = networks.define_G(opt.input_nc, output_nc, opt.ngf, 
                                        opt.which_model_netG, 'batch', opt.use_dropout, self.gpu_ids)

        # # TESTING
        if not self.isTrain:
            model_parameters = self.load_network(model, 'G', 'cgintrinsics_iiw_saw_final')
            model.load_state_dict(model_parameters)

        self.netG = model

        # model_parameters = self.load_network(self.netG, 'G', '_best_SUNCG_saw_iiw_intrinsics')
        # self.netG.load_state_dict(model_parameters)            

        self.lr = opt.lr
        self.old_lr = opt.lr
        self.netG.train()

        # if self.isTrain:            
        self.criterion_joint = networks.JointLoss() 
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.0002, betas=(0.9, 0.999))
        
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input, targets):
        self.num_pair = input.size(0)
        self.input.resize_(input.size()).copy_(input)
        self.targets = targets


    def forward_both(self):
        self.input_images = Variable(self.input.float().cuda(), requires_grad = False)        
        self.prediction_R, self.prediction_S  = self.netG.forward(self.input_images)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self, epoch, data_set_name):
        # Combined loss
        self.loss_joint = self.criterion_joint(self.input_images , self.prediction_R, self.prediction_S, 
                                                    self.targets, data_set_name, epoch)
        print("trainning loss is %f"%self.loss_joint)
        self.loss_joint_var = self.criterion_joint.get_loss_var()
        self.loss_joint_var.backward()

        return self.loss_joint

    def optimize_intrinsics(self, epoch, data_set_name):
        self.forward_both()
        self.optimizer_G.zero_grad()        

        joint_loss = self.backward_G(epoch, data_set_name)

        self.optimizer_G.step()
        return joint_loss

    def evlaute_iiw(self, input_, targets):
        # switch to evaluation mode
        input_images = Variable(input_.cuda() , requires_grad = False)
        prediction_R, prediction_S  = self.netG.forward(input_images)
        return self.criterion_joint.evaluate_WHDR(prediction_R, targets)

    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        # self.save_network(self.netR, 'R', label, self.gpu_ids)
        # self.save_network(self.netL, 'L', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        self.lr = lr
        # for param_group in self.optimizer_D.param_groups:
            # param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def scaled_learning_rate(self, rate = 2.0):
        # lrd = self.opt.lr /
        lr = self.old_lr /rate
        self.lr = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


    def compute_pr(self, pixel_labels_dir, splits_dir, dataset_split, class_weights, bl_filter_size, img_dir, thres_count=400):

        thres_list = saw_utils.gen_pr_thres_list(thres_count)
        photo_ids = saw_utils.load_photo_ids_for_split(
            splits_dir=splits_dir, dataset_split=dataset_split)

        plot_arrs = []
        line_names = []

        fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
        title = '%s Precision-Recall' % (
            {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],
        )

        print("FN ", fn)
        print("title ", title)

        # compute PR 
        rdic_list = self.get_precision_recall_list_new(pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_ids=photo_ids, class_weights=class_weights, bl_filter_size = bl_filter_size, img_dir=img_dir)

        plot_arr = np.empty((len(rdic_list) + 2, 2))

        # extrapolate starting point 
        plot_arr[0, 0] = 0.0
        plot_arr[0, 1] = rdic_list[0]['overall_prec']

        for i, rdic in enumerate(rdic_list):
            plot_arr[i+1, 0] = rdic['overall_recall']
            plot_arr[i+1, 1] = rdic['overall_prec']

        # extrapolate end point
        plot_arr[-1, 0] = 1
        plot_arr[-1, 1] = 0.5

        AP = np.trapz(plot_arr[:,1], plot_arr[:,0])

        # plot_arr = np.empty((len(rdic_list), 2))
        # for i, rdic in enumerate(rdic_list):
        #     plot_arr[i, 0] = rdic['overall_recall']
        #     plot_arr[i, 1] = rdic['overall_prec']


        # AP = 0
        # for i in range(len(rdic_list)-1):
        #     R_n_1 =  rdic_list[i]['overall_recall']
        #     R_n = rdic_list[i+1]['overall_recall']

        #     if R_n_1 < 0.1 or R_n < 0.1:
        #         continue

        #     if R_n_1 > 0.975 or R_n > 0.975:
        #         continue

        #     P_n = rdic_list[i+1]['overall_prec']
        #     AP += (R_n - R_n_1)*P_n

        # AP = AP + 0.1

        return AP


    def get_precision_recall_list_new(self, pixel_labels_dir, thres_list, photo_ids,
                                  class_weights, bl_filter_size, img_dir):

        output_count = len(thres_list)
        overall_conf_mx_list = [
            np.zeros((3, 2), dtype=int)
            for _ in xrange(output_count)
        ]

        count = 0 
        total_num_img = len(photo_ids)

        for photo_id in (photo_ids):
            print("photo_id ", count, photo_id, total_num_img)
            # load photo using photo id, hdf5 format 
            img_path = img_dir + str(photo_id) + ".png"

            saw_img = saw_utils.load_img_arr(img_path)
            original_h, original_w = saw_img.shape[0], saw_img.shape[1]
            saw_img = saw_utils.resize_img_arr(saw_img)

            saw_img = np.transpose(saw_img, (2,0,1))
            input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
            input_images = Variable(input_.cuda() , requires_grad = False)

            # run model on the image to get predicted shading 
            # prediction_S , rgb_s = self.netS.forward(input_images)
            prediction_R, prediction_S = self.netG.forward(input_images)
            # prediction_Sr = prediction_S.repeat(1,3,1,1)

            # Write predicted images
            # prediction_R_np = prediction_R.data[0,:,:,:].cpu().numpy()
            # prediction_S_np = prediction_S.data[0,:,:,:].cpu().numpy()
            # np_img = input_images.data[0,:,:,:].cpu().numpy()

            # output_path = "/phoenix/S6/zl548/SUNCG/intrinsics3/CGI_IIW_SAW_plot/" + str(photo_id) + ".h5"
            # hdf5_file_write = h5py.File(output_path,'w')

            # print(output_path)

            # hdf5_file_write.create_dataset("/prediction/img", data = np_img)            
            # hdf5_file_write.create_dataset("/prediction/R", data = prediction_R_np)
            # hdf5_file_write.create_dataset("/prediction/S", data = prediction_S_np)

            # hdf5_file_write.close()
            # end of prediction

            # output_path = root + '/phoenix/S6/zl548/SAW/prediction/' + str(photo_id) + ".png.h5"
            prediction_Sr = torch.exp(prediction_S)
            # prediction_Sr = torch.pow(prediction_Sr, 0.4545)
            prediction_S_np = prediction_Sr.data[0,0,:,:].cpu().numpy() 
            prediction_S_np = resize(prediction_S_np, (original_h, original_w), order=1, preserve_range=True)

            # compute confusion matrix
            conf_mx_list = self.eval_on_images( shading_image_arr = prediction_S_np,
                pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
                photo_id=photo_id, bl_filter_size = bl_filter_size, img_dir=img_dir
            )

            for i, conf_mx in enumerate(conf_mx_list):
                # If this image didn't have any labels
                if conf_mx is None:
                    continue
                overall_conf_mx_list[i] += conf_mx

            count += 1

            ret = []
            for i in xrange(output_count):
                overall_prec, overall_recall = saw_utils.get_pr_from_conf_mx(
                    conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
                )

                ret.append(dict(
                    overall_prec=overall_prec,
                    overall_recall=overall_recall,
                    overall_conf_mx=overall_conf_mx_list[i],
                ))



        return ret


    def eval_on_images(self, shading_image_arr, pixel_labels_dir, thres_list, photo_id, bl_filter_size, img_dir):
        """
        This method generates a list of precision-recall pairs and confusion
        matrices for each threshold provided in ``thres_list`` for a specific
        photo.

        :param shading_image_arr: predicted shading images

        :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

        :param thres_list: List of shading gradient magnitude thresholds we use to
        generate points on the precision-recall curve.

        :param photo_id: ID of the photo we want to evaluate on.

        :param bl_filter_size: The size of the maximum filter used on the shading
        gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
        """

        shading_image_linear_grayscale = shading_image_arr
        shading_image_linear_grayscale[shading_image_linear_grayscale < 1e-4] = 1e-4
        shading_image_linear_grayscale = np.log(shading_image_linear_grayscale)

        shading_gradmag = saw_utils.compute_gradmag(shading_image_linear_grayscale)
        shading_gradmag = np.abs(shading_gradmag)

        if bl_filter_size:
            shading_gradmag_max = maximum_filter(shading_gradmag, size=bl_filter_size)

        # We have the following ground truth labels:
        # (0) normal/depth discontinuity non-smooth shading (NS-ND)
        # (1) shadow boundary non-smooth shading (NS-SB)
        # (2) smooth shading (S)
        # (100) no data, ignored
        y_true = saw_utils.load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)
        
        img_path = img_dir+ str(photo_id) + ".png"

        # diffuclut and harder dataset
        srgb_img = saw_utils.load_img_arr(img_path)
        srgb_img = np.mean(srgb_img, axis = 2)
        img_gradmag = saw_utils.compute_gradmag(srgb_img)

        smooth_mask = (y_true == 2)
        average_gradient = np.zeros_like(img_gradmag)
        # find every connected component
        labeled_array, num_features = label(smooth_mask)
        for j in range(1, num_features+1):
            # for each connected component, compute the average image graident for the region
            avg = np.mean(img_gradmag[labeled_array == j])
            average_gradient[labeled_array == j]  = avg

        average_gradient = np.ravel(average_gradient)

        y_true = np.ravel(y_true)
        ignored_mask = y_true > 99

        # If we don't have labels for this photo (so everything is ignored), return
        # None
        if np.all(ignored_mask):
            return [None] * len(thres_list)

        ret = []
        for thres in thres_list:
            y_pred = (shading_gradmag < thres).astype(int)
            y_pred_max = (shading_gradmag_max < thres).astype(int)
            y_pred = np.ravel(y_pred)
            y_pred_max = np.ravel(y_pred_max)
            # Note: y_pred should have the same image resolution as y_true
            assert y_pred.shape == y_true.shape

            # confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
            confusion_matrix = saw_utils.grouped_weighted_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask], average_gradient[~ignored_mask])
            ret.append(confusion_matrix)

        return ret
