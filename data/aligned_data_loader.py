import random
import numpy as np
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.image_folder import *
import scipy.io as sio
from builtins import object
import sys
import h5py

class IIWTestData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        final_img, target_1, sparse_path_1s  = next(self.data_loader_iter)
        return {'img_1': final_img, 'target_1': target_1}

class SAWData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def sparse_loader(self, sparse_path, num_features):
        # print("sparse_path  ", sparse_path)
        # sys.exit()
        hdf5_file_sparse = h5py.File(sparse_path,'r')
        B_arr = []
        data_whole = hdf5_file_sparse.get('/sparse/mn')
        mn = np.array(data_whole)
        mn = np.transpose(mn, (1,0))
        m = int(mn[0][0])
        n = int(mn[1][0])
        # print(m, n)
        data_whole = hdf5_file_sparse.get('/sparse/S')
        S_coo = np.array(data_whole)
        S_coo = np.transpose(S_coo, (1,0))
        S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

        # print(S_coo[:,0:2])
        # print(torch.FloatTensor([3, 4]))
        S_i = S_coo[0:2,:].long()
        S_v = S_coo[2,:].float()
        S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

        for i in range(num_features+1):
            data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
            B_coo = np.array(data_whole)
            B_coo = np.transpose(B_coo, (1,0))
            B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
            B_i = B_coo[0:2,:].long()
            B_v = B_coo[2,:].float()

            B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
            B_arr.append(B_mat)


        data_whole = hdf5_file_sparse.get('/sparse/N')
        N = np.array(data_whole)
        N = np.transpose(N, (1,0))
        N = torch.from_numpy(N)

        hdf5_file_sparse.close()
        return S, B_arr, N 


    def __next__(self):
        self.iter += 1
        final_img, target_1, sparse_path_1s = next(self.data_loader_iter)

        target_1['SS'] = []
        target_1['SB_list'] = [] 
        target_1['SN'] = []

        SS_1, SB_list_1, SN_1  = self.sparse_loader(sparse_path_1s[0], 2)

        for i in range(len(sparse_path_1s)):
            target_1['SS'].append(SS_1)
            target_1['SB_list'].append(SB_list_1)
            target_1['SN'].append(SN_1)

        return {'img_1': final_img, 'target_1': target_1}



class CGIntrinsicsData(object):
    def __init__(self, data_loader, root):
        self.data_loader = data_loader
        # self.fineSize = fineSize
        # self.max_dataset_size = max_dataset_size
        self.root = root
        # st()
        self.npixels = (256 * 256* 29)

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def sparse_loader(self, sparse_path, num_features):
        # print("sparse_path  ", sparse_path)
        # sys.exit()
        hdf5_file_sparse = h5py.File(sparse_path,'r')
        B_arr = []
        data_whole = hdf5_file_sparse.get('/sparse/mn')
        mn = np.array(data_whole)
        mn = np.transpose(mn, (1,0))
        m = int(mn[0][0])
        n = int(mn[1][0])
        # print(m, n)
        data_whole = hdf5_file_sparse.get('/sparse/S')
        S_coo = np.array(data_whole)
        S_coo = np.transpose(S_coo, (1,0))
        S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

        # print(S_coo[:,0:2])
        # print(torch.FloatTensor([3, 4]))
        S_i = S_coo[0:2,:].long()
        S_v = S_coo[2,:].float()
        S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

        for i in range(num_features+1):
            data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
            B_coo = np.array(data_whole)
            B_coo = np.transpose(B_coo, (1,0))
            B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
            B_i = B_coo[0:2,:].long()
            B_v = B_coo[2,:].float()

            B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
            B_arr.append(B_mat)


        data_whole = hdf5_file_sparse.get('/sparse/N')
        N = np.array(data_whole)
        N = np.transpose(N, (1,0))
        N = torch.from_numpy(N)

        hdf5_file_sparse.close()
        return S, B_arr, N 


    def create_CGIntrinsics_pair(self, path, gt_albedo, random_filp):

        super_pixel_path = self.root + "/CGIntrinsics/intrinsics_final/superpixels/" + path + ".mat"
        super_pixel_mat = sio.loadmat(super_pixel_path)
        super_pixel_mat = super_pixel_mat['data']
        
        final_list = []

        for i in range(len(super_pixel_mat)):
            pos =super_pixel_mat[i][0]

            if pos.shape[0] < 2:
                continue

            rad_idx = random.randint(0, pos.shape[0]-1)            
            final_list.append( (pos[rad_idx,0], pos[rad_idx,1]) )

        eq_list = []
        ineq_list = []

        row = gt_albedo.shape[0]
        col = gt_albedo.shape[1]

        for i in range(0,len(final_list)-1):
            for j in range(i+1, len(final_list)):
                y_1, x_1 = final_list[i]
                y_2, x_2 = final_list[j]

                y_1 = int(y_1*row)
                x_1 = int(x_1*col)
                y_2 = int(y_2*row)
                x_2 = int(x_2*col)

                # if image is flip
                if random_filp:
                    x_1 = col - 1 - x_1
                    x_2 = col - 1 - x_2

                if gt_albedo[y_1, x_1] < 2e-4 or gt_albedo[y_2, x_2] < 2e-4:
                    continue

                ratio = gt_albedo[y_1, x_1]/gt_albedo[y_2, x_2]

                if ratio < 1.05 and ratio > 1./1.05:
                    eq_list.append([y_1, x_1, y_2, x_2])
                elif ratio > 1.5:
                    ineq_list.append([y_1, x_1, y_2, x_2])               
                elif ratio < 1./1.5:
                    ineq_list.append([y_2, x_2, y_1, x_1])               

        eq_mat = np.asarray(eq_list)
        ineq_mat = np.asarray(ineq_list)

        if eq_mat.shape[0] > 0:
            eq_mat = torch.from_numpy(eq_mat).contiguous().float()
        else:
            eq_mat = torch.Tensor(1,1)


        if ineq_mat.shape[0] > 0:
            ineq_mat = torch.from_numpy(ineq_mat).contiguous().float()
        else:
            ineq_mat = torch.Tensor(1,1)


        return eq_mat, ineq_mat


    def __next__(self):
        self.iter += 1
        self.iter += 1
        scale =4 

        final_img, target_1, sparse_path_1s = next(self.data_loader_iter)

        target_1['eq_mat'] = []
        target_1['ineq_mat'] = []
        
        # This part will make training much slower, but it will improve performance
        for i in range(len(target_1["CGIntrinsics_ordinal_path"])):
            mat_path = target_1["CGIntrinsics_ordinal_path"][i]
            gt_R = target_1['gt_R'][i,0,:,:].numpy()
            random_filp = target_1['random_filp'][i]

            eq_mat, ineq_mat = self.create_CGIntrinsics_pair(mat_path, gt_R, random_filp)
            target_1['eq_mat'].append(eq_mat)
            target_1['ineq_mat'].append(ineq_mat)


        target_1['SS'] = []
        target_1['SB_list'] = [] 
        target_1['SN'] = []

        SS_1, SB_list_1, SN_1  = self.sparse_loader(sparse_path_1s[0], 2)

        for i in range(len(sparse_path_1s)):
            target_1['SS'].append(SS_1)
            target_1['SB_list'].append(SB_list_1)
            target_1['SN'].append(SN_1)

        return {'img_1': final_img, 'target_1': target_1}



class IIWData(object):
    def __init__(self, data_loader, flip):
        self.data_loader = data_loader
        # self.fineSize = fineSize
        # self.max_dataset_size = max_dataset_size
        self.flip = flip
        # st()
        self.npixels = (256 * 256* 29)

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def sparse_loader(self, sparse_path, num_features):
        # print("sparse_path  ", sparse_path)
        # sys.exit()
        hdf5_file_sparse = h5py.File(sparse_path,'r')
        B_arr = []
        data_whole = hdf5_file_sparse.get('/sparse/mn')
        mn = np.array(data_whole)
        mn = np.transpose(mn, (1,0))
        m = int(mn[0][0])
        n = int(mn[1][0])
        # print(m, n)
        data_whole = hdf5_file_sparse.get('/sparse/S')
        S_coo = np.array(data_whole)
        S_coo = np.transpose(S_coo, (1,0))
        S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

        # print(S_coo[:,0:2])
        # print(torch.FloatTensor([3, 4]))
        S_i = S_coo[0:2,:].long()
        S_v = S_coo[2,:].float()
        S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

        for i in range(num_features+1):
            data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
            B_coo = np.array(data_whole)
            B_coo = np.transpose(B_coo, (1,0))
            B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
            B_i = B_coo[0:2,:].long()
            B_v = B_coo[2,:].float()

            B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
            B_arr.append(B_mat)


        data_whole = hdf5_file_sparse.get('/sparse/N')
        N = np.array(data_whole)
        N = np.transpose(N, (1,0))
        N = torch.from_numpy(N)

        hdf5_file_sparse.close()
        return S, B_arr, N 

    def long_range_loader(self, h5_path):
        hdf5_file_read_img = h5py.File(h5_path,'r')        
        num_eq = hdf5_file_read_img.get('/info/num_eq')
        num_eq = np.float32(np.array(num_eq))
        num_eq = int(num_eq[0][0])


        if num_eq > 0:
            equal_mat = hdf5_file_read_img.get('/info/equal')
            equal_mat = np.float32(np.array(equal_mat))
            equal_mat = np.transpose(equal_mat, (1, 0))
            equal_mat = torch.from_numpy(equal_mat).contiguous().float()
        else:
            equal_mat = torch.Tensor(1,1)
 
        num_ineq = hdf5_file_read_img.get('/info/num_ineq')
        num_ineq = np.float32(np.array(num_ineq))
        num_ineq = int(num_ineq[0][0])

        if num_ineq > 0:
            ineq_mat = hdf5_file_read_img.get('/info/inequal')
            ineq_mat = np.float32(np.array(ineq_mat))
            ineq_mat = np.transpose(ineq_mat, (1, 0))
            ineq_mat = torch.from_numpy(ineq_mat).contiguous().float()
        else:
            ineq_mat = torch.Tensor(1,1)


        hdf5_file_read_img.close()

        return equal_mat, ineq_mat

    def __next__(self):
        self.iter += 1
        self.iter += 1
        scale =4 

        final_img, target_1, sparse_path_1s = next(self.data_loader_iter)

        target_1['eq_mat'] = []
        target_1['ineq_mat'] = []

        for i in range(len(target_1["mat_path"])):
            mat_path = target_1["mat_path"][i]
            eq_mat, ineq_mat = self.long_range_loader(mat_path)
            target_1['eq_mat'].append(eq_mat)
            target_1['ineq_mat'].append(ineq_mat)


        target_1['SS'] = []
        target_1['SB_list'] = [] 
        target_1['SN'] = []

        SS_1, SB_list_1, SN_1  = self.sparse_loader(sparse_path_1s[0], 2)

        for i in range(len(sparse_path_1s)):
            target_1['SS'].append(SS_1)
            target_1['SB_list'].append(SB_list_1)
            target_1['SN'].append(SN_1)

        return {'img_1': final_img, 'target_1': target_1}

class CGIntrinsics_DataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir):
        transform = None
        dataset = CGIntrinsicsImageFolder(root=_root, \
                list_dir =_list_dir)

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle= True, num_workers=int(2))
        self.dataset = dataset
        flip = False    
        self.paired_data = CGIntrinsicsData(self.data_loader, _root)

    def name(self):
        return 'CGIntrinsics_DataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset)



class SAWDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, mode):
        # BaseDataLoader.initialize(self)
        # self.fineSize = opt.fineSize

        transform = None
 
        dataset = SAW_ImageFolder(root=_root, \
                list_dir =_list_dir, mode = mode, is_flip = True, transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 16, shuffle= True, num_workers=int(2))

        self.dataset = dataset
        # flip = False
        self.saw_data = SAWData(data_loader)

    def name(self):
        return 'sawDataLoader'

    def load_data(self):
        return self.saw_data

    def __len__(self):
        return len(self.dataset)


class IIWDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, mode):
        # BaseDataLoader.initialize(self)
        # self.fineSize = opt.fineSize

        # transformations = [
            # TODO: Scale
            #transforms.CenterCrop((600,800)),
            # transforms.Scale(256, Image.BICUBIC),
            # transforms.ToTensor() ]
        transform = None
        # transform = transforms.Compose(transformations)

        # Dataset A
        # dataset = ImageFolder(root='/phoenix/S6/zl548/AMOS/test/', \
                # list_dir = '/phoenix/S6/zl548/AMOS/test/list/',transform=transform)
        # testset 
        dataset = IIW_ImageFolder(root=_root, \
                    list_dir =_list_dir, mode = mode, is_flip = True, transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 16, shuffle= True, num_workers=int(2))

        self.dataset = dataset
        flip = False
        self.iiw_data = IIWData(data_loader, flip)

    def name(self):
        return 'iiwDataLoader'

    def load_data(self):
        return self.iiw_data

    def __len__(self):
        return len(self.dataset)


class RenderDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir):
        # BaseDataLoader.initialize(self)
        transform = None

        dataset = Render_ImageFolder(root=_root, \
                list_dir =_list_dir, transform=transform)

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle= True, num_workers=int(1))
        self.dataset = dataset

    def name(self):
        return 'renderDataLoader'

    def load_data(self):
        return self.data_loader

    def __len__(self):
        return len(self.dataset)


class IIWTESTDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, mode):

        transform = None
        dataset = IIW_ImageFolder(root=_root, \
                list_dir =_list_dir, mode= mode, is_flip = False, transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle= False, num_workers=int(1))
        self.dataset = dataset
        self.iiw_data = IIWTestData(data_loader)

    def name(self):
        return 'IIWTESTDataLoader'

    def load_data(self):
        return self.iiw_data

    def __len__(self):
        return len(self.dataset)

