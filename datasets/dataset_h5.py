from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import re

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py

import histomicstk
from histomicstk.preprocessing.color_deconvolution.find_stain_index import (
    find_stain_index)
from histomicstk.preprocessing.color_deconvolution.stain_color_map import (
    stain_color_map)

from mpi4py import MPI



def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)
        
    trnsfrms_val = transforms.Compose(
                    [
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

def resize224_imagenet():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
        
    trnsfrms_val = transforms.Compose(
                    [
                     transforms.Resize(224),
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class Whole_Slide_Bag(Dataset):
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
        train_augm=False
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)
            
        self.train_augm = train_augm

        self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            if self.train_augm:
                coord = hdf5_file['coords'][idx//8]
            else:
                coord = hdf5_file['coords'][idx]
        
        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)

        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord
    
class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1,
        unmixing = False,
        separate_stains_method = None,
        file_bgr = None,
        bgr = None,
        delete_third_stain = False, 
        convert_to_rgb = False,
        color_norm = False,
        color_norm_method = None,
        save_images_to_h5 = False,
        image_h5_dir = None
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained=pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        elif custom_transforms == 'resize224_imagenet':
            self.roi_transforms = resize224_imagenet()
        elif custom_transforms == 'nothing':
            self.roi_transforms = transforms.Compose([])
        else:
            raise NotImplementedError('Please define this custom transform first!')

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            
            if self.patch_size == target_patch_size:
                target_patch_size = -1
        
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None
                
            dset = f['coords'][()]
            coords_dict = dict(f['coords'].attrs.items())
            
        self.summary()
        
        self.unmixing = unmixing
        self.separate_stains_method = separate_stains_method
        self.file_bgr = file_bgr
        self.bgr = bgr
        self.delete_third_stain = delete_third_stain
        self.convert_to_rgb = convert_to_rgb

        self.color_norm = color_norm
        self.color_norm_method = color_norm_method
                
        if self.unmixing or (self.color_norm and self.color_norm_method == 'macenko_pca'):
            if self.bgr is None:
                if self.file_bgr is not None:
                    df_bgr = pd.read_csv(self.file_bgr, index_col=0)
                    self.bgr = np.asarray(re.split("\[|.\]|. ", df_bgr[df_bgr.index.str.startswith(os.path.basename(os.path.splitext(self.file_path)[0]))].bgr_intensity.values[0])[1:-1]).astype('float')
                else:
                    raise ValueError
            else:
                self.bgr = np.asarray(re.split("\[|.\]|. ", self.bgr)[1:-1]).astype('float')

        self.save_images_to_h5 = save_images_to_h5
        self.image_h5_dir = image_h5_dir
        
        if self.save_images_to_h5:
            if self.image_h5_dir is not None:
                os.makedirs(self.image_h5_dir, exist_ok=True)
                with h5py.File(os.path.join(self.image_h5_dir, os.path.basename(self.file_path)), "w", driver='mpio', comm=MPI.COMM_WORLD) as f_img:
                    f_img.create_dataset('coords', data=dset)
                    for key, value in coords_dict.items():
                        f_img['coords'].attrs[key] = value
                    if target_patch_size > 0:
                        dset = f_img.create_dataset('imgs', shape=(len(self), target_patch_size, target_patch_size, 3), 
                                                   maxshape=(None, target_patch_size, target_patch_size, 3), 
                                                   chunks=(1, target_patch_size, target_patch_size, 3), dtype='uint8')
                    else:
                        dset = f_img.create_dataset('imgs', shape=(len(self), self.patch_size, self.patch_size, 3), 
                                                   maxshape=(None, self.patch_size, self.patch_size, 3), 
                                                   chunks=(1, self.patch_size, self.patch_size, 3), dtype='uint8')
            else:
                raise ValueError
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        
        if self.separate_stains_method == 'macenko_pca' or self.color_norm_method == 'macenko_pca':
            def _reorder_stains(W, stains=None):
                        """Reorder stains in a stain matrix to a specific order.
                    
                        This is particularly relevant in macenco where the order of stains is not
                        preserved during stain unmixing, so this method uses
                        histomicstk.preprocessing.color_deconvolution.find_stain_index
                        to reorder the stains matrix to the order provided by this parameter
                    
                        Parameters
                        ------------
                        W : np array
                            A 3x3 matrix of stain column vectors.
                        stains : list, optional
                            List of stain names (order is important). Default is H&E.
                    
                        Returns
                        ------------
                        np array
                            A re-ordered 3x3 matrix of stain column vectors.
                    
                        """
                        stains = ['hematoxylin', 'eosin'] if stains is None else stains
                    
                        assert len(stains) == 2, "Only two-stain matrices are supported for now."
                    
                        def _get_channel_order(W):
                            first = find_stain_index(stain_color_map[stains[0]], W)
                            second = 1 - first
                            third = 2
                            return first, second, third
                    
                        def _ordered_stack(mat, order):
                            return np.stack([mat[..., j] for j in order], -1)
                    
                        return _ordered_stack(W, _get_channel_order(W))
        
        if self.unmixing:
            if self.separate_stains_method is None:
                raise ValueError
            else: 
                if self.separate_stains_method == 'macenko_pca':
                    try:
                        wc_pca = histomicstk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(np.asarray(img), self.bgr)
                        wc_pca = _reorder_stains(wc_pca, stains=['hematoxylin', 'eosin'] )
                        stains_pca_, stainsfloat_pca_, _ = histomicstk.preprocessing.color_deconvolution.color_deconvolution(np.asarray(img), wc_pca, self.bgr)
            
                        stains_pca = stains_pca_.copy()
                        stainsfloat_pca = stainsfloat_pca_.copy()
                        if self.delete_third_stain:
                            stains_pca[:,:,2] = int(self.bgr.mean())
                            stainsfloat_pca[:,:,2] = self.bgr.mean()
                            
                        if self.convert_to_rgb:
                            img = histomicstk.preprocessing.color_deconvolution.color_convolution(stainsfloat_pca, wc_pca, np.asarray(self.bgr))
                        else:
                            img = stains_pca
                            
                    except (IndexError, ValueError): 
                        img = np.array(img)
                        img[:,:,:] = int(self.bgr.mean())
                        
                elif self.separate_stains_method == 'xu_snmf': # with a initialization of a first xu_snmf stain separation (reordered) to keep the channel in right order
                    w_init = np.load('../color_normalization/44286/wc_snmf_06.npy')
                    img_sda = histomicstk.preprocessing.color_conversion.rgb_to_sda(np.asarray(img), self.bgr)
                    wc_snmf = histomicstk.preprocessing.color_deconvolution.separate_stains_xu_snmf(img_sda, w_init, 0.2)
                    stains_snmf_, stainsfloat_snmf_, _ = histomicstk.preprocessing.color_deconvolution.color_deconvolution(np.asarray(img), wc_snmf, np.asarray(self.bgr))
    
                    stains_snmf = stains_snmf_.copy()
                    stainsfloat_snmf = stainsfloat_snmf_.copy()
                    if self.delete_third_stain:
                        stains_snmf[:,:,2] = int(self.bgr.mean())
                        stainsfloat_snmf[:,:,2] = self.bgr.mean()
                        
                    if self.convert_to_rgb:
                        img = histomicstk.preprocessing.color_deconvolution.color_convolution(stainsfloat_snmf, wc_snmf, np.asarray(self.bgr))
                    else:
                        img = stains_snmf
                        
                elif self.separate_stains_method == 'fixed_hes_vector': # use a fixed HES stain vector from an image with obvious saffron
                    w_init = np.load('../color_normalization/44286/wc_snmf_06.npy')
                    stains_snmf_, stainsfloat_snmf_, _ = histomicstk.preprocessing.color_deconvolution.color_deconvolution(np.asarray(img), w_init, np.asarray(self.bgr))
                    
                    stains_snmf = stains_snmf_.copy()
                    stainsfloat_snmf = stainsfloat_snmf_.copy()
                    if self.delete_third_stain:
                        stains_snmf[:,:,2] = int(self.bgr.mean())
                        stainsfloat_snmf[:,:,2] = self.bgr.mean()
                        
                    if self.convert_to_rgb:
                        img = histomicstk.preprocessing.color_deconvolution.color_convolution(stainsfloat_snmf, w_init, np.asarray(self.bgr))
                    else:
                        img = stains_snmf
    
                else:
                    raise NotImplementedError
                    
                img = Image.fromarray(img)

        # reinhard
        if self.color_norm:
            if self.color_norm_method is not None:
                if self.color_norm_method == 'reinhard':
                    target_mean = np.load("../color_normalization/ref/merge_tiles_lab_mean.npy")
                    target_std = np.load("../color_normalization/ref/merge_tiles_lab_std.npy")
                    img = histomicstk.preprocessing.color_normalization.reinhard(np.asarray(img), target_mu=target_mean, target_sigma=target_std)
                    
                elif self.color_norm_method == 'macenko_pca':
                    wc_target = np.load("../color_normalization/ref/wc_pca.npy")
                    try:
                        wc_pca = histomicstk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(np.asarray(img), self.bgr)
                        wc_pca = _reorder_stains(wc_pca, stains=['hematoxylin', 'eosin'] )
                        stains_pca, stainsfloat_pca, _ = histomicstk.preprocessing.color_deconvolution.color_deconvolution(np.asarray(img), wc_pca, self.bgr)
                        
                        img = histomicstk.preprocessing.color_deconvolution.color_convolution(stainsfloat_pca, wc_target, self.bgr)

                    except (IndexError, ValueError): 
                        img = np.array(img)
                        img[:,:,:] = 242 # bgr of the slide which the reference images come from
                        
                else:
                    raise NotImplementedError
                img = Image.fromarray(img)
            else:
                raise ValueError

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        
        if self.save_images_to_h5:
            with h5py.File(os.path.join(self.image_h5_dir, os.path.basename(self.file_path)), "a", driver='mpio', comm=MPI.COMM_WORLD) as f_img:
                f_img['imgs'][idx] = np.asarray(img)
        
        try:
            img = self.roi_transforms(img).unsqueeze(0)
        except AttributeError:
            img = self.roi_transforms(img)
            
        return img, coord
        

class Dataset_All_Bags(Dataset):

    def __init__(self, data_source, csv_path):
        self.data_source = data_source
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        slide_id = self.df['slide_id'][idx]
        return os.path.join(self.data_source, slide_id + '.h5')





