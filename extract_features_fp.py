import torch
import torch.nn as nn
import os
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import collate_features
import h5py
import openslide
import torchvision
from models.RetCCL.ccl import CCL
import models.RetCCL.ResNet as ResNet
import models.TransPath.moco.builder_infence
import models.TransPath.vits
from functools import partial
from models.TransPath.byol_pytorch.byol_pytorch_get_feature import BYOL
from models.TransPath.ctran import ctranspath
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_hdf5(output_path, asset_dict, mode='a'):
    file = h5py.File(output_path, mode)

    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val  

    file.close()
    return output_path


def compute_w_loader(file_path, output_path, wsi, model,
     batch_size = 8, verbose = 0, print_every=20, pretrained=True,
     custom_downsample=1, target_patch_size=-1,
     unmixing = False, separate_stains_method = None, file_bgr = None, bgr = None, delete_third_stain = False, 
     convert_to_rgb = False,
     color_norm = False, color_norm_method = None,
     save_images_to_h5 = False, image_h5_dir = None,
     custom_transforms = None,
     model_name = None):
     
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet. Actually here means whether to use ImageNet transform or not
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding (overruled by target_patch_size)
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
        custom_downsample=custom_downsample, target_patch_size=target_patch_size,
        unmixing = unmixing, separate_stains_method = separate_stains_method, file_bgr = file_bgr, bgr = bgr, 
        delete_third_stain = delete_third_stain, convert_to_rgb = convert_to_rgb,
        color_norm = color_norm, color_norm_method = color_norm_method,
        save_images_to_h5 = save_images_to_h5, image_h5_dir = image_h5_dir,
        custom_transforms = custom_transforms)
    x, y = dataset[0]
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():    
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)
            mini_bs = coords.shape[0]
            
            if model_name == 'transpath-tcga-paip-byol':
                _, features = model(batch, return_embedding = True)
            else:
                features = model(batch)
            
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, mode=mode)
            mode = 'a'
    
    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str, help='path to h5 files')
parser.add_argument('--data_slide_dir', type=str, default=None, help='path to slides')
parser.add_argument('--slide_ext', nargs="+", default= ['.svs', '.ndpi', '.tiff', 'tif', 'mrxs', 'qptiff'], help='slide extensions to be recognized, svs/ndpi/tiff by default')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1, help='overruled by target_patch_size')
parser.add_argument('--target_patch_size', type=int, default=-1, help='overrule custom_downsample')
parser.add_argument('--model', type=str, default="resnet50", help='feautre extractor')
parser.add_argument('--weights_path', type=str, default=None)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--unmixing', default=False, action='store_true', help='apply color unmixing')
parser.add_argument('--separate_stains_method', default=None, help='method to separate stains')
parser.add_argument('--file_bgr', type=str, default=None, help='overruled by bgr')
parser.add_argument('--bgr', type=str, default=None, help='overrule file_bgr')
parser.add_argument('--delete_third_stain', default=False, action='store_true', help='replace the third stain with slide background')
parser.add_argument('--convert_to_rgb', default=False, action='store_true', help='use rgb instead of stain image')
parser.add_argument('--color_norm', default=False, action='store_true', help='apply color normalization')
parser.add_argument('--color_norm_method', default=None, help='method for color normalization')
parser.add_argument('--save_images_to_h5', default=False, action='store_true', help='save patch image to h5 in a new path')
parser.add_argument('--image_h5_dir', default=None, help='path to save new h5 with patch images added')
parser.add_argument('--visium', default=False, action='store_true', help='for visium data')
parser.add_argument('--disable_pretrained', default=True, action='store_false', help='use PyTorch stats instead of ImageNet')
parser.add_argument('--custom_transforms', type=str, default=None, help='name of pre-defined transform other than ImageNet')
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(args.data_dir, csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    dest_files = os.listdir(args.feat_dir)

    print('loading model checkpoint')

    if args.model == "resnet50":
        model = resnet50_baseline(pretrained=True) # default use by clam
    elif args.model == "shufflenet":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=False)
        model.fc = nn.Linear(1024, args.n_classes)
        checkpoint = torch.load(args.weights_path)
        model.load_state_dict(checkpoint)
        
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
                
            def forward(self, x):
                return x
        model.fc = Identity()

    elif args.model == "resnet18-simclr-histo":
        model = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load('models/simclr/pytorchnative_tenpercent_resnet18.ckpt', map_location='cpu')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model.fc = torch.nn.Sequential()

    elif args.model == "retccl":
        backbone = ResNet.resnet50
        model = CCL(backbone, 128, 65536, mlp=True, two_branch=True, normlinear=True).cuda()

        pretext_model = torch.load('./models/RetCCL/best_ckpt.pth', map_location='cpu')
        model.load_state_dict(pretext_model, strict=True)
        model.encoder_q.fc = nn.Identity()
        model.encoder_q.instDis = nn.Identity()
        model.encoder_q.groupDis = nn.Identity()

    elif args.model == "vit-small-tcga-paip-mocov3":
        model = models.TransPath.moco.builder_infence.MoCo_ViT(
            partial(models.TransPath.vits.__dict__['vit_small'], stop_grad_conv1=True))
    
        pretext_model = torch.load('./models/TransPath/vit_small.pth.tar', map_location='cpu')
        from collections import OrderedDict as OD
        model.load_state_dict(OD([(key.split("module.")[-1], pretext_model['state_dict'][key]) for key in pretext_model['state_dict']]), strict=True)
    
    elif args.model == "vit-conv-small-tcga-paip-mocov3":
        model = models.TransPath.moco.builder_infence.MoCo_ViT(
            partial(models.TransPath.vits.__dict__['vit_conv_small'], stop_grad_conv1=True))
    
        pretext_model = torch.load('./models/TransPath/vit_small_conv.pth.tar', map_location='cpu')
        from collections import OrderedDict as OD
        model.load_state_dict(OD([(key.split("module.")[-1], pretext_model['state_dict'][key]) for key in pretext_model['state_dict']]), strict=True)

    elif args.model == "transpath-tcga-paip-byol":
        model = BYOL(image_size=256, hidden_layer='to_latent')
        pretext_model = torch.load('./models/TransPath/TransPath.pth', map_location='cpu')
        from collections import OrderedDict as OD
        model.load_state_dict(OD([(key.split("module.")[-1], pretext_model[key]) for key in pretext_model]), strict=True)
        
        model.online_encoder.net.head = nn.Identity()

    elif args.model == "ctranspath-tcga-paip":
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load('./models/TransPath/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)
            
    else:
        raise NotImplementedError

    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        bag_candidate = bags_dataset[bag_candidate_idx]    # full path + full name (ext/.h5 included)
        bag_name = os.path.basename(os.path.normpath(bag_candidate)) # full name (ext/.h5 included)
        h5_file_path = bag_candidate
        
        slide_file_path = os.path.join(args.data_slide_dir, [sli for sli in os.listdir(args.data_slide_dir) if sli.endswith(tuple(args.slide_ext)) and 
                           (os.path.splitext(sli)[0] == os.path.splitext(bag_name)[0])][0])
                                                                     
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(bag_name)

        if not args.no_auto_skip and bag_name in dest_files:    
            print('skipped {}'.format(bag_name))
            continue 

        output_path = os.path.join(args.feat_dir, bag_name)
        time_start = time.time()
        with openslide.open_slide(slide_file_path) as wsi:
            output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
                                                model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
                                                custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
                                                unmixing = args.unmixing, separate_stains_method = args.separate_stains_method,
                                                file_bgr = args.file_bgr, bgr = args.bgr, delete_third_stain = args.delete_third_stain, 
                                                convert_to_rgb = args.convert_to_rgb,
                                                color_norm = args.color_norm, color_norm_method = args.color_norm_method,
                                                save_images_to_h5 = args.save_images_to_h5, image_h5_dir = args.image_h5_dir,
                                                pretrained = args.disable_pretrained, 
                                                custom_transforms = args.custom_transforms,
                                                model_name = args.model)

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        
        if args.visium:
            with h5py.File(h5_file_path, "r") as f_patch:
                barcodes = f_patch['barcodes'][:]
                with h5py.File(output_file_path, "a") as file:
                    file.create_dataset('barcodes', data=barcodes)
        
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)

        if args.visium:
            print('barcodes size: ', file['barcodes'].shape)

        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)

        try: # torch.__version__='1.7.0'
            torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'),_use_new_zipfile_serialization=False)
        except TypeError: # torch version '1.3.1'
            torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'))


