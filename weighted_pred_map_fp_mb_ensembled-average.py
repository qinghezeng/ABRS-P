#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:38:12 2023

Ensembled average the normalized weighted prediction scores (percentile/softmax/...) and plot the heatmap

@author: Q Zeng
"""

import argparse
from PIL import Image
import torch
import os
import time
import numpy as np
import h5py
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import random
import string
from wsi_core.WholeSlideImage import DrawMap, StitchCoords, WholeSlideImage
from PIL import ImageDraw
import math
import pandas as pd
import openslide

parser = argparse.ArgumentParser(description='CLAM Ensembled-Aver Weighted Pred Map Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
					help='directory to save eval results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--k', type=int, default=10, help='number of total trained folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--downscale', type=int, default=64, help='downsample ratio to the splitting magnification (default: 64)')
parser.add_argument('--downsample', type=float, default=-1, help='downsample ratio to the patching magnification (default: -1)')
parser.add_argument('--snapshot', action='store_true', default=False, help='export snapshot')
parser.add_argument('--grayscale', action='store_true', default=False, help='export grayscale heatmap')
parser.add_argument('--colormap', action='store_true', default=False, help='export colored heatmap')
parser.add_argument('--blended', action='store_true', default=False, help='export blended image')
parser.add_argument('--patch_bags', type=str, nargs='+', default=None, 
                    help='names of patch files (ends with .h5) for visualization (default: None), overruled by tp')
parser.add_argument('--B', type=int, default=-1, help='save the B positive and B negative patches used for slide-level decision (default: -1, not saving)')
parser.add_argument('--tp', action='store_true', default=False, help='only process true positive slides, overrule patch_bags')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--custom_downsample', type=int, default=1, help='overruled by target_patch_size')
parser.add_argument('--target_patch_size', type=int, default=-1, help='overrule custom_downsample')
parser.add_argument('--cpu', default=False, action='store_true', help='force to use cpu') # if gpu not available, use cpu automatically
parser.add_argument('--auto_skip', default=False, action='store_true', help='auto skip checking if snapshot file exists')
parser.add_argument('--slide_ext', nargs="+", default= ['.svs', '.ndpi', '.tiff', '.tif'], help='slide extensions to be recognized, svs/ndpi/tiff by default')
# for visium data
parser.add_argument('--heatmap_crop_size', type=int, default=-1, help='size to central crop patches for heatmap')
parser.add_argument('--spot_coords_dir',  type = str, default=None,
                    help='directory of spot coordinate file (.csv)')
parser.add_argument('--norm', type=str, default='percentile-rescale01', 
                    help='how to normalize att scores for better heatmap')
parser.add_argument('--brs', type=int, nargs='+', default=None, 
                    help='genes / branches to plot')
parser.add_argument('--N_highlight', type=int, default=-1, help='only the top N and low N tiles highlighted  (default: -1, plot all)')
args = parser.parse_args()

args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end
    
folds = range(start, end)
    

if __name__ == "__main__":
    if not args.tp:
        if args.patch_bags is not None:
            patch_bags = args.patch_bags
        else:
            patch_bags = sorted(os.listdir(args.data_root_dir))
            patch_bags = [patches for patches in patch_bags if os.path.isfile(os.path.join(args.data_root_dir, patches))]
    
    save_dir = args.save_dir
    save_dir = os.path.join(save_dir, f"ensembled-aver_weighted_pred_maps_{args.k}f_{args.downscale}_{args.norm}")
    os.makedirs(save_dir, exist_ok=True)
    
    total = len(patch_bags)
    times = 0.
         
    for i in range(total): 
        print("\n\nprogress: {:.2f}, {}/{} in current model.".format(i/total, i, total))
        print('processing {}'.format(patch_bags[i]))
        
        if args.auto_skip and os.path.isfile(os.path.join(save_dir, patch_bags[i].replace(".h5", "_snapshot.png"))):
            print('{} already exist in destination location, skipped'.format(patch_bags[i].replace(".h5", "_snapshot.png")))
            continue
        
        time_elapsed = -1
        start_time = time.time()
         
        fpatch = os.path.join(args.data_root_dir, patch_bags[i])
        f = h5py.File(fpatch, mode='r')
   
        patch_level = f['coords'].attrs['patch_level']
        patch_size = f['coords'].attrs['patch_size']
        
        if patch_size == args.target_patch_size:
            target_patch_size = None
        elif args.target_patch_size > 0:
            target_patch_size = (args.target_patch_size, ) * 2
        elif args.custom_downsample > 1:
            target_patch_size = (patch_size // args.custom_downsample, ) * 2
        else:
            target_patch_size = None

        for key in f.keys():
            print(key) # Names of the groups in HDF5 file.

        slide_file_path = os.path.join(args.data_slide_dir, [sli for sli in os.listdir(args.data_slide_dir) if 
                                                                 sli.endswith(tuple(args.slide_ext)) and 
                                                                 (os.path.splitext(sli)[0] == os.path.splitext(os.path.basename(fpatch))[0])][0])
        
        with openslide.open_slide(slide_file_path) as wsi:
            if args.downsample > 0:
                downsample = args.downsample
            else:
                # openslide.PROPERTY_NAME_OBJECTIVE_POWER works for both ndpi and svs but not tiff generated by libvips
                if slide_file_path.endswith("ndpi") or slide_file_path.endswith(".svs"):
                    if (wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] == '20'): # 20x(no exception magnification)~= 0.5 or 1.0 (not correctly recognized)
                        downsample = 1.0
                    elif (wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] == '40'): # 40x pixelsize ~= 0.25
                        downsample = 2.0
                    else:
                        raise Exception("The highest magnification should be 20x or 40x.")

                elif slide_file_path.endswith(".tiff"):
                    # the properties in tiff slide generated by libvips can not be correctly decoded
                    loc1st = str(wsi.properties).find('openslide.objective-power&lt;/name&gt;\\n      &lt;value type="VipsRefString"&gt;')
                    if (str(wsi.properties)[loc1st+80:loc1st+82] == '20'):
                        downsample = 1.0
                    elif (str(wsi.properties)[loc1st+80:loc1st+82] == '40'):
                        downsample = 2.0
                    else:
                        raise Exception("The highest magnification should be 20x or 40x. Check your slide properties first.")
                        
                else:
                    raise Exception("Please indicate the downsample factor for your slide ext.")
                
            dset_attrs_downsampled_level_dim = np.asarray([np.int64(math.floor(wsi.dimensions[0]/downsample)), np.int64(math.floor(wsi.dimensions[1]/downsample))])
            dset_attrs_level_dim = np.asarray([np.int64(wsi.dimensions[0]), np.int64(wsi.dimensions[1])])
            
        dset_attrs_downsample = np.asarray([np.float64(1), np.float(1)])
        dset_attrs_patch_level = np.int64(0)
        dset_attrs_wsi_name = os.path.splitext(patch_bags[i])[0]
        
        if args.blended or args.snapshot:
            snapshot = StitchCoords(fpatch, WholeSlideImage(slide_file_path), downscale=args.downscale, 
                                    bg_color=(255,255,255), alpha=-1, draw_grid=False, downsampled_level_dim=dset_attrs_downsampled_level_dim)
        if args.snapshot:
            snapshot.save(os.path.join(save_dir, patch_bags[i].replace(".h5", "_snapshot.png")))
            
        if args.cpu:
            att = torch.load(os.path.join(args.save_dir, "weighted_pred_scores_"+str(folds[0]), "patch_pred_score_"+
                                          patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage)
        else:
            att = torch.load(os.path.join(args.save_dir, "weighted_pred_scores_"+str(folds[0]), "patch_pred_score_"+
                                          patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage.cuda(0))
        
        if args.brs is not None:
            brs = args.brs
        else:
            brs = range(att.size()[0])
            
        for br in brs:
            arr_att = np.zeros(att.size()[1]) # N patches
            del(att)
            for fold in folds:
                if args.cpu:
                    att = torch.load(os.path.join(args.save_dir, "weighted_pred_scores_"+str(fold), "patch_pred_score_"+
                                                  patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage)
                else:
                    att = torch.load(os.path.join(args.save_dir, "weighted_pred_scores_"+str(fold), "patch_pred_score_"+
                                                  patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage.cuda(0))
            
                list_att = att.data.tolist()[br]
                
                if args.norm == 'percentile-rescale01': # CLAM
                    percentile = []
                    for j in range(len(list_att)):
                        percentile.append(stats.percentileofscore(list_att, list_att[j])) # the rank in ascending order
                    del(list_att)
                    list_att = percentile
                    del(percentile)
                elif args.norm == 'softmax-rescale01': # Ilse18a
                    from scipy.special import softmax
                    list_att = softmax(list_att)  # softmax over N
                elif args.norm == 'submin1-log2-rescale01':
                    list_att = [np.log2(x - min(list_att) +1) for x in list_att]
                elif args.norm == 'rescale01':
                    list_att = [(x - min(list_att)) / (max(list_att) - min(list_att)) for x in list_att]
                else:
                    raise NotImplementedError
                arr_att += np.asarray(list_att)
                del(list_att)

            if args.spot_coords_dir is not None:
                df = pd.read_csv(os.path.join(args.spot_coords_dir, patch_bags[i].replace('h5', 'csv')))
            else:
                df = pd.DataFrame({'pxl_col_in_fullres': f['coords'][:,1], 'pxl_row_in_fullres': f['coords'][:,0]})

            assert df.shape[0] == arr_att.shape[0]
            df['weighted_pred_score'] = list(arr_att/args.k)
            os.makedirs(os.path.join(args.save_dir, f"ensembled-aver_weighted_pred_scores_{args.k}f_{args.norm}"), exist_ok=True)
            df.to_csv(os.path.join(args.save_dir, f"ensembled-aver_weighted_pred_scores_{args.k}f_{args.norm}", f"{os.path.splitext(patch_bags[i])[0]}_{br}.csv"),
                      index=False)
            nor = [(x - np.min(arr_att)) / (np.max(arr_att) - np.min(arr_att)) for x in arr_att]
                
            # for the B highest and B lowest, save the original patch named with the weighted pred score and coords
            if args.B > 0:
                patch_dir = os.path.join(args.save_dir, f'repres_patches_{args.k}f_br{br}')
                os.makedirs(patch_dir, exist_ok=True)
                inds = list(range(args.B))
                inds.extend(list(range(-args.B,0)))
                sort_index = np.argsort(-np.array(nor)) # descending
                for n in inds:
                    ind = sort_index[n]
                    coords = f["coords"][ind]
 			
                    with openslide.open_slide(slide_file_path) as wsi:
                        im = wsi.read_region(coords, patch_level, (patch_size, patch_size)).convert('RGB')
            
                    if target_patch_size is not None:
                        im = im.resize(target_patch_size) # (256, 256, 3)

                    im.save(os.path.join(patch_dir, patch_bags[i].replace(".h5", '_'+str(n)+'_'+str(nor[ind])+'_['+
                                          str(f["coords"][ind][0])+','+str(f["coords"][ind][1])+'].tif'))) # coords on level 0
                    
            
            cmap = matplotlib.cm.get_cmap('RdBu_r')
            heatmap = (cmap(nor)*255)[:,:3] # (N, 3)

            letters = string.ascii_lowercase
            result_str = "".join(random.choice(letters) for n in range(5))
            filename = "garbage_"+result_str+".h5"
            file = h5py.File(filename, mode='w')

            dset = file.create_dataset("imgs", shape = (0, 256, 256, 3),
                                        maxshape=(None, 256, 256, 3), chunks=(1, 256, 256, 3), dtype=heatmap.dtype)

            a = np.ones((256,256,3)) # to further enlarge 1 pixel to 256*256 patch
            for h in range(len(heatmap)):
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[h] = (a * heatmap[h])[np.newaxis, np.newaxis,:] # dset is the heatmap, in the format for stitching
            del(heatmap, a, cmap)
            
            dset.attrs['downsample'] = dset_attrs_downsample
            dset.attrs['downsampled_level_dim'] = dset_attrs_downsampled_level_dim
            dset.attrs['level_dim'] = dset_attrs_level_dim
            dset.attrs['wsi_name'] = dset_attrs_wsi_name
            
            downscale=args.downscale
            draw_grid=False
            bg_color=(255,255,255)
            alpha=-1
    
            w, h = dset.attrs['downsampled_level_dim']
            print('original size of 20x: {} x {}'.format(w, h)) # the actual patching level size (could be not existed level)
    
            w = w // downscale
            h = h //downscale # downsample 64 of the patching level
            print('downscaled size for stiching: {} x {}'.format(w, h))
            
            print('number of patches: {}'.format(len(f["coords"])))
            
            img_shape = (256, 256, 3)
            
            print('patch shape: {}'.format(img_shape))
    
            downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)
            
            if w*h > Image.MAX_IMAGE_PIXELS: 
                raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
            
            if alpha < 0 or alpha == -1:
                heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
            else:
                heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
            
            heatmap = np.array(heatmap)
            
            if args.heatmap_crop_size <= 0:
                heatmap_crop_size = downscaled_shape
            else:
                heatmap_crop_size = (args.heatmap_crop_size // downscale, args.heatmap_crop_size // downscale) # 24 start to overlap on rows for downscale=8
            # highlight the top N and low N tiles ***
            if args.N_highlight > 0:
                inds = list(range(args.N_highlight))
                inds.extend(list(range(-args.N_highlight,0)))
                indices = np.argsort(-np.array(nor))[inds] # descending
            else:
                indices = None

            heatmap = DrawMap(heatmap, dset, (f["coords"][:]/(dset.attrs['level_dim'][0]//w)).astype(np.int32), 
                              heatmap_crop_size, indices=indices, draw_grid=draw_grid)
            
            print(patch_bags[i])
            
            if args.colormap:
                if args.N_highlight > 0:
                    heatmap.save(os.path.join(save_dir, patch_bags[i].replace(".h5", f"_heatmap_br{br}_{args.N_highlight}p.png")))
                else:
                    heatmap.save(os.path.join(save_dir, patch_bags[i].replace(".h5", f"_heatmap_br{br}.png")))
                    
            if args.grayscale:
                grayscale = Image.new('L', (w, h))
                draw = ImageDraw.Draw(grayscale)
                
                for n in range(len(nor)):
                    xcoord = (f["coords"][n][0] / (dset.attrs['level_dim'][0]//w)).astype(np.int32)
                    ycoord = (f["coords"][n][1] / (dset.attrs['level_dim'][0]//w)).astype(np.int32)
                    draw.rectangle([(xcoord, ycoord), (xcoord+downscaled_shape[0], ycoord+downscaled_shape[1])], fill= math.floor(nor[n] * 255))
                grayscale.save(os.path.join(save_dir, patch_bags[i].replace(".h5", f"_grayscale_br{br}.png")))
            
            file.close()
            # Delete this temporary file
            os.remove(filename)
            
            # Save blended image
            if args.blended:
               if args.N_highlight > 0:
                    Image.blend(snapshot, heatmap.convert("RGB"), 0.3).save(os.path.join(save_dir, patch_bags[i].replace(".h5", f"_blended_br{br}_{args.N_highlight}p.png")))
               else:
                    Image.blend(snapshot, heatmap.convert("RGB"), 0.3).save(os.path.join(save_dir, patch_bags[i].replace(".h5", f"_blended_br{br}.png")))
                
            time_elapsed = time.time() - start_time
            times += time_elapsed
            
        f.close()
        times /= total
        print("average time in s per slide: {}".format(times))
