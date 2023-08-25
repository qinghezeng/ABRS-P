#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:59:28 2022

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
import random
import string
from wsi_core.WholeSlideImage import DrawMap, StitchCoords, WholeSlideImage
from PIL import ImageDraw
import math
import pandas as pd
import openslide

parser = argparse.ArgumentParser(description='CLAM Patch Pred Map Script for mb')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
					help='directory to save eval results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--k', type=int, default=10, help='number of total trained folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
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
parser.add_argument('--norm', type=str, default='percentile-rescale01', 
                    help='how to normalize pred scores for better heatmap')
parser.add_argument('--cm', type=str, default='RdBu_r', help='colormap to apply')
parser.add_argument('--brs', type=int, nargs='+', default=None, 
                    help='genes / branches to plot')
parser.add_argument('--score_type', type=str, default='patch_pred', 
                    help='name of the prediction to use')
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
    
if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
    

if __name__ == "__main__":
        
    if not args.tp:
        if args.patch_bags is not None:
            patch_bags = args.patch_bags
        else:
            patch_bags = sorted(os.listdir(args.data_root_dir))
            patch_bags = [patches for patches in patch_bags if os.path.isfile(os.path.join(args.data_root_dir, patches))]
     
    for fold in folds:
        if args.tp:
            patch_bags = []
            
            tp_file = "fold_"+str(fold)+"_optimal.csv"
            print('Load the file indicating the true prositive slides: {}.'.format(tp_file))
            tp_file = os.path.join(args.save_dir, tp_file)
            
            tp_df = pd.read_csv(tp_file)
    
            for i in range(tp_df.shape[0]):
                if (tp_df.iloc[i, 1] == 1.0) and tp_df.iloc[i, 7]: #TP
                    patch_bags.append(tp_df.iloc[i, 0]+".h5") 
        
        save_dir = args.save_dir
        save_dir = os.path.join(save_dir, f"{args.score_type}_maps_{fold}_{args.downscale}_{args.norm}_{args.cm}")
        os.makedirs(save_dir, exist_ok=True)
        
        total = len(patch_bags)
        times = 0.
         
        for i in range(total): 
            print("\n\nprogress: {:.2f}, {}/{} in current model. {} out of {} models".format(i/total, i, total, folds.index(fold), len(folds)))
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
                pred = torch.load(os.path.join(args.save_dir, f"{args.score_type}_scores_{fold}", f"{args.score_type}_score_"+ # use patch_pred_score_ if to plot patch prediction heatmap
                                              patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage)
            else:
                pred = torch.load(os.path.join(args.save_dir, f"{args.score_type}_scores_{fold}", f"{args.score_type}_score_"+ # use patch_pred_score_ if to plot patch prediction heatmap
                                              patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage.cuda(0))

            # print(pred.shape)
            
            if args.brs is not None:
                brs = args.brs
            else:
                brs = range(pred.size()[0])
                
            for br in brs:
                list_pred = pred.data.tolist()[br]
                
                if args.norm == 'percentile-rescale01': # CLAM
                    percentile = []
                    for j in range(len(list_pred)):
                        percentile.append(stats.percentileofscore(list_pred, list_pred[j])) # the rank in ascending order
                    print(len(percentile))
                    
                    nor = [(x - min(percentile)) / (max(percentile) - min(percentile)) for x in percentile] # scale to [0, 1], 1 is the most attended
                    del(percentile)
                elif args.norm == 'softmax-rescale01': # Ilse18a
                    arr_pred = np.array(list_pred)
                    from scipy.special import softmax
                    arr_pred[~np.isnan(list_pred)] = softmax(arr_pred[~np.isnan(list_pred)])
                    list_pred = list(arr_pred)
                    del(arr_pred)
                    nor = [(x - min(list_pred)) / (max(list_pred) - min(list_pred)) for x in list_pred]
                elif args.norm == 'submin1-log2-rescale01':
                    list_pred = [np.log2(x - min(list_pred) +1) for x in list_pred]
                    nor = [(x - min(list_pred)) / (max(list_pred) - min(list_pred)) for x in list_pred]
                elif args.norm == 'rescale01':
                    nor = [(x - min(list_pred)) / (max(list_pred) - min(list_pred)) for x in list_pred]
                else:
                    raise NotImplementedError
                del(list_pred)
                
                # for the B highest and B lowest, save the original patch named with the patch pred score and coords
                if args.B > 0:
                    patch_dir = os.path.join(args.save_dir, f'repres_patches_{args.score_type}_f{fold}_br{br}')
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
                        
                
                if args.cm == "sameR":
                    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0,"#0000FF"), (0.5, "#FFFFFF"), (1, "#FF0000")])
                else:
                    cmap = matplotlib.cm.get_cmap(args.cm)
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
                    heatmap_crop_size = (args.heatmap_crop_size // downscale, args.heatmap_crop_size // downscale)

                heatmap = DrawMap(heatmap, dset, (f["coords"][:]/(dset.attrs['level_dim'][0]//w)).astype(np.int32), 
                                  heatmap_crop_size, indices=None, draw_grid=draw_grid)
                
                print(patch_bags[i])

                if args.colormap:
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
                    Image.blend(snapshot, heatmap.convert("RGB"), 0.3).save(os.path.join(save_dir, patch_bags[i].replace(".h5", f"_blended_br{br}.png")))

            
            time_elapsed = time.time() - start_time
            times += time_elapsed
            f.close()
            
        times /= total
        print("average time in s per slide: {}".format(times))
