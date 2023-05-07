import math
import os
import time
from xml.dom import minidom

import cv2
import numpy as np
import openslide
from PIL import Image
import h5py

from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, save_hdf5

import pandas as pd

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True):
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        print('start stitching {}'.format(patch_dset.attrs['wsi_name']))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)
        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1):
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['imgs']
    coords = file['coords'][:]
    if 'downsampled_level_dim' in dset.attrs.keys():
        w, h = dset.attrs['downsampled_level_dim']
    else:
        w, h = dset.attrs['level_dim']
    print('original size: {} x {}'.format(w, h)) # the patching level size
    w = w // downscale
    h = h //downscale
    coords = (coords / downscale).astype(np.int32)
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(dset)))
    img_shape = dset[0].shape
    print('patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)

    if Image.MAX_IMAGE_PIXELS:
        if w*h > Image.MAX_IMAGE_PIXELS: 
            raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)
    
    file.close()
    return heatmap

def DrawMapFromCoords(canvas, wsi_object, coords, patch_size, vis_level, indices=None, verbose=1, draw_grid=True, downscale=None):
    if wsi_object.path.endswith('.qptiff'):
        downsamples = wsi_object.level_downsamples[vis_level]
    else:
        downsamples = wsi_object.wsi.level_downsamples[vis_level]

    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        
    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        coord = coords[patch_id]
        
        if wsi_object.path.endswith('.qptiff'):
            patch = wsi_object.wsi[vis_level][coord[1]//wsi_object.level_downsamples[vis_level][0]:coord[1]//wsi_object.level_downsamples[vis_level][0]+patch_size[0], coord[0]//wsi_object.level_downsamples[vis_level][0]:coord[0]//wsi_object.level_downsamples[vis_level][0]+patch_size[0]]
        else:
            patch = np.array(wsi_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord // downsamples, patch_size)
            
    img = Image.fromarray(canvas)

    if downscale is not None:
        ds = downscale//int(downsamples)
        img = img.resize((img.size[0]//ds, img.size[1]//ds))
    return img

def StitchCoords(hdf5_file_path, wsi_object, downscale=16, draw_grid=False, bg_color=(0,0,0), alpha=-1, downsampled_level_dim=None):
    wsi = wsi_object.getOpenSlide()
    
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['coords']
    coords = dset[:]

    if wsi_object.path.endswith('.qptiff'):
        w, h = wsi_object.level_dim[0]
    else:
        w, h = wsi.level_dimensions[0]

    print('start stitching {}'.format(dset.attrs['name']))
    print('original size of level 0: {} x {}'.format(w, h))

    if downsampled_level_dim is not None:
        w0 = w
        w, h = downsampled_level_dim // downscale # downscale from splitting magnification (20x)
        downscale = w0//w # the real downsampling rate from level 0
        vis_level = wsi.get_best_level_for_downsample(downscale) # from level 0

        ds = downscale//int(wsi_object.wsi.level_downsamples[vis_level])
    else:
        if wsi_object.path.endswith('.qptiff'):
            vis_level = 4
            h, w = wsi_object.level_dim[vis_level]
        else:
            vis_level = wsi.get_best_level_for_downsample(downscale)
            w, h = wsi.level_dimensions[vis_level]
        ds = 1
        
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(coords)))
    
    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))

    if wsi_object.path.endswith('.qptiff'):
        patch_size = tuple((np.array((patch_size, patch_size)) * wsi_object.level_downsamples[patch_level]).astype(np.int32))
    else:
        patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    print('ref patch size: {}x{}'.format(patch_size, patch_size))

    if Image.MAX_IMAGE_PIXELS: 
        if w*h > Image.MAX_IMAGE_PIXELS: 
            raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w*ds,h*ds), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w*ds,h*ds), mode="RGBA", color=bg_color + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)

    if downsampled_level_dim is not None:
        heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid, downscale=downscale)
    else:
        heatmap = DrawMapFromCoords(heatmap, wsi_object, coords, patch_size, vis_level, indices=None, draw_grid=draw_grid)
    
    file.close()
    return heatmap

class WholeSlideImage(object):
    def __init__(self, path, hdf5_file=None):
        self.name = ".".join(path.split("/")[-1].split('.')[:-1]) # extract slide name from the full path
        # credit to https://github.com/pcicales/CLAM/tree/79f55bc42e7d0f955316c642265a6c5129465ad9
        if path.endswith('.qptiff'):
            import tifffile
            wsi = tifffile.imread(path)
            # generate 7 levels (with scales 1, 2, 4, 8, 16, 32, 64)
            self.level_downsamples = [(1*(2 ** count), 1*(2 ** count)) for count in range(7)]
            self.level_dim = tuple([(int(wsi.shape[0]//downsamp[0]), int(wsi.shape[1]//downsamp[1])) for downsamp in self.level_downsamples])
            self.wsi = [cv2.resize(wsi, dsize=(level_dim[1], level_dim[0])) for level_dim in self.level_dim]
        else:
            self.wsi = openslide.open_slide(path)
            self.level_downsamples = self._assertLevelDownsamples() # call inner function to do complex init
            self.level_dim = self.wsi.level_dimensions
    
        self.contours_tissue = None
        self.contours_tumor = None
        self.contours_tumor_holes = None
        self.seg_level = None
        self.hdf5_file = hdf5_file
        self.path = path

    def getOpenSlide(self):
        return self.wsi

    def initXML(self, xml_path):
        def _createContour(coord_list):
            return np.array([[[int(float(coord.attributes['X'].value)), 
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype = 'int32')

        xmldoc = minidom.parse(xml_path)
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
        self.contours_tumor  = [_createContour(coord_list) for coord_list in annotations]
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)
     
    def importTumorAnnotationsTXT(self, txt_path):
        def _createContour(coord_list):
            # offset used in QuPath for MRXS format
            if os.path.splitext(self.path)[1] == '.mrxs':
                return np.array([[[int(float(coord.X)+float(self.wsi.properties['openslide.bounds-x'])), 
                               int(float(coord.Y)+float(self.wsi.properties['openslide.bounds-y']))]] for coord in coord_list], dtype = 'int32')
            else:
                return np.array([[[int(float(coord.X)), 
                               int(float(coord.Y))]] for coord in coord_list], dtype = 'int32')

        class Object(object):
            pass
        
        annotations = []
        with open(txt_path, 'r') as f:
            for line in f:
                coord_pairs = []
                for coord_xy in line[1:-1].split(','): # remove '[' at the begining and '\n' at the end
                    if 'Point: ' in coord_xy:
                        obj = Object()
                        obj.X = coord_xy.split('Point: ')[1]
                    else:
                        obj.Y = coord_xy[1:].split(']')[0]
                        coord_pairs.append(obj)
                annotations.append(coord_pairs)
        self.contours_tumor  = [_createContour(coord_list) for coord_list in annotations]
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def importTumorAnnotationsNPY(self, npy_path_tumor, npy_path_tumor_holes):
        self.contours_tumor = list(np.load(npy_path_tumor, allow_pickle=True))
        
        self.contours_tumor_holes = [list(holes_in_atumor) for holes_in_atumor in np.load(npy_path_tumor_holes, allow_pickle=True)]

    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
                            filter_params={'a_t':100}, ref_patch_size=512, annotations=False, annotation_dir=None, slide_name=None,
                            annotation_type=None,
                            exclude_ids=[], keep_ids=[]):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        
        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)

            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                a = cv2.contourArea(cont)
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)

            all_holes = []
            for parent in filtered:
                all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            
            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours
        
        if self.path.endswith('.qptiff'):
            img = self.wsi[seg_level]
        else:
            img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
        
       
        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

        scale = self.level_downsamples[seg_level] # downsample factor
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area # how many times of a patch surface
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area # how many times of a patch surface

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:,2:]
        if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
                    
        if annotations:
            if annotation_type is not None:
                if annotation_type == 'xml':
                    self.initXML(os.path.join(annotation_dir, os.path.splitext(slide_name)[0]+'.xml'))
                elif annotation_type == 'txt':
                    self.importTumorAnnotationsTXT(os.path.join(annotation_dir, slide_name+'.txt')) # self.contours_tumor
                elif annotation_type == 'npy':
                    self.importTumorAnnotationsNPY(os.path.join(annotation_dir, os.path.splitext(slide_name)[0]+'_contours_tumor.npy'), 
                                                   os.path.join(annotation_dir, os.path.splitext(slide_name)[0]+'_holes_tumor.npy'))
                else:
                    raise NotImplementedError
            else:
                raise ValueError        

        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)
        self.seg_level = seg_level
        
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)
            
        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), annot_hole_color = (102,0,204),
                    line_thickness=12, max_size=None, crop_window=None):
        if self.path.endswith('.qptiff'):
            img = self.wsi[vis_level]
        else:
            img = np.array(self.wsi.read_region((0,0), vis_level, self.level_dim[vis_level]).convert("RGB"))
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level, 1 / downsample factor
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
        if self.contours_tissue is not None:
            cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), # scale to vis magnification
                             -1, color, line_thickness, lineType=cv2.LINE_8)

            for holes in self.holes_tissue:
                cv2.drawContours(img, self.scaleContourDim(holes, scale), 
                                 -1, hole_color, line_thickness, lineType=cv2.LINE_8)
        
        if self.contours_tumor is not None:
            cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), 
                             -1, annot_color, line_thickness, lineType=cv2.LINE_8)
            if self.contours_tumor_holes is not None:
                for tumor_holes in self.contours_tumor_holes:
                    cv2.drawContours(img, self.scaleContourDim(tumor_holes, scale), 
                                     -1, annot_hole_color, line_thickness, lineType=cv2.LINE_8)
        
        img = Image.fromarray(img)
        if crop_window is not None:
            top, left, bot, right = crop_window
            left = int(left * scale[0])
            right = int(right * scale[0])
            top =  int(top * scale[1])
            bot = int(bot * scale[1])
            crop_window = (top, left, bot, right)
            img = img.crop(crop_window)
        w, h = img.size
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img

    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, annotations=False, **kwargs):
        contours = self.contours_tissue # highest magnification

        print("Creating patches for: ", self.name, "...",)
        elapsed = time.time()
        
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, annotations, **kwargs)
            
            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)

                # empty contour, continue
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch) # patch coords and img associated with the actual magnification is saved

        return self.hdf5_file

    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, annotations=False, custom_downsample=1,
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))
        
        if custom_downsample > 1:
            target_patch_size = patch_size
            patch_size = int(target_patch_size * custom_downsample)
            step_size = int(step_size * custom_downsample)
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size, 
                target_patch_size, target_patch_size))

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1]) # at the highest magnification
        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        
        if contour_fn == 'four_pt':
            cont_check_fn = self.isInContourV3_Easy
        elif contour_fn == 'four_pt_hard':
            cont_check_fn = self.isInContourV3_Hard
        elif contour_fn == 'center': # checks if the center of the patch is inside the contour
            cont_check_fn = self.isInContourV2
        elif contour_fn == 'basic': # checks if the top-left corner of the patch is inside the contour
            cont_check_fn = self.isInContourV1
        else:
            raise NotImplementedError

        img_w, img_h = self.level_dim[0]
        if use_padding: # whether to pad the border of the slide (default: True)
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]) # no exceed allowed
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x): # pt = (x, y) which is the top left cornor of patch at the highest magnification

                if not self.isInContours(cont_check_fn, cont, (x,y), self.contours_tumor, self.contours_tumor_holes, self.holes_tissue[cont_idx], ref_patch_size[0],
                                         0.5, annotations): #point not inside contour and its associated holes
                    continue # if this patch is not in tissue (or in holes), turn to next patch
                
                count+=1
                if self.path.endswith('.qptiff'):
                    patch_PIL = self.wsi[patch_level][y//self.level_downsamples[patch_level][0]:y//self.level_downsamples[patch_level][0]+patch_size, x//self.level_downsamples[patch_level][0]:x//self.level_downsamples[patch_level][0]+patch_size]
                else:
                    patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))
                
                if white_black:
                    if self.isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or self.isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
                        continue

                patch_info = {'x':int(x / (patch_downsample[0] * custom_downsample)), 'y':int(y / (patch_downsample[1] * custom_downsample)), 'cont_idx':cont_idx, 'patch_level':patch_level, 
                'downsample': self.level_downsamples[patch_level], 'downsampled_level_dim': tuple((np.array(self.level_dim[patch_level])/custom_downsample).astype('int')), 'level_dim': self.level_dim[patch_level],
                'patch_PIL':patch_PIL, 'name':self.name, 'save_path':save_path}  # the printed x and patch_PIL is at the actual magnification

                yield patch_info

        
        print("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0

    @staticmethod
    def isInContourV1(cont, pt, patch_size=None, center_shift=None):
        return 1 if cv2.pointPolygonTest(cont, pt, False) >= 0 else 0

    @staticmethod
    def isInContourV2(cont, pt, patch_size=256, center_shift=None):
        return 1 if cv2.pointPolygonTest(cont, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) >= 0 else 0
    
    # Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
    @staticmethod
    def isInContourV3_Easy(cont, pt, patch_size=256, center_shift=0.5):
        center = (pt[0]+patch_size//2, pt[1]+patch_size//2)
        assert 0 < center_shift < 1, "center-shift equal to "+str(center_shift)
        shift = int(patch_size//2*center_shift)
        all_points = [(center[0]-shift, center[1]-shift),
                      (center[0]+shift, center[1]+shift),
                      (center[0]+shift, center[1]-shift),
                      (center[0]-shift, center[1]+shift)
                      ]
    		
        for points in all_points:
            if cv2.pointPolygonTest(cont, points, False) >= 0:
                return 1
        return 0

    # Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
    @staticmethod
    def isInContourV3_Hard(cont, pt, patch_size=256, center_shift=0.5):
        center = (pt[0]+patch_size//2, pt[1]+patch_size//2)
        assert 0 < center_shift < 1, "center-shift equal to "+str(center_shift)
        shift = int(patch_size//2*center_shift)
        all_points = [(center[0]-shift, center[1]-shift),
                      (center[0]+shift, center[1]+shift),
                      (center[0]+shift, center[1]-shift),
                      (center[0]-shift, center[1]+shift)
                      ]
    		
        for points in all_points:
            if cv2.pointPolygonTest(cont, points, False) < 0:
                return 0
        return 1
    
    # add tumoral annotation and tumoral holes
    @staticmethod
    def isInContours(cont_check_fn, contour, pt, contours_tumor, contours_tumor_holes=None, holes=None, patch_size=256, center_shift=0.5, annotations=False):
        if cont_check_fn(contour, pt, patch_size, center_shift): # check if in tissue
            if annotations: # check if use anno
                for ntumor in range(len(contours_tumor)):
					# check if in anno
                    if cv2.pointPolygonTest(contours_tumor[ntumor], pt, False)>0: # negative when point is outside the contour, positive when point is inside
                    # in this anno
                        if contours_tumor_holes is not None: # check if in anno holes  
                            if WholeSlideImage.isInHoles(contours_tumor_holes[ntumor], pt, patch_size):
                            # in tumor hole
                                return 0
                            else:
                            # not in tumor hole
                                break
                        else:
                        # no tumor hole in this anno 
                            break # in this annotation, the last anno included
                    elif ntumor == len(contours_tumor)-1: # is the last contour_tumor
                        return 0
					
			# check if in holes
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size) # return 1 if inside hole, not 1 so is 0
            else:
                return 1
        return 0
    
    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]
    
    @staticmethod
    def isWhitePatch(patch, satThresh=5):
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

    @staticmethod
    def isBlackPatch(patch, rgbThresh=40):
        return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        
        return level_downsamples

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, annotations=False, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...",)
        elapsed = time.time()
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))
            
            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path, patch_size, step_size, 
                                                          annotations=annotations, **kwargs)
            if len(asset_dict) > 0: # if exist a extracted patch
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

        return self.hdf5_file


    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 256, step_size = 256, annotations=False, 
        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None, custom_downsample=1, 
        white_black=True, white_thresh=15, black_thresh=50):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        if custom_downsample > 1:
            target_patch_size = patch_size
            patch_size = int(target_patch_size * custom_downsample)
            step_size = int(step_size * custom_downsample)
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size, 
                target_patch_size, target_patch_size))
            
        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)
        
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)
    
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = self.isInContourV3_Easy
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = self.isInContourV3_Hard
            elif contour_fn == 'center': # checks if the center of the patch is inside the contour
                cont_check_fn = self.isInContourV2
            elif contour_fn == 'basic': # checks if the top-left corner of the patch is inside the contour
                cont_check_fn = self.isInContourV1
            else:
                raise NotImplementedError
        else:
            raise ValueError

        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
                       
        results = []
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x): # pt = (x, y) which is the top left cornor of patch at the highest magnification
                results.append(self.process_coord_candidate((x, y), cont, contour_holes, custom_downsample, patch_level, patch_size, 
                                                            ref_patch_size[0], cont_check_fn, white_black, white_thresh, black_thresh, 
                                                            annotations))

        results = np.array([result for result in results if result is not None])
        
        print('Extracted {} coordinates'.format(len(results)))

        if len(results)>1:
            asset_dict = {'coords' :          results}
            
            attr = {'patch_size' :            patch_size,
                    'patch_level' :           patch_level,
                    'downsample':             self.level_downsamples[patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                    'level_dim':              self.level_dim[patch_level],
                    'name':                   self.name,
                    'save_path':              save_path}

            attr_dict = { 'coords' : attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    def process_coord_candidate(self, coord, cont, contour_holes, custom_downsample, patch_level, patch_size, 
                                ref_patch_size, cont_check_fn, white_black, white_thresh, black_thresh, annotations):
        if WholeSlideImage.isInContours(cont_check_fn, cont, coord, self.contours_tumor, self.contours_tumor_holes, contour_holes, ref_patch_size, 0.5, annotations):
            if self.path.endswith('.qptiff'):
                patch_PIL = self.wsi[patch_level][coord[1]//self.level_downsamples[patch_level][0]:coord[1]//self.level_downsamples[patch_level][0]+patch_size, coord[0]//self.level_downsamples[patch_level][0]:coord[0]//self.level_downsamples[patch_level][0]+patch_size]
            else:
                patch_PIL = self.wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
            if custom_downsample > 1:
                patch_PIL = patch_PIL.resize((int(patch_size/custom_downsample), int(patch_size/custom_downsample)))
                
            if white_black:
                if WholeSlideImage.isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or WholeSlideImage.isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
                     return None
            return coord
        else:
            return None

    def process_spots(self, save_path, patch_level=0, patch_size=256, spot_coords_dir=None, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...",)
        
        df_spots = pd.read_csv(os.path.join(spot_coords_dir, str(self.name)+'.csv'))
        n_spots = len(df_spots)
        print("Total number of spots to process: ", n_spots)
        
        if patch_level == 1:
            # switch the coordinates from the center to top left
            results = (np.array(list(map(list, zip(df_spots['pxl_row_in_fullres'].astype(int)-patch_size, df_spots['pxl_col_in_fullres'].astype(int)-patch_size))))).astype(int)
            barcodes = np.asarray(df_spots['barcode']).astype("S")
        else:
            raise ValueError("Patch level not 1!!!")
        
        print('Extracted {} coordinates'.format(len(results)))

        if len(results)>1:
            asset_dict = {'coords' :          results, # at highest magnification
                          'barcodes':         barcodes} 
            
            attr = {'patch_size' :            patch_size,
                    'patch_level' :           patch_level,
                    'downsample':             self.level_downsamples[patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                    'level_dim':              self.level_dim[patch_level],
                    'name':                   self.name,
                    'save_path':              save_path}

            attr_dict = { 'coords' : attr}
        
            save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')

        else:
            raise ValueError("No spot found!!!")

        return self.hdf5_file
