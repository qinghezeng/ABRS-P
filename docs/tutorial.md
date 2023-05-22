## Development of deep learning model
***Dataset split***
```bash
python create_splits_seq.py --task mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622 --seed 1 --label_frac 1 --k 10
```

***Tissue detection***
```bash
python create_patches_fp.py \
        --source DIR_TCGA_WSI \
        --save_dir results \
        --patch_size 256 \
        --step_size 256 \
        --seg \
        --mask_save_dir results/masks_tumor_masked \
        --patch_save_dir results/patches_tumor_masked \
        --stitch_save_dir results/stitches_tumor_masked
```

***Tumor tissue patching***
```bash
# For 20x WSIs
python create_patches_fp.py \
        --source DIR_TCGA_WSI \
        --save_dir results \
        --patch_size 256 \
        --step_size 256 \
        --seg \
        --patch_level 0 \
        --process_list process_list_edited_20x.csv \
        --patch \
        --stitch \
        --mask_save_dir results/masks_tumor_masked \
        --patch_save_dir results/patches_tumor_masked \
        --stitch_save_dir results/stitches_tumor_masked \
        --use_annotations \
        --annotation_type ANNO_FORMAT \
        --annotation_dir DIR_ANNO

# For 40x WSIs, we need to downsample to 20x as there is no such a native level
python create_patches_fp.py \
        --source DIR_TCGA_WSI \
        --save_dir results \
        --patch_size 256 \
        --step_size 256 \
        --seg \
        --patch_level 0 \
        --custom_downsample 2 \
        --process_list process_list_edited_40x.csv \
        --patch \
        --stitch \
        --mask_save_dir results/masks_tumor_masked \
        --patch_save_dir results/patches_tumor_masked \
        --stitch_save_dir results/stitches_tumor_masked \
        --use_annotations \
        --annotation_type ANNO_FORMAT \
        --annotation_dir DIR_ANNO
```

***Feature extraction***
```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
        --data_dir results/patches_tumor_masked \
        --data_slide_dir DIR_TCGA_WSI \
        --csv_path ./dataset_csv/tcga_hcc_feature_349.csv \
        --feat_dir results/features_tumor_masked \
        --batch_size 256 \
        --target_patch_size 256
```

***Model training***

The model structure is compatible with multiple outputs (ABRS is single output).
```bash
CUDA_VISIBLE_DEVICES=0 python main_reg.py \
        --drop_out \
        --early_stopping \
        --lr 2e-4 \
        --k 10 \
        --label_frac 1 \
        --data_dir ./results/features_ctranspath-tcga-paip_tumor_masked \
        --results_dir ./results/training_multi-output_regression_patch \
        --exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50 \
        --bag_loss mse \
        --task mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622 \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --log_data \
        --project_name clam_hcc_reg_patch \
        --max_epochs 500 \
        --model_activation softplus \
        --extractor_model ctranspath-tcga-paip \
        --patch_pred > log_moreg_tcga_hcc_349_ABRS-score_exp_softplus_ctranspath-tcga-paip_patch_pred.txt
```

***Training visualization***
```bash
tensorboard --logdir=.
```
Or with weights&bias

***Inference on test split***
```bash
CUDA_VISIBLE_DEVICES=0 python eval_reg.py \
        --drop_out \
        --k 10 \
        --data_dir ./results/features_ctranspath-tcga-paip_tumor_masked \
        --results_dir ./results/training_multi-output_regression_patch \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622 \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --patch_pred
```

***Inference on the whole development dataset***
```bash
python create_splits_seq.py --task mo-reg_tcga_hcc_349_ABRS-score_exp_cv_00X --seed 1 --label_frac 1 --k 10

CUDA_VISIBLE_DEVICES=0 python eval_reg.py \
        --drop_out \
        --k 10 \
        --data_dir ./results/features_ctranspath-tcga-paip_tumor_masked \
        --splits_dir ./splits/mo-reg_tcga_hcc_349_ABRS-score_exp_cv_00X_100 \
        --results_dir ./results/training_multi-output_regression_patch \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_tcga_hcc_349_ABRS-score_exp_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --patch_pred
```

Use script ensembled_average_reg.py to aggregate the predictions from the 10 folds and save as fold_average.csv.

To evaluate the aggregated predictions
```bash
python eval_reg_ensembled-average.py \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv
```

***Attention-weighted patch prediction scores***

(taking fold 0 as an example)

```bash
python weighted_pred_score_reg.py \
        --drop_out \
        --results_dir ./results/training_multi-output_regression_patch \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1_cv \
        --model_type clam_mb_reg \
        --task mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622 \
        --model_size small-768 \
        --data_dir ./results/features_ctranspath-tcga-paip_tumor_masked \
        --fold 0 \
        --model_activation softplus
```

***Attention-weighted patch prediction heatmaps***

(taking fold 0 as an example)

```bash
CUDA_VISIBLE_DEVICES=0 python weighted_pred_map_fp_mb.py \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1_cv \
        --k 10 \
        --B 16 \
        --downscale 8 \
        --snapshot \
        --grayscale \
        --colormap \
        --blended \
        --data_root_dir ./results/patches_tumor_masked \
        --data_slide_dir DIR_TCGA_WSI \
        --target_patch_size 256 \
        --fold 0 \
        --norm rescale01 \
        --score_type weighted_pred_score
```

## External validation
***Dataset split***

```bash
python create_splits_seq.py --task mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X --seed 1 --label_frac 1 --k 10

python create_splits_seq.py --task mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X --seed 1 --label_frac 1 --k 10
```

WSIs used for external validation are not allowed to be shared due to privacy concerns.
- Script create_patches_fp.py was used for tissue segmentation, patching and stitching;
- Script extract_features_fp.py was used for feature extraction and was set up similarly to the TCGA dataset.

***External validation on resection samples***
```bash
CUDA_VISIBLE_DEVICES=0 python eval_reg.py \
        --drop_out \
        --k 10 \
        --data_dir ./results/features_ctranspath-tcga-paip_mondorS2_tumor_masked \
        --splits_dir ./splits/mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X_100 \
        --results_dir ./results/training_multi-output_regression_patch \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondorS2_hcc_tumor-masked_ctranspath-tcga-paip_225_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --patch_pred

# Use script ensembled_average_reg.py to aggregate the predictions from the 10 folds and save as fold_average.csv.

python eval_reg_ensembled-average.py \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondorS2_hcc_tumor-masked_ctranspath-tcga-paip_225_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv

CUDA_VISIBLE_DEVICES=0 python weighted_pred_score_reg.py \
        --drop_out \
        --k 10 \
        --results_dir ./results/training_multi-output_regression_patch \
        --splits_dir ./splits/mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X_100 \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondorS2_hcc_tumor-masked_ctranspath-tcga-paip_225_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --data_dir ./results/features_ctranspath-tcga-paip_mondorS2_tumor_masked

CUDA_VISIBLE_DEVICES=0 python weighted_pred_map_fp_mb_ensembled-average.py \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondorS2_hcc_tumor-masked_ctranspath-tcga-paip_225_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --k 10 \
        --downscale 16 \
        --snapshot \
        --grayscale \
        --colormap \
        --blended \
        --data_root_dir ./results/patches_mondorS2_tumor_masked \
        --target_patch_size 256 \
        --data_slide_dir DIR_RESECTION_WSI \
        --B 8 \
        --norm rescale01
```

***External validation on biopsy samples***
```bash
CUDA_VISIBLE_DEVICES=0 python eval_reg.py \
        --drop_out \
        --k 10 \
        --data_dir ./results/features_ctranspath-tcga-paip_tagseq-biopsy_tumor_masked \
        --splits_dir ./splits/mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X_100 \
        --results_dir ./results/training_multi-output_regression_patch \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondor-biopsy_hcc_tumor-masked_ctranspath-tcga-paip_157_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --patch_pred

# Use script ensembled_average_reg.py to aggregate the predictions from the 10 folds and save as fold_average.csv.

python eval_reg_ensembled-average.py \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondor-biopsy_hcc_tumor-masked_ctranspath-tcga-paip_157_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv

CUDA_VISIBLE_DEVICES=0 python weighted_pred_score_reg.py \
        --drop_out \
        --k 10 \
        --results_dir ./results/training_multi-output_regression_patch \
        --splits_dir ./splits/mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X_100 \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondor-biopsy_hcc_tumor-masked_ctranspath-tcga-paip_157_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --data_dir ./results/features_ctranspath-tcga-paip_tagseq-biopsy_tumor_masked

CUDA_VISIBLE_DEVICES=0 python weighted_pred_map_fp_mb_ensembled-average.py \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_mondor-biopsy_hcc_tumor-masked_ctranspath-tcga-paip_157_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --k 10 \
        --downscale 16 \
        --snapshot \
        --grayscale \
        --colormap \
        --blended \
        --data_root_dir ./results/patches_tagseq-biopsy_tumor_masked \
        --target_patch_size 256 \
        --data_slide_dir DIR_BIOPSY_WSI \
        --B 8 \
        --norm rescale01
```

## Test on treated data
***AtezoBeva treated biopsy samples***
```bash
# As no gene expression data available for treated samples, 
# here we created dataset_csv/ABtreated-biopsy_hcc_137_ABRS-score_Exp.csv with pseudo values
python create_splits_seq.py --task mo-reg_ABtreated-biopsy_hcc_137_ABRS-score_cv_00X --seed 1 --label_frac 1 --k 10

CUDA_VISIBLE_DEVICES=0 python eval_reg.py \
        --drop_out \
        --k 10 \
        --data_dir ./results/features_ctranspath-tcga-paip_ABtreated-biopsy_tumor_masked \
        --splits_dir ./splits/mo-reg_ABtreated-biopsy_hcc_137_ABRS-score_cv_00X_100 \
        --results_dir ./results/training_multi-output_regression_patch \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_ABtreated-biopsy_hcc_tumor-masked_ctranspath-tcga-paip_137_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_ABtreated-biopsy_hcc_137_ABRS-score_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus  \
        --patch_pred
```

Use maj-voting_with_thresholds.py to generate majority-voting predictions from the 10 folds (with biopsy-157 thresholds from calculate_median_reg.ipynb)

***Other drugs treated biopsy samples***
```bash
# dataset_csv/other-systemic-treatments_hcc_49_ABRS-score_Exp.csv was created with pseudo values
python create_splits_seq.py --task mo-reg_other-systemic-treatments_hcc_49_ABRS-score_cv_00X --seed 1 --label_frac 1 --k 10

CUDA_VISIBLE_DEVICES=0 python eval_reg.py \
        --drop_out \
        --k 10 \
        --data_dir ./results/features_ctranspath-tcga-paip_other-systemic-treatments_tumor_masked \
        --splits_dir ./splits/mo-reg_other-systemic-treatments_hcc_49_ABRS-score_cv_00X_100 \
        --results_dir ./results/training_multi-output_regression_patch \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_other-systemic-treatments_hcc_tumor-masked_ctranspath-tcga-paip_49_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv \
        --task mo-reg_other-systemic-treatments_hcc_49_ABRS-score_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --patch_pred
```

Use maj-voting_with_thresholds.py to generate majority-voting predictions from the 10 folds (with biopsy-157 thresholds from calculate_median_reg.ipynb)

## Validate on Visium spatial transcriptomics (ST)
***- Process counts from Visium SpaceRanger***

script process_st_hcc4.R, line 1-59

***- Plot the ST figures***

script process_st_hcc4.R, line 61-122

***- Matching spot barcodes to the centers' pixel coordinates in the WSI***

script format_spot_coords.py

***- Patch extraction without tissue segmentation***

Here ST spot centers were used as patch centers.
```bash
python create_patches_visium_fp.py \
        --source DIR_ST_WSI \
        --save_dir results/visium \
        --patch_size 256 \
        --patch_level 1 \
        --mask_save_dir results/visium/masks \
        --patch_save_dir results/visium/patches \
        --stitch_save_dir results/visium/stitches \
        --process_list process_list_edited_visium.csv \
        --spot_coords_dir DIR_SPOT_COORDS \
        --patch \
        --stitch
```

***- Feature extraction***

Create dataset_csv/visium_st_feature.csv.
```bash
$ CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
        --data_dir results/visium/patches \
        --data_slide_dir DIR_ST_WSI \
        --csv_path ./dataset_csv/visium_st_feature.csv \
        --feat_dir results/visium/features_ctranspath-tcga-paip \
        --batch_size 256 \
        --target_patch_size 256 \
        --visium \
        --model ctranspath-tcga-paip \
        --custom_transforms resize224_imagenet
```

***- Dataset split***

Created dataset_csv/st_hcc_4_ABRS-score_Exp.csv with pseudo values.
```bash
$ python create_splits_seq.py --task mo-reg_st_hcc_4_ABRS-score_cv_00X --seed 1 --label_frac 1 --k 10
```

***- Patch prediction***

```bash
# Generate attention-weighted patch predictions for 10 folds
$ CUDA_VISIBLE_DEVICES=0 python weighted_pred_score_reg.py \
        --drop_out \
        --k 10 \
        --results_dir ./results/training_multi-output_regression_patch \
        --splits_dir ./splits/mo-reg_st_hcc_4_ABRS-score_cv_00X_100 \
        --models_exp_code mo-reg_tcga_hcc_tumor-masked_ctranspath-tcga-paip_349_ABRS-score_exp_cv_622_CLAM-MB-softplus-patch_50_s1 \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_st_hcc_ctranspath-tcga-paip_4_ABRS-score_exp_cv_00X_CLAM-MB-softplus_50_s1_cv \
        --task mo-reg_st_hcc_4_ABRS-score_cv_00X \
        --model_type clam_mb_reg \
        --model_size small-768 \
        --model_activation softplus \
        --data_dir results/visium/features_ctranspath-tcga-paip \
        --feature_bags D2_rot90.pt E4_rot90.pt E6_rot90.pt E7_rot90.pt

# Plot the average prediction maps
$ CUDA_VISIBLE_DEVICES=0 python weighted_pred_map_fp_mb_ensembled-average.py \
        --eval_dir ./eval_results_tcga-349_tumor_masked_multi-output_regression_patch \
        --save_exp_code mo-reg_st_hcc_ctranspath-tcga-paip_4_ABRS-score_exp_cv_00X_CLAM-MB-softplus_50_s1_cv \
        --k 10 \
        --downscale 16 \
        --snapshot \
        --grayscale \
        --colormap \
        --blended \
        --data_root_dir results/visium/patches \
        --target_patch_size 256 \
        --data_slide_dir DIR_ST_WSI \
        --B 8 \
        --heatmap_crop_size 218 \
        --patch_bags D2_rot90.h5 E4_rot90.h5 E6_rot90.h5 E7_rot90.h5 \
        --spot_coords_dir DIR_SPOT_COORDS \
        --norm rescale01
```

***- Select the 100 high ABRS-P patches and 100 low ABRS-P patches***

script select_topN.py

***- Differential gene expression analysis***

script process_st_hcc4.R, line 124-340

