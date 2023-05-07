import os
from datasets.dataset_generic import save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= -1,
                    help='fraction of labels (default: [0.25, 0.5, 0.75, 1.0])')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['mo-reg_tcga_hcc_349_ABRS-score_exp_cv_00X',
                                                 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622',

                                                 'mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X',
                                                 'mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X',
                                                 'mo-reg_merged-mondor-resection-biopsy_hcc_382_ABRS-score_exp_cv_00X',
                                                 
                                                 'mo-reg_ABtreated-biopsy_hcc_137_ABRS-score_cv_00X',
                                                 'mo-reg_other-systemic-treatments_hcc_49_ABRS-score_cv_00X'],
                    help='indentifier for the experimental settings, see the source code for complete list')

args = parser.parse_args()

if args.task == 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_00X':
    from datasets.dataset_generic_reg import Generic_WSI_Classification_Dataset
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_ABRS-score_Exp.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = len(dataset)
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622':
    from datasets.dataset_generic_reg import Generic_WSI_Classification_Dataset
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_ABRS-score_Exp.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = len(dataset)
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
    
elif args.task == 'mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X':
    from datasets.dataset_generic_reg import Generic_WSI_Classification_Dataset
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondorS2_hcc_225_ABRS-score_Exp.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = len(dataset)
    val_num = [0, 0]
    test_num = num_slides_cls
    
    
elif args.task == 'mo-reg_merged-mondor-resection-biopsy_hcc_382_ABRS-score_exp_cv_00X':
    from datasets.dataset_generic_reg import Generic_WSI_Classification_Dataset
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/merged-mondor-resection-biopsy_hcc_382_ABRS-score_Exp.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = len(dataset)
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X':
    from datasets.dataset_generic_reg import Generic_WSI_Classification_Dataset
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor-biopsy_hcc_157_ABRS-score_Exp.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = len(dataset)
    val_num = [0, 0]
    test_num = num_slides_cls

    
elif args.task == 'mo-reg_ABtreated-biopsy_hcc_137_ABRS-score_cv_00X':
    from datasets.dataset_generic_reg import Generic_WSI_Classification_Dataset
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/ABtreated-biopsy_hcc_137_ABRS-score_Exp.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = len(dataset)
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mo-reg_other-systemic-treatments_hcc_49_ABRS-score_cv_00X':
    from datasets.dataset_generic_reg import Generic_WSI_Classification_Dataset
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/other-systemic-treatments_hcc_49_ABRS-score_Exp.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = len(dataset)
    val_num = [0, 0]
    test_num = num_slides_cls

else:
    raise NotImplementedError

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



