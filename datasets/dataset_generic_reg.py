from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import h5py

from utils.utils_reg import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = []
    for i in range(len(split_datasets)):
        try: 
            splits.append(split_datasets[i].slide_data['slide_id'])
        except AttributeError as error:
            print(error)
            splits.append(pd.Series())
        
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
    
        list_len = []
        for dset in split_datasets:
            try:
                list_len.append(len(dset))
            except TypeError as error:
                print(error)
                list_len.append(0)
        bool_array = np.repeat(one_hot, list_len, axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)
    print()

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv',
        shuffle = False, 
        seed = 7, 
        print_info = True,
        label_dict = {},
        ignore=[],
        patient_strat=False,
        label_col = None,
        patient_voting = 'max',
        ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes=len(label_col)
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = []
        self.concat_features = False
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        self.csv_path = csv_path

        slide_data = pd.read_csv(csv_path)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data

        patients = np.unique(np.array(slide_data['case_id'])) # get unique patients
        
        self.patient_data = {'case_id':patients, 'label':self.label_col}

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])

        else:
            return len(self.slide_data)

    def summarize(self):
        print('Set in multi-output regression mode...')
        print("label column: {}".format(self.label_col))
        print("label path: {}".format(self.csv_path))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print(f'Patient-LVL; Number of samples {len(self.patient_data["case_id"])}')
        print(f'Slide-LVL; Number of samples {len(self.slide_data)}')

    def create_splits(self, k = 3, val_num = 25, test_num = 40, label_frac = 1.0, custom_test_ids = None):
        settings = {
                    'n_splits' : k, 
                    'val_num' : val_num, 
                    'test_num': test_num,
                    'label_frac': label_frac,
                    'seed': self.seed,
                    'custom_test_ids': custom_test_ids
                    }

        if self.patient_strat:
            settings.update({'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))] 

            for split in range(len(ids)): 
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
            assert len(self.slide_data[mask].dropna()) == len(self.slide_data[mask]), print('NaN in labels!')
            split = Generic_Split(df_slice, data_dir=self.data_dir, concat_features=self.concat_features, num_classes=self.num_classes, label_col=self.label_col, csv_path=self.csv_path)
        else:
            split = None
        
        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, concat_features=self.concat_features, num_classes=self.num_classes, label_col=self.label_col, csv_path=self.csv_path)
        else:
            split = None
        
        return split


    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir, concat_features=self.concat_features, num_classes=self.num_classes, label_col=self.label_col, csv_path=self.csv_path)

            else:
                train_split = None
            
            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir, concat_features=self.concat_features, num_classes=self.num_classes, label_col=self.label_col, csv_path=self.csv_path)

            else:
                val_split = None
            
            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir, concat_features=self.concat_features, num_classes=self.num_classes, label_col=self.label_col, csv_path=self.csv_path)
            
            else:
                test_split = None
            
        
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')
            
        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data[self.label_col].loc[ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):

        if return_descriptor:
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((1, len(columns)), 0, dtype=np.int32), index= ['num'],
                            columns= columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        if return_descriptor:
            df.loc['num', 'train'] = count
        
        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        if return_descriptor:
            df.loc['num', 'val'] = count

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        if return_descriptor:
            df.loc['num', 'test'] = count

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1) 
        df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
        data_dir, 
        concat_features,
        **kwargs):
    
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False
        self.concat_features = concat_features

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data[self.label_col].loc[idx]
        label = torch.FloatTensor(label.values)

        if not self.use_h5:
            if self.data_dir:
                if self.concat_features:
                    features = []
                    for data_dir_ in self.data_dir:
                        if os.path.isfile(os.path.join(data_dir_,'{}.pt'.format(slide_id))):
                            full_path = os.path.join(data_dir_,'{}.pt'.format(slide_id))
                            features.append(torch.load(full_path))
                        else:
                            raise FileNotFoundError("Please check the data_dir!")
                    features = torch.cat(features, 1)
                else:
                    find = False
                    for data_dir_ in self.data_dir:
                        if os.path.isfile(os.path.join(data_dir_,'{}.pt'.format(slide_id))):
                            full_path = os.path.join(data_dir_,'{}.pt'.format(slide_id))
                            find = True
                            break
                    if not find:
                        raise FileNotFoundError(f"Please check the data_dir!\n for {slide_id}")
                    features = torch.load(full_path)
                return features, label
            
            else:
                return slide_id, label

        else:
            if self.data_dir:
                if self.concat_features:
                    features = []
                    for data_dir_ in self.data_dir:
                        if os.path.isfile(os.path.join(data_dir_,'{}.h5'.format(slide_id))):
                            full_path = os.path.join(data_dir_,'{}.h5'.format(slide_id))
                            with h5py.File(full_path,'r') as hdf5_file:
                                features.append(torch.from_numpy(hdf5_file['features'][:]))
                                coords = hdf5_file['coords'][:]

                        else:
                            raise FileNotFoundError("Please check the data_dir!")
                    features = torch.cat(features, 1)
                else:
                    find = False
                    for data_dir_ in self.data_dir:
                        if os.path.isfile(os.path.join(data_dir_,'{}.h5'.format(slide_id))):
                            full_path = os.path.join(data_dir_,'{}.h5'.format(slide_id))
                            find = True
                            break
                    if not find:
                        raise FileNotFoundError("Please check the data_dir!")
                    with h5py.File(full_path,'r') as hdf5_file:
                        features = hdf5_file['features'][:]
                        coords = hdf5_file['coords'][:]
        
                    features = torch.from_numpy(features)
            return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, concat_features=False, data_dir=None, num_classes=2, label_col=None, csv_path='dataset_csv/ccrcc_clean.csv'):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.concat_features = concat_features
        self.num_classes = num_classes
        self.label_col = label_col
        self.csv_path = csv_path

    def __len__(self):
        return len(self.slide_data)