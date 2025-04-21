from glob import glob
from tqdm import tqdm
import mne
import torch
import scipy
import numpy as np
from braindecode.datautil.preprocess import exponential_moving_standardize
from dataloader.augmentation import cutcat,  cutcat_2
from typing import List, Union
import numpy as np
from mne.filter import resample
from torch.utils.data import Dataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import Preprocessor
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.windowers import create_windows_from_events
from filters import load_filterbank, butter_fir_filter

class Nguyen(torch.utils.data.Dataset):
    def __init__(self, args):
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        #self.subtask = args.subtask
        
        self.data, self.label = self.get_brain_data()
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]

        if not self.is_test:
            data, label = self.augmentation(data, label)

        sample = {'data': data, 'label': label}

        return sample
    
    def get_brain_data(self):
        
        filelist = sorted(glob(f'{self.base_path}/*/*.mat'))
        #filelist = sorted(glob(f'{self.base_path}/*.mat')) # subtask training
        data = np.array([])
        label = np.array([])
        
        for idx, filename in enumerate(tqdm(filelist)):
            if self.is_test and idx != self.target_subject: continue #all data training
            #if idx != self.target_subject: continue
            
            print(f'LOG >>> Filename: {filename}')
            
            raw = scipy.io.loadmat(filename)
            raw=raw["eeg_data_wrt_task_rep_no_eog_256Hz_last_beep"].reshape(-1,1)
            
            for i in range(len(raw)):
                r=raw[i,0]
                raw[i,0]=np.delete(r,[0,9,32,63],0)
            
            temp=np.zeros((len(raw),1,60,1280))
            for i in range(len(raw)):
                temp[i,0]=raw[i,0]
            
            lb=np.array([])
            if '\\Long_words\\' in filename:
                lb=np.array(100*[0]+100*[1])
        
            elif '\\Short_Long_words\\' in filename:
                if '_8d_' in filename or '_9b_' in filename:
                    lb=np.array(80*[0]+80*[2])
                else:
                    lb=np.array(100*[0]+100*[2])
                    
            elif '\\Short_words\\' in filename:
                lb=np.array(100*[3]+100*[2]+100*[4])
        
            elif '\\Vowels\\' in filename:
                lb=np.array(100*[5]+100*[6]+100*[7])
                
            

            if self.is_test:
                temp=temp[::6,:,:,:]
                lb=lb[::6]
            else:
                temp=np.delete(temp,np.arange(0,len(temp),6),0)
                lb=np.delete(lb,np.arange(0,len(lb),6),0)
            
            if len(data) == 0:
                data = temp
                label = lb
            else:
                data = np.concatenate((data, temp), axis=0)
                label = np.concatenate((label, lb), axis=0)
        
        return data, label
        
    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        #data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=10)
        return data, label
    


class BciCompetV3(torch.utils.data.Dataset):
    def __init__(self, args):
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/Training set/Data_Sample*.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/Validation set/Data_Sample*.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            #if self.is_test and idx != self.target_subject: continue
            if idx != self.target_subject: continue
            
            print(f'LOG >>> Filename: {filename}')
            if not self.is_test:
                raw_train=scipy.io.loadmat(filename)
                x=raw_train['epo_train'][0]['x'][0]
                y=raw_train['epo_train'][0]['y'][0]
                x=np.transpose(x,(2,1,0))
                x=x.reshape((300,1,64,795))
                y=np.argmax(y,axis=0)
                x = torch.tensor(x.copy(), dtype=torch.float32)
                y = torch.tensor(y.copy(), dtype=torch.long)
                
            else:
                raw_test=scipy.io.loadmat(filename)
                x=raw_test['epo_validation'][0]['x'][0]
                y=raw_test['epo_validation'][0]['y'][0]
                x=np.transpose(x,(2,1,0))
                x=x.reshape((50,1,64,795))
                y=np.argmax(y,axis=0)
                x = torch.tensor(x.copy(), dtype=torch.float32)
                y = torch.tensor(y.copy(), dtype=torch.long)
                
            if len(data) == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
        
        return data, label
            
    
    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=10)
        return data, label

class KaraOne(torch.utils.data.Dataset):
    def __init__(self, args):
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        
        filelist = sorted(glob(f'{self.base_path}/p/spoclab/users/szhao/EEG/data/*/*.cnt'))
        
        data = np.array([])
        label = np.array([])
        
        for idx, filename in enumerate(tqdm(filelist)):
            if self.is_test and idx != self.target_subject: continue #all data training
            #if idx != self.target_subject: continue
            
            print(f'LOG >>> Filename: {filename}')
            
            raw=mne.io.read_raw_cnt(filename)
            events, annot = mne.events_from_annotations(raw)
            
            raw.load_data()
            #raw.filter(0., 40., fir_design='firwin')
            raw.info['bads'] += ['M1', 'M2', 'Trigger', 'EKG', 'EKG', 'VEO', 'HEO']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0, 3
            if not self.is_test:
                event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10}) if idx != 3 \
                else dict({'769': 5,'770': 6,'771': 7,'772': 8})
            else:
                event_id = dict({'783': 7})
            
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 4))
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]
            
            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1

            if self.is_test:
                splited_data=splited_data[::6,:,:,:]
                label_list=label_list[::6]
            else:
                splited_data=np.delete(splited_data,np.arange(0,len(splited_datated),6),0)
                label_list=np.delete(label_list,np.arange(0,len(label_list),6),0)
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)
        
        return data, label
            
    

class ThinkOutLoud(torch.utils.data.Dataset):
    def __init__(self, args):
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/Training set/Data_Sample*.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/Validation set/Data_Sample*.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            if idx != self.target_subject: continue
            
            print(f'LOG >>> Filename: {filename}')
            if not self.is_test:
                raw1=scipy.io.loadmat(filename).copy()
                np.array
                x1=raw1['epo_train'].copy()
                x1=np.ascontiguousarray(x1)
                x1=x1[0].copy()
                x1=np.ascontiguousarray(x1)
                x1=x1['x'].copy()
                x1=np.ascontiguousarray(x1)
                x1=x1[0].copy()
                x1=np.ascontiguousarray(x1)
                y1=raw1['epo_train'].copy()
                y1=np.ascontiguousarray(y1)
                y1=y1[0].copy()
                y1=np.ascontiguousarray(y1)
                y1=y1['y'].copy()
                y1=np.ascontiguousarray(y1)
                y1=y1[0].copy()
                y1=np.ascontiguousarray(y1)
                
                
                
                
                x2=np.transpose(x1,(2,1,0)).copy()
                x2=np.ascontiguousarray(x2)
                x=x2.reshape((300,1,64,795)).copy()
                x=np.ascontiguousarray(x)
                y=np.argmax(y1,axis=0).copy()
                y1=np.ascontiguousarray(y1)
                x=np.ascontiguousarray(x)
                x=torch.from_numpy(x.copy())
                
            else:
                raw=scipy.io.loadmat(filename)
                x=raw['epo_validation'][0]['x'][0]
                y=raw['epo_validation'][0]['y'][0]
                
                x=np.transpose(x,(2,1,0))
                x=x.reshape((50,1,64,795))
                
                y=np.argmax(y,axis=0)
            
            if len(data) == 0:
                data = x
                label = y
            else:
                data = np.concatenate((data, x), axis=0)
                label = np.concatenate((label, y), axis=0)
        
        return data, label
            
    

class BCICompet2aIV(torch.utils.data.Dataset):
    def __init__(self, args):
        
        '''
        * 769: Left
        * 770: Right
        * 771: foot
        * 772: tongue
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/*T*.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E*.gdf'))
        
        label_filelist = sorted(glob(f'{self.base_path}/*T.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if idx != self.target_subject: continue
                    
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)
            
            raw.load_data()
            raw.filter(0., 40., fir_design='firwin')
            raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0, 3
            if not self.is_test:
                event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10}) if idx != 3 \
                else dict({'769': 5,'770': 6,'771': 7,'772': 8})
            else:
                event_id = dict({'783': 7})
            
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 4))
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]
            
            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)



        return torch.Tensor(data), torch.Tensor(label)
    

    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=8)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=8)
        return data, label
    
    
class BCICompet2bIV(torch.utils.data.Dataset):
    def __init__(self, args):
        '''
        * 769: left
        * 770: right
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/*T.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E.gdf'))
        
        label_filelist = sorted(glob(f'{self.base_path}/*T.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if not self.is_test:
                if idx // 3 != self.target_subject: continue
            else:
                if idx // 2 != self.target_subject: continue
                        
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)

            raw.load_data()
            raw.filter(0., 40., fir_design='firwin')
            raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0., 3.
            if not self.is_test: event_id = dict({'769': annot['769'], '770': annot['770']})
            else: event_id = dict({'783': annot['783']})
                
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 4))
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]

            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)
        
        return data, label
            
    
    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=10)
        return data, label
    
class OpenBMI(torch.utils.data.Dataset):
    """
    Not supported subject-independent manner not yet.
    Therefore, we recommend session-to-session manner with single subject.
    """

    def __init__(self, args):
        import warnings
        warnings.filterwarnings('ignore')
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args

        self.data, self.label = self.get_brain_data()

    def get_brain_data(self):

        x_bundle, y_bundle = [], []
        for (low_hz, high_hz) in [[0, 40]]:
            x_list = []
            y_list = []
            # Load data from MOABBDataset
            dataset = MOABBDataset(dataset_name="Lee2019_MI", subject_ids=self.target_subject+1)

            # Preprocess data
            factor_new = 1e-3
            init_block_size = 1000

        preprocessors = [
            # Keep only EEG sensors
            Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False, apply_on_array=True),
            # Convert from volt to microvolt
            Preprocessor(fn=lambda x: x * 1e+06, apply_on_array=True),
            # Apply bandpass filtering
            Preprocessor(fn='filter', l_freq=low_hz, h_freq=high_hz, apply_on_array=True),
            # Apply exponential moving standardization
            Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new,
                         init_block_size=init_block_size, apply_on_array=True)
        ]
        preprocess(dataset, preprocessors)

        # Check sampling frequency
        sfreq = dataset.datasets[0].raw.info['sfreq']
        if not all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]):
            raise ValueError("Not match sampling rate.")

        # Divide data by trial
        trial_start_offset_samples = int(0 * sfreq)

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True
        )


        # Make session-to-session data (subject dependent)
        if self.is_test == False:
            for trial in windows_dataset.split('session')['session_1']:
                x_list.append(trial[0])
                y_list.append(trial[1])
        else:
            for trial in windows_dataset.split('session')['session_2']:
                x_list.append(trial[0])
                y_list.append(trial[1])

        # Return numpy array
        x_list = np.array(x_list)
        y_list = np.array(y_list)

        # Cut time points
        tmin, tmax = 0., 3.0
        x_list = x_list[..., int(tmin * sfreq): int(tmax * sfreq)]

        # Resampling
        if self.args.downsampling is not None:
            x_list = resample(np.array(x_list, dtype=np.float64), self.args.downsampling / sfreq)

        x_bundle.append(x_list)
        y_bundle.append(y_list)

        data = np.stack(x_bundle, axis=1)
        data = data[:, :, 20:40, :]
        label = np.array(y_bundle[0])
        return data, label

    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=10)
        return data, label
    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     sample = [self.x[idx], self.y[idx]]
    #     return sample

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]

        if not self.is_test:
            data, label = self.augmentation(data, label)

        sample = {'data': data, 'label': label}

        return sample



def get_dataset(config_name, args):
    
    if 'bcicompet2a_config' in config_name:
        dataset = BCICompet2aIV(args)
        if args['filter_bank']:
            #### FBCNet####
            # data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1], len(args['bank']),
            #                             dataset.data.shape[2], dataset.data.shape[3]))
            #
            # for num, Fband in enumerate(args['bank']):
            #     bw = np.array(Fband)
            #     filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
            #     X_filtered = np.zeros_like(dataset.data)
            #     for i, trial in enumerate(dataset.data):
            #         # filtering
            #         trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            #         trail_filter = trail_filter.reshape(1, 22, 751)
            #         X_filtered[i, :, :, :] = trail_filter
            #     data_filterbank[:, :, num, :, :] = X_filtered

            #### IFNet####
            data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1],2*dataset.data.shape[2], dataset.data.shape[3]))

            for num, Fband in enumerate(args['bank']):
                bw = np.array(Fband)
                filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
                X_filtered = np.zeros_like(dataset.data)
                for i, trial in enumerate(dataset.data):
                    # filtering
                    trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
                    trail_filter = trail_filter.reshape(1, 22, 751)
                    X_filtered[i, :, :, :] = trail_filter
                data_filterbank[:, :, num*dataset.data.shape[2]: (num+1)*dataset.data.shape[2], :] = X_filtered
            dataset.data = data_filterbank
        else:
            dataset = dataset
    elif 'bcicompet2b_config' in config_name:
        dataset = BCICompet2bIV(args)
        if args['filter_bank']:
            #### FBCNet####
            # data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1], len(args['bank']),
            #                             dataset.data.shape[2], dataset.data.shape[3]))
            #
            # for num, Fband in enumerate(args['bank']):
            #     bw = np.array(Fband)
            #     filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
            #     X_filtered = np.zeros_like(dataset.data)
            #     for i, trial in enumerate(dataset.data):
            #         # filtering
            #         trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            #         trail_filter = trail_filter.reshape(1, 3, 751)
            #         X_filtered[i, :, :, :] = trail_filter
            #     data_filterbank[:, :, num, :, :] = X_filtered

            #### IFNet####
            data_filterbank = np.zeros(
                (dataset.data.shape[0], dataset.data.shape[1], 2 * dataset.data.shape[2], dataset.data.shape[3]))

            for num, Fband in enumerate(args['bank']):
                bw = np.array(Fband)
                filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
                X_filtered = np.zeros_like(dataset.data)
                for i, trial in enumerate(dataset.data):
                    # filtering
                    trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
                    trail_filter = trail_filter.reshape(1, 3, 751)
                    X_filtered[i, :, :, :] = trail_filter
                data_filterbank[:, :, num * dataset.data.shape[2]: (num + 1) * dataset.data.shape[2], :] = X_filtered

            dataset.data = data_filterbank
        else:
            dataset = dataset

    elif 'KUMI_config' in config_name:
        dataset = OpenBMI(args)
        if args['filter_bank']:
            #### FBCNet####
            # data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1], len(args['bank']),
            #                             dataset.data.shape[2], dataset.data.shape[3]))
            #
            # for num, Fband in enumerate(args['bank']):
            #     bw = np.array(Fband)
            #     filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
            #     X_filtered = np.zeros_like(dataset.data)
            #     for i, trial in enumerate(dataset.data):
            #         # filtering
            #         trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            #         trail_filter = trail_filter.reshape(1, 22, 751)
            #         X_filtered[i, :, :, :] = trail_filter
            #     data_filterbank[:, :, num, :, :] = X_filtered

            #### IFNet####
            data_filterbank = np.zeros(
                (dataset.data.shape[0], dataset.data.shape[1], 2 * dataset.data.shape[2], dataset.data.shape[3]))

            for num, Fband in enumerate(args['bank']):
                bw = np.array(Fband)
                filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
                X_filtered = np.zeros_like(dataset.data)
                for i, trial in enumerate(dataset.data):
                    # filtering
                    trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
                    trail_filter = trail_filter.reshape(1, 20, 751)
                    X_filtered[i, :, :, :] = trail_filter
                data_filterbank[:, :, num * dataset.data.shape[2]: (num + 1) * dataset.data.shape[2], :] = X_filtered

            dataset.data = data_filterbank
        else:
            dataset = dataset
    
    elif 'Nguyen_config' in config_name:
        dataset = Nguyen(args)
        #import dtcwt
        
        #transform = dtcwt.Transform1d()
        #for i in range(len(dataset.data)):
        #    for j in range(60):
        #        vecs = transform.forward(dataset.data[i][0][j], nlevels=5)
        #        dataset.data[i][0][j] = transform.inverse(vecs)
        
    elif 'bcicompetv3' in config_name:
        from filters import butter_bandpass_filter
        from scipy.signal import iirnotch
        dataset=BciCompetV3(args)
        
        dataset.data= butter_bandpass_filter(dataset.data,8,70,256,5)
        [b,a]= iirnotch(60, 30, 256)
        dataset.data= scipy.signal.filtfilt(b,a, dataset.data)
        
        import dtcwt
        
        transform = dtcwt.Transform1d()
        for i in range(len(dataset.data)):
            for j in range(64):
                vecs = transform.forward(np.concatenate((dataset.data[i][0][j],[0])), nlevels=5)
                dataset.data[i][0][j] = transform.inverse(vecs)[:-1]
        
    elif 'KaraOne' in config_name:
        dataset = KaraOne(args)
        
    elif 'MultiDataset' in config_name:
        from filters import butter_bandpass_filter
        from scipy.signal import iirnotch
        if args.dataset =="BCI Competition V-3":
            dataset= BciCompetV3(args)
        
            dataset.data= butter_bandpass_filter(dataset.data,8,70,256,5)
            [b,a]= iirnotch(60, 30, 256)
            dataset.data= scipy.signal.filtfilt(b,a, dataset.data)
        
        elif args.dataset =="KaraOne":
            dataset= KaraOne(args)
            dataset.data= butter_bandpass_filter(dataset.data,8,70,256,5)
            [b,a]= iirnotch(60, 30, 256)
            dataset.data= scipy.signal.filtfilt(b,a, dataset.data)
            
        elif args.dataset =="Think Out Loud":
            dataset= ThinkOutLoud(args)
            dataset.data= butter_bandpass_filter(dataset.data,8,70,256,5)
            [b,a]= iirnotch(60, 30, 256)
            dataset.data= scipy.signal.filtfilt(b,a, dataset.data)
            
    
            
    else:
        raise Exception('get_dataset function Wrong dataset input!!!')

    return dataset
