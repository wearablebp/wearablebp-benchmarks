from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import torch
from scipy.signal import lfilter, firwin, savgol_filter, cheby2, butter, sosfiltfilt, sosfilt
from scipy.interpolate import interp1d
from pyampd.ampd import find_peaks, find_peaks_adaptive

class jeong21_dataloader(Dataset):
    # load UCI MIMIC-II database
    # use 0.5-20Hz bandpass
    # [-1, 1] minmax scaling
    # extract peaks and use mean as sbp and dbp ground truth
    
    def __init__(self, fpath, idxs, opt):
        self.fpath = fpath
        self.opt = opt
        self.idxs = idxs
        self.dset = h5py.File(fpath, 'r')

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # call opt = opt() beforehand
        # returns [1, signal length, 1]
        subj = np.fromiter(self.dset.keys(), dtype=float)[idx].astype(int).astype(str)
        d = self.dset[subj]
        n = np.random.randint(0, d.shape[0])
        window_size = 250
        window_lb = np.random.randint(0, d.shape[2]-window_size)
        X1 = (d[n, 0, window_lb:window_lb+window_size] - d[n, 0, window_lb:window_lb+window_size].min())/(d[n, 0, window_lb:window_lb+window_size].max() - d[n, 0, window_lb:window_lb+window_size].min())
        X1 = X1*2-1
        sos = butter(N=6, Wn=[0.5, 20], btype='bandpass', fs=125, output='sos')
        X1 = sosfiltfilt(sos, X1)
        X1 = torch.tensor(X1.copy(), dtype=torch.float32)
        X2 = (d[n, 2, window_lb:window_lb+window_size] - d[n, 2, window_lb:window_lb+window_size].min())/(d[n, 2, window_lb:window_lb+window_size].max() - d[n, 2, window_lb:window_lb+window_size].min())
        X2 = X2*2-1
        X2 = torch.tensor(X2.copy(), dtype=torch.float32)
        X = torch.cat([X1.unsqueeze(0), X2.unsqueeze(0)], axis=0)
        
        y = d[n, 2, window_lb:window_lb+window_size]
        try:
            sbp_idxs = find_peaks(y, scale=200)
            if len(sbp_idxs) > 0:
                sbp_idxs_del = []
                for i in range(len(sbp_idxs)):
                    if y[i] < 30:
                        sbp_idxs_del.append(i)
                sbp_idxs = np.delete(sbp_idxs, sbp_idxs_del)
        except:
            sbp_idxs = np.array([y.argmax()])
        if len(sbp_idxs) == 0:
            sbp_idxs = np.array([y.argmax()])
        
        try:
            dbp_idxs = find_peaks(-y, scale=200)
            if len(dbp_idxs) > 0:
                dbp_idxs_del = []
                for i in range(len(dbp_idxs)):
                    if y[i] < 30:
                        dbp_idxs_del.append(i)
                dbp_idxs = np.delete(dbp_idxs, dbp_idxs_del)
        except:
            dbp_idxs = np.array([y.argmin()])
        if len(dbp_idxs) == 0:
            dbp_idxs = np.array([y.argmin()])
            
        y = torch.tensor([np.median(y[sbp_idxs]), np.median(y[dbp_idxs])], dtype=torch.float32).to(self.opt.device)
        return X, y

class yang18_dataloader(Dataset):
    # load ppg-bp dataset from https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10820/2502315/Cuff-less-blood-pressure-measurement-using-fingertip-photoplethysmogram-signals-and/10.1117/12.2502315.full?SSO=1
    # median filter, FIR filter and cubic spline
    # downsample to 125Hz instead of 1000Hz. 125Hz compatible with algorithms
    # does not give details about filtering parameters
    # additional processing: min-max normalization
    
    def __init__(self, fpath, idxs, opt):
        self.fpath = fpath
        self.opt = opt
        self.idxs = idxs
        self.dset = h5py.File(fpath, 'r')

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # call opt = opt() beforehand
        # returns [1, signal length, 1]
        d = self.dset[self.idxs[idx]]
        n = np.random.randint(0, d.shape[0])
        X = (d[n, 0, :] - d[n, 0, :].min())/(d[n, 0, :].max() - d[n, 0, :].min())
        X = X*2-1
        X = downsample(X, 1000)
        X = torch.tensor(X.copy(), dtype=torch.float32)
        
        randseg_idx = np.random.randint(0, X.shape[1]-128) 
        X = X[:, randseg_idx:randseg_idx+128]
        y = d[n, 1, randseg_idx:randseg_idx+128]
        try:
            sbp_idxs = find_peaks(y, scale=200)
            if len(sbp_idxs) > 0:
                sbp_idxs_del = []
                for i in range(len(sbp_idxs)):
                    if y[i] < 30:
                        sbp_idxs_del.append(i)
                sbp_idxs = np.delete(sbp_idxs, sbp_idxs_del)
        except:
            sbp_idxs = np.array([y.argmax()])
        if len(sbp_idxs) == 0:
            sbp_idxs = np.array([y.argmax()])
        
        try:
            dbp_idxs = find_peaks(-y, scale=200)
            if len(dbp_idxs) > 0:
                dbp_idxs_del = []
                for i in range(len(dbp_idxs)):
                    if y[i] < 30:
                        dbp_idxs_del.append(i)
                dbp_idxs = np.delete(dbp_idxs, dbp_idxs_del)
        except:
            dbp_idxs = np.array([y.argmin()])
        if len(dbp_idxs) == 0:
            dbp_idxs = np.array([y.argmin()])
            
        y = torch.tensor([np.median(y[sbp_idxs]), np.median(y[dbp_idxs])], dtype=torch.float32).to(self.opt.device)
        return X, y

class zhang21_dataloader(Dataset):
    # load vitaldb dataset from https://iopscience.iop.org/article/10.1088/1361-6579/abf889/pdf
    # vitaldb dataset only has 8s PPG signals that also have ABP signals
    # no 8s segment that has all PPG, ECG and ABP signals
    # use 125Hz instead of 500Hz (dataset is large). 125Hz compatible with algorithms
    def __init__(self, fpath, idxs, opt):
        self.fpath = fpath
        self.opt = opt
        self.idxs = idxs
        self.dset = h5py.File(fpath, 'r')
        self.window_size = 256

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # call opt = opt() beforehand
        # returns [1, signal length, 1]
        d = self.dset[self.idxs[idx]]
        n = np.random.randint(0, d.shape[0])
        X = (d[n, 0, :] - d[n, 0, :].min())/(d[n, 0, :].max() - d[n, 0, :].min())
        X = X*2-1
        sos = butter(N=6, Wn=[0.5, 12], btype='bandpass', fs=125, output='sos')
        X = sosfiltfilt(sos, X)
        X = torch.tensor(X.copy(), dtype=torch.float32)

        randseg_idx = np.random.randint(0, X.shape[1]-self.window_size) 
        X = X[:, randseg_idx:randseg_idx+self.window_size]
        y = d[n, 1, randseg_idx:randseg_idx+self.window_size]
        try:
            sbp_idxs = find_peaks(y, scale=200)
            if len(sbp_idxs) > 0:
                sbp_idxs_del = []
                for i in range(len(sbp_idxs)):
                    if y[i] < 30:
                        sbp_idxs_del.append(i)
                sbp_idxs = np.delete(sbp_idxs, sbp_idxs_del)
        except:
            sbp_idxs = np.array([y.argmax()])
        if len(sbp_idxs) == 0:
            sbp_idxs = np.array([y.argmax()])
        
        try:
            dbp_idxs = find_peaks(-y, scale=200)
            if len(dbp_idxs) > 0:
                dbp_idxs_del = []
                for i in range(len(dbp_idxs)):
                    if y[i] < 30:
                        dbp_idxs_del.append(i)
                dbp_idxs = np.delete(dbp_idxs, dbp_idxs_del)
        except:
            dbp_idxs = np.array([y.argmin()])
        if len(dbp_idxs) == 0:
            dbp_idxs = np.array([y.argmin()])
            
        y = torch.tensor([np.median(y[sbp_idxs]), np.median(y[dbp_idxs])], dtype=torch.float32).to(self.opt.device)
        return X1, y

class huang22_dataloader(Dataset):
    # data loader for MLP-Mixer
    # note: dataset is not peak-aligned
    
    # Index Signal Filtering operations (parameters)
    # 1 ECG Bandpass filter (0.01–40 Hz)
    # 2 ECG Bandpass filter (0.01–20 Hz)
    # 3 ECG Low frequency filter (0.01–0.3 Hz, 2 order)
    # 4 ECG Low frequency filter (0.01–0.5 Hz, 2 order)
    # 5 ECG Lowpass Nyquist filter (0.9 Hz)
    # 6 ECG Savgol filter (window = 3, ployorder = 3)
    # 7 ECG Savgol filter (window = 3, ployorder = 1)
    # 8 ECG ChebyshevII (order = 6, rs = 2) UNKNOWN PARAMETERS
    # 9 PPG Bandpass filter (0.5–20 Hz)
    # 10 PPG Low frequency filter (0.01–0.3 Hz, 2 order)
    # 11 PPG Savgol filter (window = 3, ployorder = 3)
    # 12 PPG ChebyshevII (order = 6, rs = 10) 
    
    def __init__(self, fpath, idxs, opt):
        self.fpath = fpath
        self.opt = opt
        self.idxs = idxs
        self.dset = h5py.File(fpath, 'r')
        self.window_size = 128

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        # call opt = opt() beforehand
        # returns [1, signal length, 1]
        d = self.dset[self.idxs[idx]]
#         print(d.shape)
        n = np.random.randint(0, d.shape[0])
        PPG = d[n, 0, :]
        ECG = d[n, 1, :]
        
        # 1 ECG Bandpass filter (0.01–40 Hz)
        sos = butter(N=10, Wn=[0.01, 40], btype='bandpass', fs=125, output='sos')
        X1 = sosfilt(sos, ECG)
        # 2 ECG Bandpass filter (0.01–20 Hz)
        sos = butter(N=10, Wn=[0.01, 20], btype='bandpass', fs=125, output='sos')
        X2 = sosfilt(sos, ECG)
        # 3 ECG Low frequency filter (0.01–0.3 Hz, 2 order)
        sos = butter(N=2, Wn=[0.01, 0.3], btype='bandpass', fs=125, output='sos')
        X3 = sosfilt(sos, ECG)
        # 4 ECG Low frequency filter (0.01–0.5 Hz, 2 order)
        sos = butter(N=2, Wn=[0.01, 0.5], btype='bandpass', fs=125, output='sos')
        X4 = sosfilt(sos, ECG)
        # 5 ECG Lowpass Nyquist filter (0.9 Hz) UNKNOWN PARAMETERS
        taps = firwin(numtaps=3, cutoff=0.9)
        X5 = lfilter(taps, 1.0, ECG)  
        # 6 ECG Savgol filter (window = 3, ployorder = 3), CANNOT HAVE WINDOW=POLYORDER, assigned polyorder=2
        X6 = savgol_filter(ECG, window_length=3, polyorder=2)
        # 7 ECG Savgol filter (window = 3, ployorder = 1)
        X7 = savgol_filter(ECG, window_length=3, polyorder=1)
        # 8 ECG ChebyshevII (order = 6, rs = 2) UNKNOWN PARAMETERS
        sos = cheby2(N=6, rs=2, Wn=[0.1, 0.8], btype='bandpass', fs=125, output='sos')
        X8 = sosfilt(sos, ECG.flatten())
        # 9 PPG Bandpass filter (0.5–20 Hz)
        sos = butter(N=10, Wn=[0.5, 20], btype='bandpass', fs=125, output='sos')
        X9 = sosfilt(sos, PPG)
        # 10 PPG Low frequency filter (0.01–0.3 Hz, 2 order)
        sos = butter(N=10, Wn=[0.01, 0.3], btype='bandpass', fs=125, output='sos')
        X10 = sosfilt(sos, PPG)
        # 11 PPG Savgol filter (window = 3, ployorder = 3), CANNOT HAVE WINDOW=POLYORDER, assigned polyorder=2
        X11 = savgol_filter(PPG, window_length=3, polyorder=2)
        # 12 PPG ChebyshevII (order = 6, rs = 10) UNKNOWN PARAMETERS
        sos = cheby2(N=6, rs=10, Wn=[0.1, 0.8], btype='bandpass', fs=125, output='sos')
        X12 = sosfilt(sos, PPG)
        
        X = []
        for x in [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12]:
            X.append(minmax_scaler_1m1(x))
        X = np.array(X)
        
        window_size = 128
        randseg_idx = np.random.randint(0, X.shape[1]-self.window_size) 
        X = X[:, randseg_idx:randseg_idx+self.window_size]
        y = d[n, 2, randseg_idx:randseg_idx+self.window_size]
        try:
            sbp_idxs = find_peaks(y, scale=200)
            if len(sbp_idxs) > 0:
                sbp_idxs_del = []
                for i in range(len(sbp_idxs)):
                    if y[i] < 30:
                        sbp_idxs_del.append(i)
                sbp_idxs = np.delete(sbp_idxs, sbp_idxs_del)
        except:
            sbp_idxs = np.array([y.argmax()])
        if len(sbp_idxs) == 0:
            sbp_idxs = np.array([y.argmax()])
        
        try:
            dbp_idxs = find_peaks(-y, scale=200)
            if len(dbp_idxs) > 0:
                dbp_idxs_del = []
                for i in range(len(dbp_idxs)):
                    if y[i] < 30:
                        dbp_idxs_del.append(i)
                dbp_idxs = np.delete(dbp_idxs, dbp_idxs_del)
        except:
            dbp_idxs = np.array([y.argmin()])
        if len(dbp_idxs) == 0:
            dbp_idxs = np.array([y.argmin()])
            
        y = torch.tensor([np.median(y[sbp_idxs]), np.median(y[dbp_idxs])], dtype=torch.float32).to(self.opt.device)
        
        return X, y
    
# MISC functions
def minmax_scaler_1m1(x):
    x = (x - x.min())/(x.max() - x.min())
    x = x*2-1
    return x

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled