import os
import hmp
import numpy as np
import re

def get_files(path):
    """
    function that returns arrays of .fif, .npy, .nc (with or without true_model classification) files found in a given folder

    Parameters
    ----------
    folder (string): path to the folder where the files are found

    Returns
    -------
    returns (fif_files, npy_files, nc_files, true_fif, true_npy)
    """
    #set path
    folder = os.listdir(path)

    #append files to array
    fif_files = []
    npy_files = []
    nc_files = []
    true_fif = []
    true_npy = []

    for file in folder:
        if file.startswith("true"):
            if file.endswith("fif"):
                true_fif.append(path + file)

            elif file.endswith("npy"):
                true_npy.append(path + file)
        else:
            if file.endswith("fif"):
                fif_files.append(path + file)
            elif file.endswith("npy"):
                npy_files.append(path + file) 
            elif file.endswith("nc"):
                nc_files.append(path + file) 

    return(fif_files, npy_files, nc_files, true_fif, true_npy)


def epoch_data(fif_files, npy_files):
    """
    function that creates epoched data from .fif and .npy files

    Parameters
    ----------
    fif_files (string[]): an array that contains all the names of the .fif files 
    npy_files (string[]): an array that contains all the names of the .npy files 

    returns
    -------
    epoch_data (xarray): the epoched data
    events_matrix (ndarray[]): array with all the event matrices
    source_times (float[]): array with all the source times
    """

    epoch_data = []
    events_matrix = []
    source_times = []

    #read files as EEG data
    for i in range(len(fif_files)):

        #npy
        events = np.load(npy_files[i])
        resp_trigger = int(np.max(np.unique(events[:,2])))#Resp trigger is the last source in each trial
        event_id = {'stimulus':1}#trigger 1 = stimulus
        resp_id = {'response':resp_trigger}
        

        #fif
        eeg_data = hmp.utils.read_mne_data(fif_files[i], event_id=event_id, resp_id=resp_id, sfreq=500, 
                events_provided=events, verbose=False)
        
        #source_times
        n_trials = len(eeg_data["epochs"])
        source_time = np.mean(np.reshape(np.diff(events[:,0], prepend=0),(n_trials,5+1))[:,1:]) #By-trial generated event times

        epoch_data.append(eeg_data)
        events_matrix.append(events)
        source_times.append(source_time)

    return(epoch_data, events_matrix, source_times)

def fit_model(epoch_data, n_comp, method, filename, save=None, true_model=False):
    """
    function that fits an HMP model

    Parameters
    ----------
    epoch_data (xarray): the epoched data
    n_comp (int): number of PCs/CCs to be extracted
    method(string): method used to extract the projections
    filename(name): name of the file it will be saved as
    save(bool): if true will save the HMP-model
    true_model(bool): if true will give the file the true_model label

    returns
    -------
    Selected (hmp): the predicted events of the HMP-model
    Init (hmp): the initialzation of the HMP-model
    """

    print("Initializating a hmp model using " + str(method) + "...")

    hmp_dat = hmp.utils.transform_data(epoch_data, apply_standard=False, n_comp=n_comp, method=method)

    #Initialization of the model
    init = hmp.models.hmp(hmp_dat, sfreq=epoch_data.sfreq, event_width=50, cpus=1)

    #Fitting
    print("fitting...")
    if not true_model:
        selected = init.fit() #function to fit an instance of a x events model
    else:
        selected = init.fit_single(4) #function to fit an instance of a 4 events model
    

    if save == 'yes':
        hmp.utils.save_fit(selected, filename + '.nc')

    return selected, init

def get_meta(txt_file):
    """
    function that reads analysis files and translates them into dictionary objects

    Parameters
    ----------
    txt_file (string): the name of the analysis file

    returns
    -------
    meta (dict): all the info of the analysis file stored into a dictionary
    """

    f = open(txt_file, "r")
    string = f.read().replace(":", "")
    list = re.split(r'[ \n]', string)
    keys = []
    values = []

    for i, item in enumerate(list):
        if i%2 == 0:
            keys.append(item)
        else:
            values.append(float(item))
    
    meta = dict(zip(keys, values))

    return meta