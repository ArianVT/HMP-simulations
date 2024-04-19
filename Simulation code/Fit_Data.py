import hmp
import numpy as np
import matplotlib.pyplot as plt
from hmp import simulations
import General_Functions as general
import multiprocessing as mp

#0. initialization
cpus = 4 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM
mp.set_start_method("spawn")
folder = input("From which folder do you want to extract data? ")
path  = "/Users/Arian van Tilburg/Documents/KI/Scriptie/Code/" + folder + "/" #this is fixed to the folder where the datasets are to be found
n_comp = int(input("How many principal components do you want to use? "))
use_pca = input("Do you want to train a pca model? [yes/no] ")
use_mcca = input("Do you want to train a mcca model? [yes/no] ")
save = input("Do you want to save the model(s)? [yes/no] ")
fif_files, npy_files, nc_files, true_fif, true_npy = general.get_files(path)

epoch_data, events_matrix, source_times = general.epoch_data(fif_files, npy_files) 
true_epoch_data, true_events, true_source_times = general.epoch_data(true_fif, true_npy) 

#set parameters (based on the tutorial from Weindel et al.)
events = events_matrix[0]
resp_trigger = int(np.max(np.unique(events[:,2])))#Resp trigger is the last source in each trial
event_id = {'stimulus':1}#trigger 1 = stimulus
resp_id = {'response':resp_trigger}
sfreq = 100 #at what sampling rate we want the data, downsampling to 250Hz just to show that we can use any SF
tmin, tmax = -.2, 2 #window size for the epochs, from 250ms before the stimulus up to 2 seconds after, data will be baseline corrected from tmin to 0
high_pass = 1 #High pass filtering to be applied, useful to avoid slow drifts, parameters of the filters are defined automatically by MNE
lower_limit_RT, upper_limit_RT = .2, 2 #lower and upper limit for the RTs all values outside of this range are discarded
info = simulations.simulation_positions() 

#epoch data
epoch_data = hmp.utils.read_mne_data(fif_files, event_id=event_id, resp_id=resp_id, epoched=False,
                            tmin=tmin, tmax=tmax, sfreq=sfreq, 
                            high_pass=high_pass, events_provided = events_matrix,
                            verbose=False)

true_epoch_data = hmp.utils.read_mne_data(true_fif, event_id=event_id, resp_id=resp_id, epoched=False,
                            tmin=tmin, tmax=tmax, sfreq=sfreq, 
                            high_pass=high_pass, events_provided = true_events,
                            verbose=False)
                        
#1. PCA
if use_pca == 'yes':
    selected_pca, init_pca = general.fit_model(epoch_data, n_comp, 'pca', folder + "_pca", save=save)
    true_estimates_pca, true_init_pca = general.fit_model(true_epoch_data, n_comp, 'pca', folder + "_true_estimates_pca", save=save, true_model=True)

#2. M-CCA
if use_mcca == 'yes':
    selected_mcca, init_mcca = general.fit_model(epoch_data, n_comp, 'mcca', folder + "_mcca", save=save)
    true_estimates_mcca, true_init_mcca = general.fit_model(true_epoch_data, n_comp, 'mcca', folder + "_true_estimates_mcca", save=save, true_model=True)

    


#3. getting TPR and PPV

#getting the topologies
test_topologies_pca = init_pca.compute_topologies(epoch_data, selected_pca, init_pca, mean=True)
true_topologies_pca = init_pca.compute_topologies(true_epoch_data, true_estimates_pca, true_init_pca, mean=True)
test_topologies_mcca = init_mcca.compute_topologies(epoch_data, selected_mcca, init_mcca, mean=True)
true_topologies_mcca = init_mcca.compute_topologies(true_epoch_data, true_estimates_mcca, true_init_mcca, mean=True)

#recovering the correct events and corresponding indexes
correct_event_capture_pca, corresponding_index_event_pca = simulations.classification_true(true_topologies_pca, test_topologies_pca)
correct_event_capture_mcca, corresponding_index_event_mcca = simulations.classification_true(true_topologies_mcca, test_topologies_mcca)

#calculating TPR and PPV
TP_PCA = len(correct_event_capture_pca)
TP_MCCA = len(correct_event_capture_mcca)
FN_PCA = len(true_estimates_pca['event']) - TP_PCA
FN_MCCA = len(true_estimates_mcca['event']) - TP_MCCA
FP_PCA = len(selected_pca['event']) - TP_PCA
FP_MCCA = len(selected_mcca['event']) - TP_MCCA

TPR_PCA = TP_PCA/(TP_PCA + FN_PCA)
TPR_MCCA = TP_MCCA/(TP_MCCA + FN_MCCA)
PPV_PCA = TP_PCA/(TP_PCA + FP_PCA)
PPV_MCCA = TP_MCCA/(TP_MCCA + FP_MCCA)

#print TPR and PPV
print("\n")
print("PCA")
print("N_events: " + str(len(selected_pca['event'])))
print("TPR: " + str(TPR_PCA))
print("PPV: " + str(PPV_PCA))
print("\n")
print("MCCA")
print("N_events: " + str(len(selected_mcca['event'])))
print("TPR: " + str(TPR_MCCA))
print("PPV: " + str(PPV_MCCA))

#save TPR and PPV in a .txt files
if save == 'yes':
    f = open(folder + "_analysis.txt", "w+")
    f.write("N_events_PCA: " + str(len(selected_pca['event'])))
    f.write("\n")
    f.write("TPR_PCA: " + str(TPR_PCA))
    f.write("\n")
    f.write("PPV_PCA: " + str(PPV_PCA))
    f.write("\n")
    f.write("N_events_MCCA: " + str(len(selected_mcca['event'])))
    f.write("\n")
    f.write("TPR_MCCA: " + str(TPR_MCCA))
    f.write("\n")
    f.write("PPV_MCCA: " + str(PPV_MCCA))
    f.close()

#4. visualizing 

#create subplots
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))
fig.suptitle(folder)
ax[0][0].set_title('PCA estimate')
ax[1][0].set_title('PCA true')
ax[0][1].set_title('MCCA estimate')
ax[1][1].set_title('MCCA true')

if use_pca == 'yes':
    #Visualizing PCA prediction
    hmp.visu.plot_topo_timecourse(epoch_data, selected_pca, info, init_pca, magnify=1, sensors=False,
                                times_to_display = np.mean(source_times), ax = ax[0][0], as_time = True)
    
    #Visualizing PCA true estimates
    hmp.visu.plot_topo_timecourse(true_epoch_data, true_estimates_pca, info, true_init_pca, magnify=1, sensors=False,
                                times_to_display = true_source_times, ax = ax[1][0], as_time = True)
    
if use_mcca == 'yes':
    #Visualizing MCCA prediction
    hmp.visu.plot_topo_timecourse(epoch_data, selected_mcca, info, init_mcca, magnify=1, sensors=False,
                                times_to_display = np.mean(source_times), ax = ax[0][1], as_time = True)
    
    #Visualizing PCA true estimates
    hmp.visu.plot_topo_timecourse(true_epoch_data, true_estimates_mcca, info, true_init_mcca, magnify=1, sensors=False,
                                times_to_display = np.mean(true_source_times), ax = ax[1][1], as_time = True)

#visualize entire plot
plt.show()
