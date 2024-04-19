import matplotlib.pyplot as plt
import mne
import math
import General_Functions as general

def visualize(path):
    """
    Function that plots...
    - the raw eeg signals for the first ten participants
    - the sensor locations of the first participant

    Parameters
    ----------
    path (string): folder location where the .fif and .npy files for the participants are stored
    """

    fif_files, npy_files, nc_files, true_fif, true_npy = general.get_files(path)
    epoch_data = general.epoch_data(fif_files, npy_files)[0]

    lines = []  
    
    #EEG/line
    for eeg_data in epoch_data:
        line = eeg_data.sel(channels=['EEG 001','EEG 002','EEG 003'], samples=range(212))\
        .data.groupby('samples').mean(['participant','epochs'])
        lines.append(line)

    #create subplots
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 5))

    #plot first ten participants
    for i in range(len(epoch_data)):
        if i < 10:
            lines[i].plot.line(
                hue='channels', ax = ax[math.floor(i/5)][i%5]
            )
    
    #plot sensors
    epoch = mne.io.read_raw_fif(fif_files[0])
    epoch.plot_sensors(show_names=True)
    plt.show()


#user input
path = "/Users/Arian van Tilburg/Documents/KI/Scriptie/Code/" + str(input("From which folder do you want to extract data? ")) + "/"
visualize(path)
