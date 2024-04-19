This is the code that is used for the simulations in "Does PCA simplify EEG data?" by van Tilburg (2024). Before running anything note that I am using an adjusted version of HMP. Make sure this is imported instead of any other version of the HMP package. Also most of the code imports functions from General_Functions.py so make sure that this is in the same folder as the code you are using. Lastly, the path for the folders is currently set to the path on the device where the original simulations ran on. Make sure to change these to a new path if you want to use it for yourself (The variables that need to be changed are commented with "CHANGE THIS" in the code).

OVERVIEW


Generate_Data.py

This script was used to generate .fif and .npy files by using the simulation function from HMP. When this script has finished its run it will store these files in the same folder as were the code is stored. Make sure to put these files in a folder in order to make them detectable for the other scripts.

Read_Data.py

This script was used to visualize the raw data. It reads in all .fif and .npy of a given folder and outputs a plot of the electrodes and a plot of the EEG signals for the first ten participants. Note that every participant needs one .fif file and one .npy file.

Fit_Data.py

This script was used to create HMP models for a dataset. It reads in all .fif and .npy files in a given folder and outputs a plot of the PCA model, the M-CCA model and the true model. Note that the folder where this script is called on contains at least one .fif file and one .npy file that starts with true in order to be able to train a true model.

This script also analyses the models by giving the TPR and PPV of the models. These can be saved the same way Generate_Data.py stores the data. Make sure that these are then put in a folder the same way for Plot_Results.py to work.

Plot_Results.py

This script was used to visualize the model analysis done in Fit_Data.py. It reads in all .txt files in a folder and uses them to make three barplots:

- a TPR/PPV analysis for all snippets.
- a TPR/PPV analysis for the mean of all snippets.
- an analysis of the number of events predicted.

General_Functions.py

This script contains all returning functions that are used by the other scripts.