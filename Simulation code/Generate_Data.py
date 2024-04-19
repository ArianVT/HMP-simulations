from hmp import simulations
import numpy as np
import scipy.stats

#Frequency of the event defining its duration, half-sine of 10Hz = 50ms
#Amplitude of the event in nAm, defining signal to noise ratio
#location, extent and source parameters according to the MNE package
adjustable_pars = {
    "frequency": 10.,
    "amplitude": 2.5e-7,
    "location" : 'center',
    "extent": 0.0,
    "source": [0, 0, 0]
}


def generate_participant(parameter, mu, sigma, filename, n_trials):
    """
    Function to generate a single participant given a parameter and a normal distribution

    parameters
    ---------
    parameter (string): the parameter that will be changed during the generation process 
    mu (float): the mean value for the normal distribution
    sigma (float): the standard deviation for the normal distribution
    filename (string): the name for the participant files
    n_trials (int): the number of trails that will be simulated 
    """

    #parameters for the simulation process (based on the tutorial from Weindel et al.)
    cpus = 2 # For multiprocessing, usually a good idea to use multiple CPUs as long as you have enough RAM
    sfreq = 500
    shape = 2 #shape of the gamma distribution
    means = np.array([60, 150, 200, 100, 80])/shape #Mean duration of the between event times in ms
    names = ['inferiortemporal-lh','caudalanteriorcingulate-rh','bankssts-lh','superiorparietal-lh','superiorparietal-lh'] #Which source to activate for each event (see atlas when calling simulations.available_sources())
    sources = []


    #pick adjustable parameter value from the given normal distribution
    if parameter == "location":
        #for location the value has to be fixed to an integer
        adjustable_pars[parameter] = int(scipy.stats.norm.rvs(loc=mu, scale=sigma, size=1)[0])
    elif parameter == "source":
        #for source three values are selected from the distribution
        adjustable_pars[parameter][0] = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=1)[0]
        adjustable_pars[parameter][1] = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=1)[0]
        adjustable_pars[parameter][2] = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=1)[0]
    elif parameter == "frequency":
        #for frequency durations have to be translated to frequencies
        duration = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=1)[0]
        print("duration " + str(duration))
        adjustable_pars[parameter] = 1000/duration/2
    else:
        #default method to select the value for the parameter
        adjustable_pars[parameter] = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=1)[0]
        
    print(parameter + " " + str(adjustable_pars[parameter]))


    #create sources
    for source in zip(names, means): #One source = one frequency, one amplitude and a given by-trial variability distribution
        sources.append([source[0], adjustable_pars["frequency"], adjustable_pars["amplitude"], scipy.stats.gamma(shape, scale=source[1])])


    #Generate the data with the given parameters
    simulations.simulate(sources, n_trials, cpus, filename, overwrite=False, sfreq=sfreq, seed=1234, a=adjustable_pars["location"], b=adjustable_pars["extent"], x=adjustable_pars["source"][0], y=adjustable_pars["source"][1], z=adjustable_pars["source"][2])


def generate_dataset(n_participants, parameter, mu, sigma, filename, n_trials):
    """
    Function that uses generate_participant() to generate a dataset of multiple participants

    Parameters
    ----------
    n_participants (int): number of participants that will be simulated
    parameter (string): the parameter that will be changed during the generation process 
    mu (float): the mean value for the normal distribution
    sigma (float): the standard deviation for the normal distribution
    filename (string): the name for the participant files
    n_trials (int): the number of trails that will be simulated 
    """

    for participant in range(n_participants):
        generate_participant(parameter, mu, sigma, filename + " " + str(participant), n_trials)


#call generate_dataset given the user input
n_participants = int(input("How many subjects do you want to simulate: "))
n_trials = int(input("How many trials do you want to simulate: "))
mu = float(input("What mu do you want to use for the normal distribution: "))
sigma = float(input("What sigma do you want to use for the normal distribution: "))
for i, key in enumerate(adjustable_pars.keys()):
    print(str(i) + " ", key)
parameter = input("Which of these parameters do you want to adjust: ")
filename = input("What do you want the files to be called: ")

generate_dataset(n_participants, parameter, mu, sigma, filename, n_trials)


