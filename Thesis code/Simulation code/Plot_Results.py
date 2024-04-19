import os
import General_Functions as general
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import math

#0. initialization
folder = input("From which folder do you want to extract data? ")
path  = "/Users/Arian van Tilburg/Documents/KI/Scriptie/Code/" + folder + "/" #CHANGE THIS to the folder where the datasets are to be found
save = input("Do you want to save the results? [yes/no] ")

#1. recovering TPR and PPV
TPR_PCA = []
TPR_MCCA = []
PPV_PCA = []
PPV_MCCA = []
N_events_PCA = []
N_events_MCCA = []


for file in os.listdir(path):
    info = general.get_meta(path + file)
    TPR_PCA.append(info['TPR_PCA'])
    TPR_MCCA.append(info['TPR_MCCA'])
    PPV_PCA.append(info['PPV_PCA'])
    PPV_MCCA.append(info['PPV_MCCA'])
    N_events_PCA.append(info['N_events_PCA'])
    N_events_MCCA.append(info['N_events_MCCA'])

#calculating the standard error
SE_TPR_PCA = math.sqrt(stat.variance(TPR_PCA))/math.sqrt(len(TPR_PCA))
SE_TPR_MCCA = math.sqrt(stat.variance(TPR_MCCA))/math.sqrt(len(TPR_MCCA))
SE_PPV_PCA = math.sqrt(stat.variance(PPV_PCA))/math.sqrt(len(PPV_PCA))
SE_PPV_MCCA = math.sqrt(stat.variance(PPV_MCCA))/math.sqrt(len(PPV_MCCA))
SE_N_events_PCA = math.sqrt(stat.variance(N_events_PCA))/math.sqrt(len(N_events_PCA))
SE_N_events_MCCA = math.sqrt(stat.variance(N_events_MCCA))/math.sqrt(len(N_events_MCCA))

#saving the results in a .txt file
if save == 'yes':
    f = open(folder + "_results.txt", "w+")
    f.write("SE_TPR_PCA: " + str(SE_TPR_PCA))
    f.write("\n")
    f.write("SE_TPR_MCCA: " + str(SE_TPR_MCCA))
    f.write("\n")
    f.write("SE_PPV_PCA: " + str(SE_PPV_PCA))
    f.write("\n")
    f.write("SE_PPV_MCCA: " + str(SE_PPV_MCCA))
    f.write("\n")
    f.write("SE_N_events_PCA: " + str(SE_N_events_PCA))
    f.write("\n")
    f.write("SE_N_events_MCCA: " + str(SE_N_events_MCCA))
    f.write("\n")
    f.write("\n")
    f.write('TPR PCA mean: ' + str(round(np.mean(TPR_PCA), 3)))
    f.write("\n")
    f.write('TPR PCA SD: ' + str(round(stat.variance(TPR_PCA), 3)))
    f.write("\n")
    f.write('TPR MCCA mean: ' + str(round(np.mean(TPR_MCCA), 3)))
    f.write("\n")
    f.write('TPR MCCA SD: ' + str(round(stat.variance(TPR_MCCA), 3)))
    f.write("\n")
    f.write('PPV PCA mean: ' + str(round(np.mean(PPV_PCA), 3)))
    f.write("\n")
    f.write('PPV PCA SD: ' + str(round(stat.variance(PPV_PCA), 3)))
    f.write("\n")
    f.write('PPV MCCA mean: ' + str(round(np.mean(PPV_MCCA), 3)))
    f.write("\n")
    f.write('PPV MCCA SD: ' + str(round(stat.variance(PPV_MCCA), 3)))
    f.write("\n")
    f.write('N_events_PCA mean: ' + str(round(np.mean(N_events_PCA), 3)))
    f.write("\n")
    f.write('N_events_PCA SD: ' + str(round(stat.variance(N_events_PCA), 3)))
    f.write("\n")
    f.write('N_events_MCCA mean: ' + str(round(np.mean(N_events_MCCA), 3)))
    f.write("\n")
    f.write('N_events_MCCA SD: ' + str(round(stat.variance(N_events_MCCA), 3)))
    f.close()

#2. visualization
labels = ["PCA", "MCCA"]
colors = ['tab:blue', 'tab:orange']

#2.1 single snippets
distance = 0.2
x1a = [x - distance + 1 for x in range(10)]
x1b = [x + distance + 1 for x in range(10)]

fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

for a in ax1:
    a.set_ylim(-0.1, 1.1)
    a.set_xlim(0.1, 10.9)


ax1[0].set_title('TPR', fontsize = 20)
ax1[1].set_title('PPV', fontsize = 20)

ax1[0].bar(x1a, TPR_PCA, label = labels[0], width = 0.3, color=colors[0])
ax1[0].bar(x1b, TPR_MCCA, label = labels[1], width = 0.3, color=colors[1])
ax1[1].bar(x1a, PPV_PCA, label = labels[0], width = 0.3, color=colors[0])
ax1[1].bar(x1b, PPV_MCCA, label = labels[1], width = 0.3, color=colors[1])

ax1[0].set_xlabel('Snippet', fontsize = 12)
ax1[0].set_ylabel('TPR', fontsize = 12)
ax1[1].set_xlabel('Snippet', fontsize = 12)
ax1[1].set_ylabel('PPV', fontsize = 12)

ax1[0].legend(loc='upper right')
ax1[1].legend(loc='upper right')

#2.2 means
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

for a in ax2:
    a.set_ylim(-0.1, 1.1)



ax2[0].set_title('TPR', fontsize = 20)
ax2[1].set_title('PPV', fontsize = 20)
ax2[0].set_ylabel('TPR', fontsize = 12)
ax2[1].set_ylabel('PPV', fontsize = 12)

ax2[0].bar(labels, [np.mean(TPR_PCA), np.mean(TPR_MCCA)], color=colors, yerr=[SE_TPR_PCA, SE_TPR_MCCA])
ax2[1].bar(labels, [np.mean(PPV_PCA), np.mean(PPV_MCCA)], color=colors, yerr=[SE_PPV_PCA, SE_PPV_MCCA])
ax2[0].set_xticklabels(labels, fontsize=12)
ax2[1].set_xticklabels(labels, fontsize=12)


#2.3 number of events
fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

ax3.set_title('N events', fontsize = 20)
ax3.set_ylabel('N events', fontsize = 12)

ax3.bar(labels, [np.mean(N_events_PCA), np.mean(N_events_MCCA)], color=colors, yerr=[SE_N_events_PCA, SE_N_events_MCCA])
ax3.set_xticklabels(labels, fontsize=12)

#show all plots
plt.show()
    
