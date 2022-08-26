import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0] = f1
X_full[:,1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# X_phonemes_1_2 = ...
X_phoneme_1 = np.zeros((np.sum(phoneme_id==1), 2))
X_phoneme_2 = np.zeros((np.sum(phoneme_id==2), 2))

#Getting the data for phoneme 1
element_index_phoneme1 = []
for i in range(len(phoneme_id)): #extract the indexes that have the phoneme_id as 1
    if phoneme_id[i] == 1:
        element_index_phoneme1.append(i)

for i in range(X_phoneme_1.shape[0]): 
    index = element_index_phoneme1[i]    #Store the index 
    X_phoneme_1[i] = X_full[index]


#Getting the data for phoneme 2
element_index_phoneme2 = []
for i in range(len(phoneme_id)): #extract the indexes that have the phoneme_id as 2
    if phoneme_id[i] == 2:
        element_index_phoneme2.append(i)

for i in range(X_phoneme_2.shape[0]): 
    index = element_index_phoneme2[i]    #Store the index 
    X_phoneme_2[i] = X_full[index]

#Stacking X_phoneme_1 and X_phoneme_2
X_phonemes_1_2 = np.vstack((X_phoneme_1, X_phoneme_2))
########################################/

# as dataset X, we will use only the samples of phoneme 1 and 2
X = X_phonemes_1_2.copy()

min_f1 = int(np.min(X[:,0]))
max_f1 = int(np.max(X[:,0]))
min_f2 = int(np.min(X[:,1]))
max_f2 = int(np.max(X[:,1]))
N_f1 = max_f1 - min_f1
N_f2 = max_f2 - min_f2
print('f1 range: {}-{} | {} points'.format(min_f1, max_f1, N_f1))
print('f2 range: {}-{} | {} points'.format(min_f2, max_f2, N_f2))

#########################################
# Write your code here

# Create a custom grid of shape N_f1 x N_f2
# The grid will span all the values of (f1, f2) pairs, between [min_f1, max_f1] on f1 axis, and between [min_f2, max_f2] on f2 axis
# Then, classify each point [i.e., each (f1, f2) pair] of that grid, to either phoneme 1, or phoneme 2, using the two trained GMMs
# Do predictions, using GMM trained on phoneme 1, on custom grid
# Do predictions, using GMM trained on phoneme 2, on custom grid
# Compare these predictions, to classify each point of the grid
# Store these prediction in a 2D numpy array named "M", of shape N_f2 x N_f1 (the first dimension is f2 so that we keep f2 in the vertical axis of the plot)
# M should contain "0.0" in the points that belong to phoneme 1 and "1.0" in the points that belong to phoneme 2
########################################/

#Initialize custom_grid array
custom_grid = np.zeros((N_f1, N_f2, 2))

#Get the range for f1 and f2
f1_range_values = range(min_f1, max_f1)
f2_range_values = range(min_f2, max_f2)

#Store the custom_grid with the data points between the range
#Each element in f1_range_values will be grouped with all the elements in f2_range_values one by one
for i, v1 in enumerate(f1_range_values):
    for j, v2 in enumerate(f2_range_values):
        custom_grid[i][j] = [v1, v2] 

# File that contains the phoneme 1 and k=3
# Loading data from .npy file
data2 = np.load('data/GMM_params_phoneme_01_k_03.npy', allow_pickle=True)
data2 = np.ndarray.tolist(data2) #data is stored in the from of dictionary
#Extracting the values from the corresponding keys
mu_p1_k3 = data2['mu']
s_p1_k3 = data2['s']
p_p1_k3 = data2['p']

# File that contains the phoneme 1 and k=6
# Loading data from .npy file
data3 = np.load('data/GMM_params_phoneme_01_k_06.npy', allow_pickle=True)
data3 = np.ndarray.tolist(data3) #data is stored in the from of dictionary
#Extracting the values from the corresponding keys
mu_p1_k6 = data3['mu']
s_p1_k6 = data3['s']
p_p1_k6 = data3['p']

# File that contains the phoneme 2 and k=3
# Loading data from .npy file
data4 = np.load('data/GMM_params_phoneme_02_k_03.npy', allow_pickle=True)
data4 = np.ndarray.tolist(data4) #data is stored in the from of dictionary
#Extracting the values from the corresponding keys
mu_p2_k3 = data4['mu']
s_p2_k3 = data4['s']
p_p2_k3 = data4['p']

# File that contains the phoneme 2 and k=6
# Loading data from .npy file
data5 = np.load('data/GMM_params_phoneme_02_k_06.npy', allow_pickle=True)
data5 = np.ndarray.tolist(data5) #data is stored in the from of dictionary
#Extracting the values from the corresponding keys
mu_p2_k6 = data5['mu']
s_p2_k6 = data5['s']
p_p2_k6 = data5['p']


# Get the predictions in the form of probabilities based on the gaussians created for each phoneme 
# As well as, Calculate the sum of the probabilities of the gaussians belonging to corresponding phoneme

#Calculating the sum for phoneme 1 for k=3
sum_p1_k3 = []
for i in range(N_f2): #for all the 1900 rows
    prediction_p1_k3 = get_predictions(mu_p1_k3, s_p1_k3, p_p1_k3, custom_grid[:, i]) #All values present at the i-th row of each array 
    sum_p1_k3.append(np.sum(prediction_p1_k3, axis=1)) #Summing the probabilties
sum_p1_k3 = np.array(sum_p1_k3) 

#Calculating the sum for phoneme 2 for k=3
sum_p2_k3 = []
for i in range(N_f2): #for all 1900 arrays
    prediction_p2_k3 = get_predictions(mu_p2_k3, s_p2_k3, p_p2_k3, custom_grid[:, i]) #All values present at the i-th row of each array
    sum_p2_k3.append(np.sum(prediction_p2_k3, axis=1))  #Summing the probabilties
sum_p2_k3 = np.array(sum_p2_k3)

#Calculating the sum for phoneme 1 for k=6
sum_p1_k6 = []
for i in range(N_f2): #for all the 1900 rows
    prediction_p1_k6 = get_predictions(mu_p1_k6, s_p1_k6, p_p1_k6, custom_grid[:, i]) #All values present at the i-th row of each array 
    sum_p1_k6.append(np.sum(prediction_p1_k6, axis=1)) #Summing the probabilties
sum_p1_k6 = np.array(sum_p1_k6)

#Calculating the sum for phoneme 2 for k=6
sum_p2_k6 = []
for i in range(N_f2): #for all 1900 arrays
    prediction_p2_k6 = get_predictions(mu_p2_k6, s_p2_k6, p_p2_k6, custom_grid[:, i]) #All values present at the i-th row of each array
    sum_p2_k6.append(np.sum(prediction_p2_k6, axis=1))  #Summing the probabilties
sum_p2_k6 = np.array(sum_p2_k6)

#Create the classification matrix M
M = np.ndarray((N_f2, N_f1))
'''
#Compare the predictions for the 3 gaussian clusters
for i in range(len(sum_p1_k3)):
    for j in range(len(sum_p1_k3[0])):
        if sum_p1_k3[i][j] > sum_p2_k3[i][j]: #Compare the probability of which phoneme is higher
            M[i][j] = 0  
        else:
            M[i][j] = 1
'''
#Compare the predictions for the 6 gaussian clusters
for i in range(len(sum_p1_k6)):
    for j in range(len(sum_p1_k6[0])):
        if sum_p1_k6[i][j] > sum_p2_k6[i][j]: #Compare the probability of which phoneme is higher
            M[i][j] = 0  
        else:
            M[i][j] = 1


#Printing the classification matrix
print("M: \n", M)

################################################
# Visualize predictions on custom grid

# Create a figure
#fig = plt.figure()
fig, ax = plt.subplots()

# use aspect='auto' (default is 'equal'), to force the plotted image to be square, when dimensions are unequal
plt.imshow(M, aspect='auto')

# set label of x axis
ax.set_xlabel('f1')
# set label of y axis
ax.set_ylabel('f2')

# set limits of axes
plt.xlim((0, N_f1))
plt.ylim((0, N_f2))

# set range and strings of ticks on axes
x_range = np.arange(0, N_f1, step=50)
x_strings = [str(x+min_f1) for x in x_range]
plt.xticks(x_range, x_strings)
y_range = np.arange(0, N_f2, step=200)
y_strings = [str(y+min_f2) for y in y_range]
plt.yticks(y_range, y_strings)

# set title of figure
title_string = 'Predictions on custom grid'
plt.title(title_string)

# add a colorbar
plt.colorbar()

N_samples = int(X.shape[0]/2)
plt.scatter(X[:N_samples, 0] - min_f1, X[:N_samples, 1] - min_f2, marker='.', color='red', label='Phoneme 1')
plt.scatter(X[N_samples:, 0] - min_f1, X[N_samples:, 1] - min_f2, marker='.', color='green', label='Phoneme 2')

# add legend to the subplot
plt.legend()

# save the plotted points of the chosen phoneme, as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'GMM_predictions_on_grid.png')
plt.savefig(plot_filename)

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()