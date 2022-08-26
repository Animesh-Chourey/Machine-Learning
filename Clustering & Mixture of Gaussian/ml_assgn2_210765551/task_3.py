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

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"


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
prediction_p1_k3 = get_predictions(mu_p1_k3, s_p1_k3, p_p1_k3, X_phonemes_1_2)
prediction_p1_k6 = get_predictions(mu_p1_k6, s_p1_k6, p_p1_k6, X_phonemes_1_2)
prediction_p2_k3 = get_predictions(mu_p2_k3, s_p2_k3, p_p2_k3, X_phonemes_1_2)
prediction_p2_k6 = get_predictions(mu_p2_k6, s_p2_k6, p_p2_k6, X_phonemes_1_2)

# Calculate the sum of the probabilities of the gaussians belonging to corresponding phoneme
sum_p1_k3 = np.sum(prediction_p1_k3, axis=1)
sum_p1_k6 = np.sum(prediction_p1_k6, axis=1)
sum_p2_k3 = np.sum(prediction_p2_k3, axis=1)
sum_p2_k6 = np.sum(prediction_p2_k6, axis=1)

# Calculate the highest probability among two different gaussian sets

# Calculate whether the data point is of phoneme 1 or phoneme 2 in the gaussian set of 3
prediction_k3 = np.zeros(len(X_phonemes_1_2))
for i in range(len(prediction_k3)):
    if sum_p1_k3[i] > sum_p2_k3[i]:
        prediction_k3[i] = 1
    else:
        prediction_k3[i] = 2

# Calculate whether the data point is of phoneme 1 or phoneme 2 in the gaussian set of 6
prediction_k6 = np.zeros(len(X_phonemes_1_2))
for i in range(len(prediction_k6)):
    if sum_p1_k6[i] > sum_p2_k6[i]:
        prediction_k6[i] = 1
    else:
        prediction_k6[i] = 2

# Storing the ground truth phoneme for each data point
ground_truth = np.zeros(len(X_phonemes_1_2))
for i in range(len(X_phoneme_1)): # From range(0,152) value 1 will be stored
    ground_truth[i] = 1
for i in range(len(X_phoneme_1),len(X_phonemes_1_2)): # From range(152,304) value 2 will be stored
    ground_truth[i] = 2

# Calculating the correct values predicted for 3 gaussian clusters
total_correct_prediction_k3 = 0
for i in range(len(prediction_k3)):
    if prediction_k3[i] == ground_truth[i]: #Comparing the predicted value with ground truth
        total_correct_prediction_k3 = total_correct_prediction_k3 + 1

# Calculating the correct values predicted for 6 gaussian clusters
total_correct_prediction_k6 = 0
for i in range(0, len(prediction_k6)):
    if prediction_k6[i] == ground_truth[i]: #Comparing the predicted value with ground truth
        total_correct_prediction_k6 = total_correct_prediction_k6 + 1

# Calculating the accuracy
accuracy_k3 = total_correct_prediction_k3/len(prediction_k3) * 100
accuracy_k6 = total_correct_prediction_k6/len(prediction_k6) * 100

########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(3, accuracy_k3))
print('Accuracy using GMMs with {} components: {:.2f}%'.format(6, accuracy_k6))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()