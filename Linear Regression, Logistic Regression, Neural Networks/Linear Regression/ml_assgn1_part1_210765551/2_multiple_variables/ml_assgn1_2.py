from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.4
iterations = 100

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)
print("Theta Final ->",theta_final)

#########################################
# Write your code here
# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples

X_test = np.array([[1650, 3],[3000, 4]]) #Created the test array
X_test_normalized = np.zeros(X_test.shape) #stored zero array of same shape as test set
for i in range(len(X_test_normalized)):
    for j in range(len(X_test_normalized)):
        X_test_normalized[i][j] = ( X_test[i][j] - mean_vec[0][j] ) / std_vec[0][j]    #normalized the test set

column_of_ones = np.ones((X_test_normalized.shape[0], 1))
X_test_normalized = np.append(column_of_ones, X_test_normalized, axis=1)

y_predicted = np.zeros(X_test[0].shape)
for i in range(len(X_test)):
    y_predicted[i] = calculate_hypothesis(X_test_normalized, theta_final, i)

print("House 1 Predicted Price = ",y_predicted[0])
print("House 2 Predicted Price = ",y_predicted[1])
########################################/
