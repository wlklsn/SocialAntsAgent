import pickle
import numpy as np
import matplotlib.pyplot as plt # type: ignore

# Define the file paths (make sure these match the paths used in the original script)
brains_file = "simulations/test/Test3_6-28brains.pkl"
scores_file = "simulations/test/Test3_6-28scores.pkl"

# Load the data
with open(brains_file, 'rb') as file:
    all_brains = pickle.load(file)

with open(scores_file, 'rb') as file2:
    all_scores = pickle.load(file2)

# Extract the best scores
best_scores = np.max(all_scores, axis=1)

# Plot the best scores over generations
plt.figure(figsize=(10, 6))
plt.plot(best_scores, marker='o', linestyle='-', color='b')
plt.title('Best Agent Score per Generation')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.grid(True)
plt.show()

""" # Optionally, visualize the parameters of the best brains
# This part can be customized based on what you want to visualize
# For example, plotting the weights of the best brain in the last generation
best_brain = all_brains[-1]
weights_ih, weights_ho, bias_h, bias_o = best_brain

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title('Input-Hidden Weights')
plt.imshow(weights_ih, aspect='auto', cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title('Hidden-Output Weights')
plt.imshow(weights_ho, aspect='auto', cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title('Hidden Layer Bias')
plt.imshow(bias_h, aspect='auto', cmap='viridis')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title('Output Layer Bias')
plt.imshow(bias_o, aspect='auto', cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show() """
