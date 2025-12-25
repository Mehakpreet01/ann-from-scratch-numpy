import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. INITIAL SETUP
# ===============================
np.random.seed(1)

input_nodes = 2
hidden_nodes = 3
learning_rate = 0.1
epochs = 200

# Weights & Bias initialization
weights = np.random.randn(input_nodes, hidden_nodes)
biases = np.zeros((1, hidden_nodes))

# ===============================
# 2. ACTIVATION FUNCTION
# ===============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ===============================
# 3. INPUT & TARGET DATA
# ===============================
inputs = np.array([0.5, -0.2])
target = np.array([1, 0, 1])

# ===============================
# 4. TRAINING LOOP
# ===============================
loss_history = []
output_history = []

print("\n--- Training Started ---")

for epoch in range(epochs):

    # ---- Forward Propagation ----
    layer_input = np.dot(inputs, weights) + biases
    output = sigmoid(layer_input)

    # ---- Loss (Mean Squared Error) ----
    loss = np.mean((output - target) ** 2)
    loss_history.append(loss)
    output_history.append(output.copy())

    # ---- Backpropagation ----
    error = output - target
    d_output = error * sigmoid_derivative(output)

    d_weights = np.dot(inputs.reshape(-1, 1),
                        d_output.reshape(1, -1))

    # ---- Update Weights ----
    weights -= learning_rate * d_weights

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

print("\n--- Training Completed ---")
print("Final Output:", output)
print("Target:", target)

# ===============================
# 5. VISUALIZATIONS
# ===============================

# ---- Loss Curve ----
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# ---- Output vs Target ----
plt.figure(figsize=(8, 5))
plt.plot(output, marker='o', label="Model Output")
plt.plot(target, marker='s', label="Target Output")
plt.xlabel("Neuron Index")
plt.ylabel("Value")
plt.title("Model Output vs Target")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 6. TESTING ON NEW INPUT
# ===============================
print("\n--- Testing Phase ---")

test_input = np.array([0.4, -0.1])
test_output = sigmoid(np.dot(test_input, weights) + biases)

print("Test Input:", test_input)
print("Prediction:", test_output)
