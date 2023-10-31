import numpy as np

# ฟังค์ชันอ่าน data
def read_file(filename):
    input_arrays = []
    output_arrays = []

    with open(filename, 'r') as file:
        for line in file:

            integers = list(map(int, line.strip().split()))

            if len(integers) >= 9:

                input_arrays.append(integers[:8])
                output_arrays.append([integers[-1]])
            else:
                print("Invalid data format in a line:", line)

    return input_arrays, output_arrays

# ฟังค์ชัน nomalize
def normalize_data(input_arrays, output_arrays):

    flattened_inputs = np.array(input_arrays).flatten()
    flattened_outputs = np.array(output_arrays).flatten()


    min_val = min(np.min(flattened_inputs), np.min(flattened_outputs))
    max_val = max(np.max(flattened_inputs), np.max(flattened_outputs))

    normalized_input_arrays = (np.array(input_arrays) - min_val) / (max_val - min_val)
    normalized_output_arrays = (np.array(output_arrays) - min_val) / (max_val - min_val)

    return normalized_input_arrays, normalized_output_arrays, min_val, max_val

# ฟังค์ชันสลับdata 
def shuffle_data(input_arrays, output_arrays):
    assert len(input_arrays) == len(output_arrays), "Input and output arrays must have the same length."

    indices = np.arange(len(input_arrays))
    np.random.shuffle(indices)

    shuffled_input_arrays = [input_arrays[i] for i in indices]
    shuffled_output_arrays = [output_arrays[i] for i in indices]

    return shuffled_input_arrays, shuffled_output_arrays

def denormalize_data(data, min_val, max_val):
    denormalized = (data * (max_val - min_val)) + min_val

    return denormalized

# Multi Layer Perceptron
class MLP:
    def __init__(self, input_size, hidden_size, output_size, momentum_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.momentum_rate = momentum_rate

        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        self.prev_delta_weights_input_hidden = np.zeros((input_size, hidden_size))
        self.prev_delta_weights_hidden_output = np.zeros((hidden_size, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backprop(self, inputs, desired_output, learning_rate):
        error = desired_output - self.final_output
        delta_output = error * self.sigmoid_derivative(self.final_output)

        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Update bias terms for the output layer
        delta_bias_output = delta_output * learning_rate
        self.bias_output += delta_bias_output

        # Update weights for the hidden-to-output layer with momentum
        delta_weights_hidden_output = np.outer(self.hidden_output, delta_output) * learning_rate
        self.weights_hidden_output += delta_weights_hidden_output + self.momentum_rate * self.prev_delta_weights_hidden_output
        self.prev_delta_weights_hidden_output = delta_weights_hidden_output

        # Update bias terms for the hidden layer
        delta_bias_hidden = delta_hidden * learning_rate
        self.bias_hidden += delta_bias_hidden

        # Update weights for the input-to-hidden layer with momentum
        delta_weights_input_hidden = np.outer(inputs, delta_hidden) * learning_rate
        self.weights_input_hidden += delta_weights_input_hidden + self.momentum_rate * self.prev_delta_weights_input_hidden
        self.prev_delta_weights_input_hidden = delta_weights_input_hidden

def train(mlp, normalized_inputs, normalized_outputs, learning_rate, epochs, print_interval=1000):
    mse_history = []  # To store MSE values for each epoch
    mae_history = []  # To store MAE values for each epoch
    weight_history = []  # To store weights for each epoch
    bias_history = []  # To store biases for each epoch

    best_weights_input_hidden = mlp.weights_input_hidden.copy()
    best_weights_hidden_output = mlp.weights_hidden_output.copy()
    best_bias_hidden = mlp.bias_hidden.copy()
    best_bias_output = mlp.bias_output.copy()

    best_mse = float('inf')

    for epoch in range(epochs):
        total_MSE = 0
        total_MAE = 0
        for i in range(len(normalized_inputs)):

            output = mlp.feedforward(normalized_inputs[i])

            MSE = np.mean(np.square(normalized_outputs[i] - output))
            total_MSE += MSE

            MAE = np.mean(np.abs(normalized_outputs[i] - output))
            total_MAE += MAE

            mlp.backprop(normalized_inputs[i], normalized_outputs[i], learning_rate)

        mse_history.append(total_MSE / len(normalized_inputs))
        mae_history.append(total_MAE / len(normalized_inputs))

        weight_history.append((mlp.weights_input_hidden, mlp.weights_hidden_output))
        bias_history.append((mlp.bias_hidden, mlp.bias_output))

        if mse_history[-1] < best_mse:
            best_mse = mse_history[-1]
            best_weights_input_hidden = mlp.weights_input_hidden.copy()
            best_weights_hidden_output = mlp.weights_hidden_output.copy()
            best_bias_hidden = mlp.bias_hidden.copy()
            best_bias_output = mlp.bias_output.copy()

        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}: MSE = {mse_history[-1]}, MAE = {mae_history[-1]}")

    return mse_history, mae_history, (best_weights_input_hidden, best_weights_hidden_output), (best_bias_hidden, best_bias_output)

input_filename = r'C:\Users\ASUS\Desktop\CI\Flood_dataset.txt'

input_arrays, output_arrays = read_file(input_filename)

input_suffer, output_suffer = shuffle_data(input_arrays, output_arrays)

normalized_inputs, normalized_outputs, min_val, max_val = normalize_data(input_suffer, output_suffer)

# parameters
input_size = normalized_inputs.shape[1]
hidden_size = 12
output_size = 1
learning_rate = 0.3
momentum_rate = 0.7
epochs = 12000

num_folds = 10

fold_size = len(normalized_inputs) // num_folds

mse_per_fold = []
mae_per_fold = []
best_correct = 0

for fold in range(num_folds):

    start_idx = fold * fold_size
    end_idx = start_idx + fold_size

    train_inputs = np.concatenate((normalized_inputs[:start_idx], normalized_inputs[end_idx:]), axis=0)
    train_outputs = np.concatenate((normalized_outputs[:start_idx], normalized_outputs[end_idx:]), axis=0)
    test_inputs = normalized_inputs[start_idx:end_idx]
    test_outputs = normalized_outputs[start_idx:end_idx]

    mlp = MLP(input_size, hidden_size, output_size, momentum_rate)

    mse_history, mae_history, best_weights, best_biases = train(mlp, train_inputs, train_outputs, learning_rate, epochs, print_interval=1000)

    total_MSE = 0
    total_MAE = 0
    correct_count = 0

    for i in range(len(test_inputs)):
        output = mlp.feedforward(test_inputs[i])
        denormalized_output = denormalize_data(output, min_val, max_val) 
        design_output = denormalize_data(test_outputs[i], min_val, max_val) 

        absolute_percentage_error = abs((denormalized_output - design_output) / design_output) * 100

        if absolute_percentage_error <= 5:
            correct_count += 1

    correct_percentage = (correct_count / len(test_inputs)) * 100
    
    if correct_percentage > best_correct:
        best_correct = correct_percentage

    print(f"Fold {fold + 1}/{num_folds} - Correct Percentage = {correct_percentage:.2f}%")


print(f"Best correct of all folds: {best_correct:.2f}%")

print('Best Weights (Input to Hidden):')
print(best_weights[0])
print('Best Weights (Hidden to Output):')
print(best_weights[1])
print('Best Bias (Hidden Layer):')
print(best_biases[0])
print('Best Bias (Output Layer):')
print(best_biases[1])
