# Define the ReLU activation function and its derivative
def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

# Initialize weights and biases
w_hidden = 0.2  # Weight from input to hidden neuron
b_hidden = 0.3  # Bias for hidden neuron
w_output = 0.4  # Weight from hidden to output neuron
b_output = 0.5  # Bias for output neuron

# Learning rate
alpha = 0.1  # Learning rate

# Training data
training_data = [
    (1, 2),  # t = x + 1
    (2, 3)
]

def forward_propagation(x):
    """Performs the forward propagation step."""
    net_hidden = w_hidden * x + b_hidden
    out_hidden = relu(net_hidden)

    net_output = w_output * out_hidden + b_output
    out_output = relu(net_output)

    # Print intermediate values for forward pass
    print(f"  Forward Propagation for input x={x}")
    print(f"    net_hidden: {net_hidden}, out_hidden: {out_hidden}")
    print(f"    net_output: {net_output}, out_output: {out_output}")

    return net_hidden, out_hidden, net_output, out_output

def back_propagation(x, t, net_hidden, out_hidden, net_output, out_output):
    """Performs the backpropagation step and updates weights and biases."""
    global w_hidden, b_hidden, w_output, b_output

    # Output layer gradients
    dL_dout_output = 2 * (out_output - t)
    dout_output_dnet_output = relu_derivative(net_output)
    dnet_output_dw_output = out_hidden
    dnet_output_db_output = 1

    # Chain rule for output layer weights and biases
    dL_dw_output = dL_dout_output * dout_output_dnet_output * dnet_output_dw_output
    dL_db_output = dL_dout_output * dout_output_dnet_output * dnet_output_db_output

    # Hidden layer gradients
    dnet_output_dout_hidden = w_output
    dout_hidden_dnet_hidden = relu_derivative(net_hidden)
    dnet_hidden_dw_hidden = x
    dnet_hidden_db_hidden = 1

    # Chain rule for hidden layer weights and biases
    dL_dw_hidden = dL_dout_output * dout_output_dnet_output * dnet_output_dout_hidden * dout_hidden_dnet_hidden * dnet_hidden_dw_hidden
    dL_db_hidden = dL_dout_output * dout_output_dnet_output * dnet_output_dout_hidden * dout_hidden_dnet_hidden * dnet_hidden_db_hidden

    # Update weights and biases
    w_output -= alpha * dL_dw_output
    b_output -= alpha * dL_db_output

    w_hidden -= alpha * dL_dw_hidden
    b_hidden -= alpha * dL_db_hidden

    # Print updated weights and biases
    print(f"  Backpropagation for input x={x}, target t={t}")
    print(f"    Updated weights and biases:")
    print(f"      w_hidden: {w_hidden}, b_hidden: {b_hidden}")
    print(f"      w_output: {w_output}, b_output: {b_output}")

def predict(x):
    """Uses the forward propagation to predict the output for a given input."""
    _, _, _, out_output = forward_propagation(x)
    return out_output

def train(epochs, training_data):
    """Trains the neural network for a given number of epochs."""
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch}:")
        for x, t in training_data:
            # Forward pass
            net_hidden, out_hidden, net_output, out_output = forward_propagation(x)

            # Compute loss
            L = (out_output - t) ** 2
            total_loss += L

            # Print loss
            print(f"    Loss: {L}")

            # Backpropagation
            back_propagation(x, t, net_hidden, out_hidden, net_output, out_output)

        print(f"  Total Loss for Epoch {epoch}: {total_loss}\n")

# Train the neural network
train(30, training_data)

# Test the neural network after training

x = 7
y = predict(x)
print("\nPredictions after training:")
print("################################")
print(f"# x = {x}, y = {y} #")
print("################################")