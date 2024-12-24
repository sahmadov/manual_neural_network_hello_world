from my_neural_network.neural_network import NeuralNetwork


# Training data
training_data = [
    (1, 2),  # t = x + 1
    (2, 3)
]

# Create and train the neural network
nn = NeuralNetwork(learning_rate=0.1, w_hidden=0.2, b_hidden=0.3, w_output=0.4, b_output=0.5)
nn.train(epochs=30, training_data=training_data)

# Test the neural network after training
x = 7
y = nn.predict(x)
print("\nPredictions after training:")
print("################################")
print(f"# x = {x}, y = {y} #")
print("################################")