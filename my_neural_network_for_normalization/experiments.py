from my_neural_network_for_normalization.data_normalizer import DataNormalizer
from my_neural_network_for_normalization.neural_network import NeuralNetwork

def train_and_predict(training_data, input_value, normalize=False):
    """
    Trains the neural network on the given data and predicts output for a given input value.
    Optionally normalizes the training data.
    """
    # Initialize Neural Network
    nn = NeuralNetwork(learning_rate=0.1, w_hidden=0.2, b_hidden=0.3, w_output=0.4, b_output=0.5)

    if normalize:
        normalizer = DataNormalizer(training_data)
        training_data = normalizer.get_normalized_data()
        normalized_input_value = normalizer.normalize_input(input_value)
        nn.train(epochs=74, training_data=training_data)
        prediction = normalizer.denormalize_output(nn.predict(normalized_input_value))
    else:
        nn.train(epochs=74, training_data=training_data)
        prediction = nn.predict(input_value)

    print("################################")
    print(f"# x = {input_value}, y = {prediction} (normalized: {normalize}) #")
    print("################################")

def run_experiment():
    """
    Runs experiments with different datasets and normalization settings.
    """
    experiments = {
        "1": ("Raw Data with Large Input Differences", [(1, 2), (1002, 1003)], 7, False),
        "2": ("Normalized Data with Large Input Differences", [(1, 2), (1002, 1003)], 7, True),
        "3": ("Raw Data with Small Input Differences", [(1, 2), (2, 3)], 7, False),
        "4": ("Normalized Data with Small Input Differences", [(1, 2), (2, 3)], 7, True),
    }

    print("Select an experiment to run:")
    for key, (description, _, _, _) in experiments.items():
        print(f"{key}: {description}")

    choice = input("Enter the number of the experiment you want to run: ")

    if choice in experiments:
        description, training_data, input_value, normalize = experiments[choice]
        print(f"\nRunning {description}...")
        train_and_predict(training_data, input_value, normalize)
    else:
        print("Invalid choice. Please select a valid experiment number.")

if __name__ == "__main__":
    run_experiment()
