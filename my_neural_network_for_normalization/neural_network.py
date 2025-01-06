from my_neural_network_for_normalization.warning_manager import WarningManager
from my_neural_network_for_normalization.activation_function import ReLU
import random


class NeuralNetwork:
    def __init__(self, learning_rate=0.1, w_hidden=None, b_hidden=None, w_output=None, b_output=None):
        self.out_output = None
        self.net_output = None
        self.out_hidden = None
        self.net_hidden = None

        self.w_hidden = w_hidden if w_hidden is not None else random.uniform(-0.5, 0.5)
        self.b_hidden = b_hidden if b_hidden is not None else random.uniform(-0.5, 0.5)
        self.w_output = w_output if w_output is not None else random.uniform(-0.5, 0.5)
        self.b_output = b_output if b_output is not None else random.uniform(-0.5, 0.5)

        self.learning_rate = learning_rate

    def forward_propagation(self, x):
        self.net_hidden = self.w_hidden * x + self.b_hidden
        self.out_hidden = ReLU.activation(self.net_hidden)

        self.net_output = self.w_output * self.out_hidden + self.b_output
        self.out_output = ReLU.activation(self.net_output)

        print(f"  Forward Propagation for input x={x}")
        print(f"    net_hidden: {self.net_hidden}, out_hidden: {self.out_hidden}")
        print(f"    net_output: {self.net_output}, out_output: {self.out_output}")

        return self.net_hidden, self.out_hidden, self.net_output, self.out_output

    def back_propagation(self, x, t):
        dL_dout_output = 2 * (self.out_output - t)
        dout_output_dnet_output = ReLU.derivative(self.net_output)
        dnet_output_dw_output = self.out_hidden
        dnet_output_db_output = 1

        dL_dw_output = dL_dout_output * dout_output_dnet_output * dnet_output_dw_output
        dL_db_output = dL_dout_output * dout_output_dnet_output * dnet_output_db_output

        dnet_output_dout_hidden = self.w_output
        dout_hidden_dnet_hidden = ReLU.derivative(self.net_hidden)
        dnet_hidden_dw_hidden = x
        dnet_hidden_db_hidden = 1

        dL_dw_hidden = dL_dout_output * dout_output_dnet_output * dnet_output_dout_hidden * dout_hidden_dnet_hidden * dnet_hidden_dw_hidden
        dL_db_hidden = dL_dout_output * dout_output_dnet_output * dnet_output_dout_hidden * dout_hidden_dnet_hidden * dnet_hidden_db_hidden

        # Check warnings
        WarningManager.check_exploding_gradients(dL_dw_hidden, dL_db_hidden, x, t)
        WarningManager.check_weight_explosion(self.w_hidden, self.b_hidden)
        WarningManager.check_dead_neuron(self.out_hidden, x, t)


        print(f"  Backpropagation for input x={x}, target t={t}")
        print(f"    Gradients: dL_dw_hidden={dL_dw_hidden}, dL_db_hidden={dL_db_hidden}")
        print(f"    Updated weights and biases:")
        print(f"      w_hidden: {self.w_hidden}, b_hidden: {self.b_hidden}")
        print(f"      w_output: {self.w_output}, b_output: {self.b_output}")

        self.w_output -= self.learning_rate * dL_dw_output
        self.b_output -= self.learning_rate * dL_db_output

        self.w_hidden -= self.learning_rate * dL_dw_hidden
        self.b_hidden -= self.learning_rate * dL_db_hidden

    def predict(self, x):
        """Uses the forward propagation to predict the output for a given input."""
        _, _, _, out_output = self.forward_propagation(x)
        return out_output

    def train(self, epochs, training_data):
        prev_loss = None

        for epoch in range(epochs):
            total_loss = 0
            print(f"Epoch {epoch}:")
            for x, t in training_data:
                self.forward_propagation(x)
                L = (self.out_output - t) ** 2
                total_loss += L
                print(f"    Loss: {L}")
                self.back_propagation(x, t)

            WarningManager.check_high_total_loss(total_loss)
            WarningManager.check_stagnant_learning(total_loss, prev_loss, epoch)

            prev_loss = total_loss