class ReLU:
    @staticmethod
    def activation(x):
        return max(0, x)

    @staticmethod
    def derivative(x):
        return 1 if x > 0 else 0
