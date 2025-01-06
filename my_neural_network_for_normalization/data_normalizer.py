class DataNormalizer:
    def __init__(self, data):
        """Initialize the normalizer and fit it to the data."""
        input_data, output_data = zip(*data)
        self.input_min = min(input_data)
        self.input_max = max(input_data)
        self.output_min = min(output_data)
        self.output_max = max(output_data)

        self.normalized_data = self._normalize_data(data)

    def _normalize_data(self, data):
        """Normalize the data based on the fitted min and max values."""
        range_input = self.input_max - self.input_min if self.input_max != self.input_min else 1
        range_output = self.output_max - self.output_min if self.output_max != self.output_min else 1

        return [
            ((x - self.input_min) / range_input, (y - self.output_min) / range_output)
            for x, y in data
        ]

    def get_normalized_data(self):
        """Return the normalized data."""
        return self.normalized_data

    def normalize_input(self, x):
        range_input = self.input_max - self.input_min if self.input_max != self.input_min else 1
        return (x - self.input_min) / range_input

    def denormalize_output(self, y_normalized):
        range_output = self.output_max - self.output_min if self.output_max != self.output_min else 1
        return y_normalized * range_output + self.output_min
