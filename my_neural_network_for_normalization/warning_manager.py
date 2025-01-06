from colorama import Fore, Style

# Define thresholds as constants
MAX_THRESHOLD = 1000  # Maximum threshold for gradients, weights, and loss
MIN_THRESHOLD = 0.000000001  # Minimum threshold for weights


class WarningManager:
    @staticmethod
    def print_warning(message):
        print(Fore.RED + f"[Warning] {message}" + Style.RESET_ALL)

    @staticmethod
    def check_exploding_gradients(dL_dw_hidden, dL_db_hidden, x, t):
        if abs(dL_dw_hidden) > MAX_THRESHOLD or abs(dL_db_hidden) > MAX_THRESHOLD:
            WarningManager.print_warning(f"Exploding gradient detected during training at x={x}, t={t}")

    @staticmethod
    def check_weight_explosion(w_hidden, b_hidden):
        if abs(w_hidden) > MAX_THRESHOLD or abs(b_hidden) > MAX_THRESHOLD:
            WarningManager.print_warning(f"Weight explosion detected: w_hidden={w_hidden}, b_hidden={b_hidden}")

    @staticmethod
    def check_dead_neuron(out_hidden, x, t):
        if out_hidden == 0:
            WarningManager.print_warning(f"Dead neuron detected in hidden layer during training at x={x}, t={t}")

    @staticmethod
    def check_high_total_loss(total_loss):
        if total_loss > MAX_THRESHOLD:
            WarningManager.print_warning(f"High total loss detected: {total_loss:.2f}")

    @staticmethod
    def check_stagnant_learning(total_loss, prev_loss, epoch):
        if prev_loss is not None and abs(total_loss - prev_loss) < MIN_THRESHOLD:
            WarningManager.print_warning(
                f"Stagnant learning detected at epoch {epoch}. Loss difference: {abs(total_loss - prev_loss):.6f}")
