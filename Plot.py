import matplotlib.pyplot as plt
import numpy as np

class experiment:
    def __init__(self, filename):
        self.timing, self.train_mse, self.train_std, self.test_mse, self.test_std = load_data(filename)

def load_data(filename):
    data = np.load(filename)
    timing = data[0]
    train_mse = data[1]
    train_std = data[2]
    test_mse = data[3]
    test_std = data[4]
    return timing, train_mse, train_std, test_mse, test_std

def plot_experiment(experiments, plot_training_mse=True, plot_testing_mse=True):
    for e in experiments:
        if plot_training_mse:
            plt.plot(e.timing, e.train_mse,
                 label="Training MSE of elite individual")
            plt.fill_between(e.timing, e.train_mse+e.train_std, e.train_mse-e.train_std, alpha=0.2)
        if plot_testing_mse:
            plt.plot(e.timing, e.test_mse,
                         label="Testing MSE of elite individual")
            plt.fill_between(e.timing, e.test_mse+e.test_std, e.test_mse-e.test_std, alpha=0.2)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('MSE')
        plt.show()