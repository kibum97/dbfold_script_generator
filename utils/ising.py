import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, states, n_samples):
        self.n_spins = np.
        self.n_samples = n_samples
        self.spins = np.random.choice([-1, 1], size=(n_samples, n_spins))
        self.J = np.random.rand(n_spins, n_spins)  # Coupling matrix
        self.h = np.random.rand(n_spins)  # External field

    def energy(self):
        """Calculate the energy of the system."""
        return -0.5 * np.sum(self.J * (self.spins[:, :, None] * self.spins[:, None, :]), axis=(1, 2)) - np.sum(self.h * self.spins, axis=1)

    def train_logistic_regression(self):
        """Train a logistic regression model on the spin configurations."""
        X = self.spins
        y = (self.energy() < 0).astype(int)  # Label based on energy
        model = LogisticRegression()
        model.fit(X, y)
        return model

    def plot_energy_distribution(self):
        """Plot the distribution of energies."""
        energies = self.energy()
        sns.histplot(energies, bins=30)
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.title('Energy Distribution')
        plt.show()