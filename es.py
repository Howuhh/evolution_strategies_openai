import numpy as np

from copy import deepcopy


class OpenAiES:
    def __init__(self, model, learning_rate, noise_std):
        self.model = model
        self.lr = learning_rate
        self.noise_std = noise_std

        self._population = None

    def generate_population(self, npop=50):
        self._population = []

        for i in range(npop):
            new_model = deepcopy(self.model)
            new_model.E = []

            for i, layer in enumerate(new_model.W):
                noise = np.random.randn(layer.shape[0], layer.shape[1])

                new_model.E.append(noise)
                new_model.W[i] = new_model.W[i] + self.noise_std * noise
            self._population.append(new_model)

        return self._population

    def update_population(self, rewards):
        if self._population is None:
            raise ValueError("populations is none, generate & eval it first")

        # z-normalization (?) - works better, but slower
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for i, layer in enumerate(self.model.W):
            w_updates = np.zeros_like(layer)

            for j, model in enumerate(self._population):
                w_updates = w_updates + (model.E[i] * rewards[j])

            # SGD weights update
            self.model.W[i] = self.model.W[i] + (self.lr / (len(rewards) * self.noise_std)) * w_updates

    def get_model(self):
        return self.model
