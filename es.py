import numpy as np

from copy import deepcopy


class OpenAiES:
    def __init__(self, model, learning_rate, noise_std, \
                    noise_decay=1.0, lr_decay=1.0, decay_step=50, norm_rewards=True):
        self.model = model
        
        self._lr = learning_rate
        self._noise_std = noise_std
        
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step

        self.norm_rewards = norm_rewards

        self._population = None
        self._count = 0

    @property
    def noise_std(self):
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))

        return self._noise_std * step_decay

    @property
    def lr(self):
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))

        return self._lr * step_decay

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
        if self.norm_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for i, layer in enumerate(self.model.W):
            w_updates = np.zeros_like(layer)

            for j, model in enumerate(self._population):
                w_updates = w_updates + (model.E[i] * rewards[j])

            # SGD weights update
            self.model.W[i] = self.model.W[i] + (self.lr / (len(rewards) * self.noise_std)) * w_updates
        
        self._count = self._count + 1

    def get_model(self):
        return self.model


class OpenAIES_NSR:
    # TODO: novelity search
    def __init__(self):
        pass