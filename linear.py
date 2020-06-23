import pickle

import numpy as np 


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()


def tanh(x):
    return np.tanh(x)
    

class ThreeLayerNetwork:
    def __init__(self, in_features, out_features, hidden_sizes=(32, 32)):
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        self.W = self._init_layers()

    # TODO: init weights from model -> load_model(self, path)

    def _init_layers(self):
        layer1_dim, layer2_dim = self.hidden_sizes

        # +1 to dims for bias trick & He weight init
        W1 = np.random.randn(self.in_features + 1, layer1_dim + 1) * np.sqrt(2 / (self.in_features + 1))
        W2 = np.random.randn(layer1_dim + 1, layer2_dim + 1) * np.sqrt(2 / (layer1_dim + 1))
        W3 = np.random.randn(layer2_dim + 1, self.out_features) * np.sqrt(2 / (layer2_dim + 1))

        return [W1, W2, W3]

    @staticmethod
    def from_model(path):
        with open(path, "rb") as file:
            model = pickle.load(file)

        assert isinstance(model, ThreeLayerNetwork), "init model is not instance of ThreeLayerNetwork class"

        return model

    def forward(self, X):
        bias = np.ones((X.shape[0], 1))
        X_bias = np.hstack((X, bias))

        output = ReLU(ReLU(X_bias @ self.W[0]) @ self.W[1]) @ self.W[2]
        
        return output

    def predict(self, X, scale="softmax"):
        X_norm = (X - X.mean()) / (X.std() + 1e-5)

        raw_output = self.forward(X_norm)
        
        if scale == "tanh":
            return tanh(raw_output)[0]       
        elif scale == "softmax":
            prob = softmax(raw_output)[0]
            # TODO: action choice more about agent than model
            return np.random.choice(self.out_features, p=prob)

        return raw_output[0]


if __name__ == "__main__":
    model = ThreeLayerNetwork(4, 4)
    data = np.random.randn(1, 4)

    prediction = model.predict(data, scale="tanh")
    
    print(prediction)
    
