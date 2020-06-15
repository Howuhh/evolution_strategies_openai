import numpy as np 


def ReLU(x):
    return np.maximum(0, x)


def SoftMax(x):
    x_exp = np.exp(x - np.max(x))

    return x_exp / x_exp.sum()


class ThreeLayerNetwork:
    def __init__(self, in_features, out_features, hidden_sizes=(32, 32)):
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        self.W = self.init_layers()

    def init_layers(self):
        layer1_dim, layer2_dim = self.hidden_sizes

        # +1 to dims for bias trick & He weight init
        W1 = np.random.randn(self.in_features + 1, layer1_dim + 1) * np.sqrt(2 / (self.in_features + 1))
        W2 = np.random.randn(layer1_dim + 1, layer2_dim + 1) * np.sqrt(2 / (layer1_dim + 1))
        W3 = np.random.randn(layer2_dim + 1, self.out_features) * np.sqrt(2 / (layer2_dim + 1))

        return [W1, W2, W3]

    def forward(self, X):
        bias = np.ones((X.shape[0], 1))
        X_bias = np.hstack((X, bias))

        output = ReLU(ReLU(X_bias @ self.W[0]) @ self.W[1]) @ self.W[2]
        
        return SoftMax(output)

    def predict(self, X):
        prob = self.forward(X)[0]

        return np.random.choice(self.out_features, p=prob)


if __name__ == "__main__":
    model = ThreeLayerNetwork(4, 4)
    data = np.random.randn(100, 4)

    prediction = model.forward(data)

    print(prediction.shape)
    print(np.max(prediction))
    print(prediction)
    
