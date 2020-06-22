import numpy as np

# maybe will useful later
def flatten_weights(layers):
    flat_layers = np.hstack([layer.flatten() for layer in layers])
    return flat_layers


def split_weights(weights, shapes):
    splits = np.cumsum([dim[0]*dim[1] for dim in shapes[:-1]])
    new_weights = np.split(weights, splits, axis=0)

    return [w.reshape(shapes[i]) for i, w in enumerate(new_weights)]