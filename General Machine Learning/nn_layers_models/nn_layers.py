"""
Script Name: nn_layers.py

Description:
This script develops a single classifier neural network of n-layers to classify points generated randomly.
The data is generated from matplotlib and it is fed to the neural network to classify the points and generate a decision boundary. Both the   is reads data from a CSV file and trains three models: Linear Regression, Decision Tree, and a Neural Network.
For each model, it computes evaluation metrics including accuracy, precision, and recall.
After selecting the best-performing model, the script analyzes the most important features contributing to the prediction.

Dependencies:
- pandas
- numpy
- matplotlib

Usage:
$ python3 nn_layers.py

Author: Riccardo Nicolò Iorio
Date:
"""
import copy
from typing import List, Dict, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import sklearn


### Synthetic Data Generation for Classification Model Evaluation ###

def data_generation() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(3)
    X, Y = sklearn.datasets.make_moons(100, noise = 0.3)
    plt.scatter(X[:, 0], X[:, 1], s = 40, c = Y, cmap = 'Spectral')
    plt.title("Two classes dataset")

    # Save plot as an image (JPG format)
    plt.savefig(f'Generated_data_nn_layer.jpg', dpi=500)
    plt.close()

    return X.T, Y.reshape(1, -1)

### Forward Propagation part for the Classification model ###

def sigmoid(Z: np.ndarray) -> np.ndarray:
    A = 1 / (1 + np.exp(-Z))

    return A

def tanh(Z: np.ndarray) -> np.ndarray:
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    return A

def relu(Z: np.ndarray) -> np.ndarray:
    A = np.maximum(0,Z)

    return A

activations_func : Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    }

def parameters_initialisation(layers_dims: List[int]) -> Dict[str, np.ndarray]:

    np.random.seed(2)
    param = {}
    depth = len(layers_dims)

    for layer in range(1, depth):
        param['W' + str(layer)] = np.random.rand(layers_dims[layer], layers_dims[layer - 1]) * np.sqrt(2. / layers_dims[layer - 1])
        param['b' + str(layer)] = np.zeros((layers_dims[layer], 1))

    return param

def linear_forward(layer: int, A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    weights_cache = {}
    activation_cache = {}
    Z = W @ A + b

    weights_cache['W' + str(layer)] = W
    weights_cache['b' + str(layer)] = b

    activation_cache['A' + str(layer - 1)] = A
    activation_cache['Z' + str(layer)] = Z

    return activation_cache, weights_cache

def activation_linear_forward(layer: int, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, function: str) -> Tuple[np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    if function not in activations_func:
        raise ValueError(f"Unsupported activation function: {function}")

    layer_activation_cache, layer_weights_cache = linear_forward(layer, A_prev, W, b)
    A = activations_func[function](layer_activation_cache['Z' + str(layer)])

    layer_cache = (layer_weights_cache, layer_activation_cache)

    return A, layer_cache

def model_forward_propagation(X: np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]]:
    total_forward_caches = []
    A = X
    total_depth = len(parameters) // 2

    for layer in range(1, total_depth):
        A_prev = A
        A, layer_cache = activation_linear_forward(layer, A_prev, parameters['W' + str(layer)], parameters['b' + str(layer)], 'tanh')
        total_forward_caches.append(layer_cache)

    A_last_layer, last_layer_cache = activation_linear_forward(total_depth, A, parameters['W' + str(total_depth)], parameters['b' + str(total_depth)], 'sigmoid')
    total_forward_caches.append(last_layer_cache)

    return A_last_layer, total_forward_caches

### Cost Function definition for the Classification model ###

def cost_function(A_last_layer: np.ndarray, Y: np.ndarray) -> float:
    m = Y.shape[1]

    cost = - (1 / m) * np.sum(Y * np.log(A_last_layer) + (1 - Y) * np.log(1 - A_last_layer))

    return cost

### Backward Propagation part for the Classification model ###

def sigmoid_backward(layer: int, grad_A: np.ndarray, layer_activation_cache: Dict[str, np.ndarray]) -> np.ndarray:
    Z = layer_activation_cache['Z' + str(layer + 1)]
    s = 1 / (1 + np.exp(-Z))
    grad_Z = grad_A * s * (1 - s)

    assert (grad_Z.shape == Z.shape)

    return grad_Z

def tanh_backward(layer: int, grad_A: np.ndarray, layer_activation_cache: Dict[str, np.ndarray]) -> np.ndarray:
    Z = layer_activation_cache['Z' + str(layer + 1)]
    s = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    grad_Z = grad_A * (1 - s ** 2)

    assert (grad_Z.shape == Z.shape)

    return grad_Z

def relu_backward(layer: int, grad_A: np.ndarray, layer_activation_cache: Dict[str, np.ndarray]) -> np.ndarray:
    Z = layer_activation_cache['Z' + str(layer + 1)]
    grad_Z = np.array(grad_A, copy = True)
    grad_Z[Z < 0] = 0
    assert (grad_Z.shape == Z.shape)

    return grad_Z

activations_func_backward : Dict[str, Callable[[int, np.ndarray, Dict[str, np.ndarray]], np.ndarray]] = {
    'sigmoid': sigmoid_backward,
    'tanh': tanh_backward,
    'relu': relu_backward,
    }

def linear_backward(layer: int, grad_Z: np.ndarray, layer_weights_cache: Dict[str, np.ndarray], layer_activation_cache: Dict[str, np.ndarray])\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    W = layer_weights_cache['W' + str(layer + 1)]
    A_prev = layer_activation_cache['A' + str(layer)]
    m = A_prev.shape[1]

    grad_W = (1 / m) * grad_Z @ A_prev.T
    grad_b = (1 / m) * np.sum(grad_Z, axis=1, keepdims=True)
    grad_A_prev = W.T @ grad_Z

    return grad_W, grad_A_prev, grad_b

def activation_linear_backward(layer: int, grad_A: np.ndarray, layer_cache: Tuple[Dict[str, np.ndarray], Dict[str,np.ndarray]], function: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if function not in activations_func_backward:
        raise ValueError(f"Unsupported activation function: {function}")

    layer_weights_cache, layer_activation_cache = layer_cache

    layer_grad_cache = {}

    grad_Z = activations_func_backward[function](layer, grad_A, layer_activation_cache)
    grad_W, grad_A_prev, grad_b = linear_backward(layer, grad_Z, layer_weights_cache, layer_activation_cache)

    layer_grad_cache['dW' + str(layer + 1)] = grad_W
    layer_grad_cache['db' + str(layer + 1)] = grad_b
    layer_grad_cache['dA' + str(layer)] = grad_A_prev
    layer_grad_cache['dZ' + str(layer + 1)] = grad_Z

    return grad_A_prev, layer_grad_cache

def model_backward_propagation(A_last_layer: np.ndarray, Y: np.ndarray, total_forward_caches: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]) -> List[Dict[str, np.ndarray]]:

    total_grads = []
    total_depth_cache = len(total_forward_caches)

    grad_A_last_layer = - (np.divide(Y, A_last_layer) - np.divide(1 - Y, 1 - A_last_layer))
    current_cache = total_forward_caches[total_depth_cache - 1]
    grad_A_prev_layer, last_layer_grad_cache = activation_linear_backward(total_depth_cache - 1, grad_A_last_layer, current_cache, 'sigmoid')

    total_grads.append(last_layer_grad_cache)

    for layer in reversed(range(total_depth_cache - 1)):
        current_cache = total_forward_caches[layer]
        grad_A_prev_layer, layer_grad_cache = activation_linear_backward(layer, grad_A_prev_layer, current_cache, 'tanh')

        total_grads.append(layer_grad_cache)

    return total_grads

def parameters_update(params: Dict[str, np.ndarray], total_grads: List[Dict[str, np.ndarray]], learning_rate: float) -> Dict[str, np.ndarray]:
    total_depth = len(params) // 2

    for layer in range(total_depth):
        layer_grad_cache = total_grads[total_depth - (layer + 1)]
        params['W' + str(layer + 1)] = params['W' + str(layer + 1)] - learning_rate * layer_grad_cache['dW' + str(layer + 1)]
        params['b' + str(layer + 1)] = params['b' + str(layer + 1)] - learning_rate * layer_grad_cache['db' + str(layer + 1)]

    return params

def neural_network_model(data: Tuple[np.ndarray, np.ndarray], layers: List[int], total_epochs: int, learning_rate: float, print_cost: bool) -> List[Tuple[int, float]]:

    X, Y = data
    layers_dim = [X.shape[0]] + layers + [Y.shape[0]]
    params = parameters_initialisation(layers_dim)
    print("Print of the initialised weights of the model")
    for key, value in params.items():
        print(f"{key} = {value}")

    cost_track = []

    for epoch in range(total_epochs):
        prediction, total_caches = model_forward_propagation(X, params)
        cost = cost_function(prediction, Y)
        total_grads = model_backward_propagation(prediction, Y, total_caches)
        params = parameters_update(params, total_grads, learning_rate)

        if print_cost and epoch % 1000 == 0:
                print(f"Cost after iteration {epoch}: {cost}")

        cost_track.append((epoch,cost))

    for key, value in params.items():
        print(f"{key} = {value}")

    return cost_track

### Plot functions for debugging ###

#def

#def

def main() -> None:

    layers = [4, 3, 5, 3]
    data = data_generation()
    neural_network_model(data, layers, 10000, 0.1, True)
    # params = parameters_initialisation([2,3,1])
    # print("Print of the initialised weights of the model")
    # for key, value in params.items():
    #     print(f"{key} = {value}")
    # prediction, total_caches = model_forward_propagation(data1, params)
    # for layer in range(len(total_caches)):
    #     weights_cache, activation_cache = total_caches[layer]
    #     print(f"Caches for layer{layer + 1}")
    #     print(f"W{layer + 1} = {weights_cache[f'W{layer + 1}']}")
    #     print(f"A{layer} = {activation_cache[f'A{layer}']}")
    #     print(f"b{layer + 1} = {weights_cache[f'b{layer + 1}']}")
    #     print(f"Z{layer + 1} = {activation_cache[f'Z{layer + 1}']}")
    # print(f"The cost function values is: {cost_function(prediction, data2)}")
    # total_grads =  model_backward_propagation(prediction, data2, total_caches)
    # depth_grad_cache = len(total_grads)
    # for layer in reversed(range(depth_grad_cache)):
    #     grads_cache = total_grads[layer]
    #     print(f"Gradient cache for layer{depth_grad_cache - layer}")
    #     print(f"dW{depth_grad_cache - layer} = {grads_cache[f'dW{depth_grad_cache - layer}']}")
    #     print(f"db{depth_grad_cache - layer} = {grads_cache[f'db{depth_grad_cache - layer}']}")
    #     print(f"dA{depth_grad_cache - (layer + 1)} = {grads_cache[f'dA{depth_grad_cache - (layer + 1)}']}")
    #     print(f"dZ{depth_grad_cache - layer} = {grads_cache[f'dZ{depth_grad_cache - layer}']}")
    # print(parameters_update(params, total_grads, 0.01))





if __name__ == "__main__":
    main()