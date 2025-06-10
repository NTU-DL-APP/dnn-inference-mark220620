import numpy as np
import gzip
import json
import h5py

# === Load MNIST Functions ===
def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        f.read(4)
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows * cols) / 255.0
        return data

def load_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        f.read(4)
        num_labels = int.from_bytes(f.read(4), 'big')
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

# === Neural Network Components ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(predictions, labels):
    m = labels.shape[0]
    log_likelihood = -np.log(predictions[range(m), labels] + 1e-9)
    return np.sum(log_likelihood) / m

def softmax_derivative(predictions, labels):
    grad = predictions.copy()
    grad[range(labels.shape[0]), labels] -= 1
    grad /= labels.shape[0]
    return grad

# === Model Initialization ===
def initialize_model(input_dim, hidden_dim, output_dim):
    weights = {
        'dense_1/kernel:0': np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim),
        'dense_1/bias:0': np.zeros(hidden_dim),
        'dense_2/kernel:0': np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim),
        'dense_2/bias:0': np.zeros(output_dim),
    }
    model_arch = [
        {'name': 'dense_1', 'type': 'Dense', 'config': {'activation': 'relu'}, 'weights': ['dense_1/kernel:0', 'dense_1/bias:0']},
        {'name': 'dense_2', 'type': 'Dense', 'config': {'activation': 'softmax'}, 'weights': ['dense_2/kernel:0', 'dense_2/bias:0']},
    ]
    return model_arch, weights

# === Forward and Backward Passes ===
def nn_forward(model_arch, weights, x, cache=False):
    a = x
    if cache:
        activations = {'a0': a}
    for i, layer in enumerate(model_arch):
        W = weights[layer['weights'][0]]
        b = weights[layer['weights'][1]]
        z = a @ W + b
        if layer['config']['activation'] == 'relu':
            a = relu(z)
        elif layer['config']['activation'] == 'softmax':
            a = softmax(z)
        if cache:
            activations[f'z{i+1}'] = z
            activations[f'a{i+1}'] = a
    return (a, activations) if cache else a

def backward_pass(x, y, model_arch, weights, learning_rate=0.01):
    y_hat, cache = nn_forward(model_arch, weights, x, cache=True)

    grads = {}
    dz = softmax_derivative(y_hat, y)

    for i in reversed(range(len(model_arch))):
        layer = model_arch[i]
        W_name, B_name = layer['weights']
        a_prev = cache[f'a{i}']
        W = weights[W_name]

        grads[W_name] = a_prev.T @ dz
        grads[B_name] = np.sum(dz, axis=0)

        if i > 0:
            da_prev = dz @ W.T
            dz = da_prev * (cache[f'z{i}'] > 0)  # ReLU derivative

    for name in grads:
        weights[name] -= learning_rate * grads[name]

# === Training Loop ===
def train(x_train, y_train, input_dim=784, hidden_dim=128, output_dim=10,
          epochs=5, batch_size=64, lr=0.01):
    model_arch, weights = initialize_model(input_dim, hidden_dim, output_dim)

    for epoch in range(epochs):
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]

        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            backward_pass(x_batch, y_batch, model_arch, weights, lr)

        y_pred = nn_forward(model_arch, weights, x_train)
        loss = cross_entropy_loss(y_pred, y_train)
        acc = np.mean(np.argmax(y_pred, axis=1) == y_train)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    return model_arch, weights

# === Save Functions ===
def save_model_to_h5(weights, architecture, file_path):
    with h5py.File(file_path, 'w') as f:
        weights_group = f.create_group('weights')
        for name, array in weights.items():
            weights_group.create_dataset(name, data=array)
        arch_group = f.create_group('architecture')
        arch_group.attrs['model_arch'] = json.dumps(architecture)

def save_weights_to_npz(weights, file_path):
    np.savez(file_path, **weights)

def save_architecture_to_json(architecture, file_path):
    with open(file_path, 'w') as f:
        json.dump(architecture, f, indent=2)

# === Run Training and Save Outputs ===
if __name__ == "__main__":
    x_test = load_mnist_images("data/fashion/t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels("data/fashion/t10k-labels-idx1-ubyte.gz")

    model_arch, weights = train(x_test, y_test, epochs=100)

    save_model_to_h5(weights, model_arch, 'fashion_mnist.h5')
    save_weights_to_npz(weights, 'fashion_mnist.npz')
    save_architecture_to_json(model_arch, 'fashion_mnist.json')

    print("✅ Model saved to: fashion_mnist_model.h5")
    print("✅ Weights saved to: fashion_mnist.npz")
    print("✅ Architecture saved to: fashion_mnist.json")
