import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from customOptimizer import NewtonOptimizer
from sklearn.datasets import load_iris
from customOptimizer import NewtonOptimizer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({'NewtonOptimizer': NewtonOptimizer})
# Bereite die Daten vor
X, y = load_iris(return_X_y=True)

# Definiere eine Funktion zum Erstellen des Modells
def create_model(optimizer):
    model = Sequential([
        Dense(10, input_shape=(4,), activation='tanh'),
        Dense(10, activation='tanh'),
        Dense(1, activation='linear')
    ])
    NewtonOptimizer()
    model.compile(optimizer='NewtonOptimizer', loss='mse')
    return model

# Benchmark-Funktion für NewtonOptimizer
def benchmark_newton_optimizer(benchmark):
    model = create_model(NewtonOptimizer())
    benchmark(model.fit, X, y, epochs=10, verbose=0)

# Benchmark-Funktion für SGD
def benchmark_sgd_optimizer(benchmark):
    model = create_model(SGD())
    benchmark(model.fit, X, y, epochs=10, verbose=0)
