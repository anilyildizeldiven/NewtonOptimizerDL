#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:13:50 2024

@author: anilcaneldiven
"""

import pytest
from customOptimizer import NewtonOptimizer
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

@pytest.fixture
def iris_data():
    return load_iris(return_X_y=True)

def test_newton_optimizer_performance(benchmark, iris_data):
    X, y = iris_data

    def train_model():
        model = Sequential([
            Dense(10, input_shape=(4,), activation='tanh'),
            Dense(10, activation='tanh'),
            Dense(1, activation='linear')
        ])
        optimizer = NewtonOptimizer()
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(X, y, epochs=10, verbose=0)

    # Benchmark the training function
    benchmark(train_model)
