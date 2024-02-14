# NewtonOptimizer for TensorFlow

The `NewtonOptimizer` is a custom implementation of an optimization algorithm for TensorFlow, utilizing the Newton method. This optimizer is particularly well-suited for optimizing machine learning models where fine-tuning of model parameters is required.

## Features

- Implemented as a class inheriting from `tensorflow.python.keras.optimizer_v2.OptimizerV2`.
- Supports customization of the subsampling rate through the `subsampling_rate` parameter.
- Can be directly used in TensorFlow Keras models as an optimizer.

## Prerequisites

Before you can use the `NewtonOptimizer`, ensure the following software is installed on your system:

- Python 3.8 or higher
- TensorFlow 2.x

## Installation

Since the `NewtonOptimizer` is provided as part of a Python script and not as a standalone package, you will need to copy the script directly into your project directory or paste its contents into an existing Python file.

1. Copy `newton_optimizer.py` (the filename for the script containing the `NewtonOptimizer`) into your project directory.

## Usage

To use the `NewtonOptimizer` in your TensorFlow Keras model, import the class and specify it as the optimizer when compiling the model. Here's a simple example:

```python
import tensorflow as tf
from newton_optimizer import NewtonOptimizer # Ensure the import path is correct

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(None, 20)),
    tf.keras.layers.Dense(1)
])

optimizer = NewtonOptimizer(subsampling_rate=0.7) 
get_custom_objects().update({'NewtonOptimizer': optimizer})
model.compile(optimizer='NewtonOptimizer', loss='mse')

# Now you can train your model as usual
model.fit(x_train, y_train, epochs=5)

##Contributing

Contributions to this project are welcome! If you'd like to suggest an improvement, please feel free to create a pull request or open an issue.

##License

This project is made freely available. You may use it for personal or commercial projects as you wish.
