#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:03:21 2024

@author: anilcaneldiven
"""

import unittest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import get_custom_objects
# Importieren Sie hier Ihren NewtonOptimizer
from customOptimizer import NewtonOptimizer 
import numpy as np

class NewtonOptimizerTestCase(unittest.TestCase):
    def test_initialization(self):
        """Testet die Initialisierung des Newton Optimizers."""
        optimizer = NewtonOptimizer(subsampling_rate=0.5)

        self.assertEqual(optimizer.subsampling_rate, 0.5)

    def test_simple_optimization(self):
        """Testet den Optimizer mit einem einfachen Modell."""
        # Definiere ein einfaches Modell
        model = Sequential([
            Dense(1, input_shape=(1,), activation='linear')
        ])

        # Initialisiere den NewtonOptimizer
        optimizer = NewtonOptimizer(subsampling_rate=0.4)
        get_custom_objects().update({'NewtonOptimizer': optimizer})
        # Kompiliere das Modell
        model.compile(optimizer='NewtonOptimizer', loss='mse')

        # Generiere einfache Trainingsdaten
        x_train = tf.constant([[1.0], [2.0], [3.0], [4.0]])
        y_train = tf.constant([[2.0], [4.0], [6.0], [8.0]])

        # Trainiere das Modell
        history = model.fit(x_train, y_train, epochs=10, verbose=0)

        # Überprüfe, ob das Training ohne Fehler durchgeführt wurde
        self.assertTrue(history is not None)
    
    def test_weight_update(self):
        """Testet, ob die Gewichte des Modells nach dem Training aktualisiert werden."""
        model = Sequential([
            Dense(2, input_shape=(3,), kernel_initializer='ones', use_bias=False)
        ])
        optimizer = NewtonOptimizer(subsampling_rate=1.0)  # 100% Sampling für den Test
        get_custom_objects().update({'NewtonOptimizer': optimizer})
        # Kompiliere das Modell
        model.compile(optimizer='NewtonOptimizer', loss='mse')
    
        initial_weights = model.get_weights()[0].copy()
    
        # Einfache Trainingsdaten
        x_train = np.random.rand(10, 3)
        y_train = np.random.rand(10, 2)
    
        model.fit(x_train, y_train, epochs=1, verbose=0)
    
        updated_weights = model.get_weights()[0]
    
        # Überprüfen, ob sich die Gewichte verändert haben
        self.assertFalse(np.array_equal(initial_weights, updated_weights), "Gewichte sollten sich nach dem Training aktualisieren.")

    def test_optimizer_integration(self):
        """Testet die Integration des Optimizers mit einem Keras-Modell."""
        model = Sequential([
            Dense(1, input_shape=(1,), activation='linear')
        ])
        optimizer = NewtonOptimizer()
        get_custom_objects().update({'NewtonOptimizer': optimizer})
    
        # Dies sollte keine Fehler werfen
        try:
            model.compile(optimizer='NewtonOptimizer', loss='mse')
            compiled = True
        except Exception as e:
            compiled = False
        
        self.assertTrue(compiled, "Optimizer sollte erfolgreich in einem Keras-Modell kompiliert werden.")

    def test_large_dataset_handling(self):
        """Überprüft, ob der Optimizer große Datensätze verarbeiten kann."""
        model = Sequential([
            Dense(10, input_shape=(20,), activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = NewtonOptimizer(subsampling_rate=0.1)  # Niedrigere Rate für größere Datasets
        get_custom_objects().update({'NewtonOptimizer': optimizer})
        model.compile(optimizer='NewtonOptimizer', loss='binary_crossentropy')
    
        # Simulieren eines großen Datensatzes
        x_train = np.random.rand(10000, 20)
        y_train = np.random.randint(2, size=(10000, 1))
    
        try:
            model.fit(x_train, y_train, epochs=1, batch_size=1000, verbose=0)
            handled = True
        except Exception as e:
            handled = False
        
        self.assertTrue(handled, "Optimizer sollte große Datensätze ohne Fehler verarbeiten können.")

    def test_numerical_stability(self):
        """Überprüft die numerische Stabilität des Optimizers."""
        model = Sequential([
            Dense(5, input_shape=(10,), activation='relu', kernel_initializer='random_normal'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = NewtonOptimizer(subsampling_rate=0.5)
        get_custom_objects().update({'NewtonOptimizer': optimizer})
        model.compile(optimizer='NewtonOptimizer', loss='binary_crossentropy')
    
        # Daten, die potenziell numerische Instabilitäten provozieren könnten
        x_train = np.random.rand(100, 10) * 1e-4
        y_train = np.random.randint(2, size=(100, 1))
    
        history = model.fit(x_train, y_train, epochs=5, verbose=0)
    
        final_loss = history.history['loss'][-1]
        self.assertFalse(np.isnan(final_loss) or np.isinf(final_loss), "Verlust sollte numerisch stabil sein (kein NaN oder Inf).")


    def test_compatibility_with_different_models(self):
        """Testet die Kompatibilität des Optimizers mit verschiedenen Modellarchitekturen."""
        for activation_function in ['relu', 'sigmoid', 'tanh']:
            with self.subTest(activation=activation_function):
                model = Sequential([
                    Dense(10, input_shape=(10,), activation=activation_function),
                    Dense(1, activation='linear')
                ])
                optimizer = NewtonOptimizer()
                get_custom_objects().update({'NewtonOptimizer': optimizer})
                model.compile(optimizer='NewtonOptimizer', loss='mse')
    
                x_train = np.random.rand(100, 10)
                y_train = np.random.rand(100, 1)
    
                try:
                    model.fit(x_train, y_train, epochs=1, verbose=0)
                    compatible = True
                except Exception as e:
                    compatible = False
    
                self.assertTrue(compatible, f"Optimizer sollte mit {activation_function}-Aktivierung kompatibel sein.")

if __name__ == '__main__':
    unittest.main()
