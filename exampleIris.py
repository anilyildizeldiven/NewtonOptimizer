#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:02:59 2024

@author: anilcaneldiven
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import get_custom_objects
# Annahme: NewtonOptimizer ist deine benutzerdefinierte Optimiererklasse
from customOptimizer import NewtonOptimizer  # Stelle sicher, dass du den korrekten Pfad ersetzt
import matplotlib.pyplot as plt
# Registriere deinen benutzerdefinierten Optimierer in get_custom_objects
get_custom_objects().update({'NewtonOptimizer': NewtonOptimizer})
from sklearn.datasets import load_iris
from tensorflow.keras.utils import get_custom_objects
# Dein importierter benutzerdefinierter Optimierer
from customOptimizer import NewtonOptimizer

# Registriere den benutzerdefinierten Optimierer
get_custom_objects().update({'NewtonOptimizer': NewtonOptimizer})

# Iris-Daten laden
iris = load_iris()
X = iris.data  # Features
y = iris.target.reshape(-1, 1)  # Zielwerte, hier nur als Beispiel, evtl. Anpassung nötig

# Ein einfaches Sequential-Modell erstellen
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='tanh'))  # Anpassung an 4 Features
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

# NewtonOptimizer initialisieren
optimizer = NewtonOptimizer()

# Model kompilieren
model.compile(optimizer='NewtonOptimizer', loss='mse')

# Modell trainieren, mit Trainings- und Validierungsdaten
history = model.fit(X, y, batch_size=10, epochs=100, validation_split=0.2)

# Plot Loss-Verlauf
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
