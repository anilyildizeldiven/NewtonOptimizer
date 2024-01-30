#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:41:34 2023

@author: anilcaneldiven
"""
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/gradient_descent.py
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class NewtonOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=1.0, name="NewtonOptimizer", **kwargs):
        super(NewtonOptimizer, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "accumulator")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        grad_flat = tf.reshape(grad, [-1])
        
        a = var
        v = a.get_shape()
        loop = v.num_elements()
        
        hessian_list = []
        for i in range(loop):
            # Berechnung der zweiten Ableitung für den i-ten Parameter
            second_derivative = tf.gradients(grad_flat[i], var)[0]
            hessian_list.append(tf.reshape(second_derivative, [-1]))
        # Zusammenbau der Hesse-Matrix
        hessian_flat = tf.stack(hessian_list, axis=1)
        
        
        # Reshape des Gradienten und der Hesse-Matrix
        n_params = tf.reduce_prod(var.shape)
        g_vec = tf.reshape(grad_flat, [n_params, 1])
        h_mat = tf.reshape(hessian_flat, [n_params, n_params])

        #tf.print("Hessian Matrix:", h_mat)
        # Durchführung der Hessian-basierten Newton's Methode mit abgeschnittener Hesse-Matrix
        eps = 1e-4
        eye_eps = tf.eye(h_mat.shape[0]) * eps
        update = tf.linalg.solve(h_mat + eye_eps, g_vec)
        update *= self.learning_rate
        # Aktualisierung des Variablenwertes
        var_update = var - tf.reshape(update, var.shape)
        var.assign(var_update)
    
        return var_update

        
    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(NewtonOptimizer, self).get_config()
        config.update({ "learning_rate": self._serialize_hyperparameter("learning_rate")})
        return config
    
    def initialize_weights(self, model):
        for var in model.trainable_variables:
            var.assign(tf.random.uniform(var.shape, -1, 1))
   
