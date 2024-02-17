import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import get_custom_objects
import numpy as np
# import matplotlib.pyplot as plt


class NewtonOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, name="NewtonOptimizer", subsampling_rate=0.4, **kwargs):
        super(NewtonOptimizer, self).__init__(name, **kwargs)
        self.subsampling_rate = subsampling_rate 


    def _resource_apply_dense(self, grad, var, apply_state=None):
        # tf.print("var:",var_list.type)
        grad_flat = tf.reshape(grad, [-1])
        # tf.print("gradshape:",grad_flat.shape)
        loop = var.shape.num_elements()
        #tf.print("gradshape:",grad_flat.shape)
        
        
        min_subsample_size = 1
        
        
        subsample_size = max(int(loop * self.subsampling_rate), min_subsample_size)
        # tf.print("subsample_size:",subsample_size)
        subsample_indices = tf.random.shuffle(tf.range(loop))[:subsample_size]
        # tf.print("subsample_indices:",subsample_indices, subsample_indices.shape)
        subsample_indices = tf.sort(subsample_indices, direction='ASCENDING')    



        grad_flat = tf.gather(grad_flat, subsample_indices)
        # tf.print("grad_flat:",grad_flat, grad_flat.shape)
        
        
        loop_1 = subsample_indices.shape.num_elements()
        # tf.print("subsample_indices:",subsample_indices)
        
        hessian_list = []
        for i in range(loop_1):
            # tf.print("i:",i)
            # Berechnung der zweiten Ableitung für den i-ten Parameter
            second_derivative = tf.gradients(grad_flat[i], var)[0]
            # tf.print("second_derivative:",second_derivative)
            hessian_list.append(tf.reshape(second_derivative, [-1]))
            #tf.print("hessian_list:",hessian_list,hessian_list.shape)
        # Zusammenbau der Hesse-Matrix
        hessian_flat = tf.stack(hessian_list, axis=1)
        # tf.print("hessian_flat:",hessian_flat.shape)
        # Filtern der Hesse-Matrix, um nur relevante Variablen zu behalten
        
        
        # hessian_filtered = tf.gather(hessian_flat, subsample_indices, axis=1)
        # tf.print("hessian_filtered :",hessian_filtered .shape)
        hessian_filtered = tf.gather(hessian_flat, subsample_indices)
        # tf.print("hessian_filtered :",hessian_filtered.shape)
        
        
        # Reshape des Gradienten und der Hesse-Matrix
        n_params = tf.reduce_prod(grad_flat.shape)
        
        g_vec = tf.reshape(grad_flat, [n_params, 1])
        # tf.print("g_vec:",g_vec, g_vec.shape)
        # tf.print("grad_flat:",grad_flat)
        
        h_mat = tf.reshape(hessian_filtered, [n_params, n_params])
        # tf.print("h_mat:",h_mat.shape)
        
        # Durchführung der Hessian-basierten Newton's Methode mit abgeschnittener Hesse-Matrix
        eps = 1e-4
        eye_eps = tf.eye(h_mat.shape[0]) * eps

        # hesse inverse auch in update_filtered berechnet?
        #inv_hessian = tf.linalg.inv(flat_hessians + eye_eps)
        update_filtered = tf.linalg.solve(h_mat + eye_eps, g_vec)
        
        full_update = tf.scatter_nd(tf.reshape(subsample_indices, [-1, 1]), update_filtered, [loop, 1])
        # tf.print("h_mat:",full_update.shape,full_update)
        
        var_update = var - tf.reshape(full_update, var.shape)
        
        var.assign(var_update)
    
        # return var_update
        return var_update
        
        
    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(NewtonOptimizer, self).get_config()
        config.update({ "subsampling_rate": self.subsampling_rate})
        return config

          
            

            
