import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.keras.utils import get_custom_objects
import numpy as np

class NewtonOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, name="NewtonOptimizer", subsampling_rate=0.9, **kwargs):
        super(NewtonOptimizer, self).__init__(name, **kwargs)
        self.subsampling_rate = subsampling_rate 

    def _resource_apply_dense(self, grad, var, apply_state=None):
        grad_flat = tf.reshape(grad, [-1])
        loop = var.shape.num_elements()
        
        min_subsample_size = 1
        subsample_size = max(int(loop * self.subsampling_rate), min_subsample_size)
        subsample_indices = tf.random.shuffle(tf.range(loop))[:subsample_size]
        subsample_indices = tf.sort(subsample_indices, direction='ASCENDING')
        grad_flat = tf.gather(grad_flat, subsample_indices)
        
        loop_1 = subsample_indices.shape.num_elements()
        hessian_list = []
        for i in range(loop_1):
            second_derivative = tf.gradients(grad_flat[i], var)[0]
            hessian_list.append(tf.reshape(second_derivative, [-1]))
        hessian_flat = tf.stack(hessian_list, axis=1)
        hessian_filtered = tf.gather(hessian_flat, subsample_indices)
        
        n_params = tf.reduce_prod(grad_flat.shape)
        g_vec = tf.reshape(grad_flat, [n_params, 1])
        h_mat = tf.reshape(hessian_filtered, [n_params, n_params])
        
        # Verstärkte Regularisierung
        eps = 1e-2  # Erhöhung von eps zur Verbesserung der Invertierbarkeit
        eye_eps = tf.eye(h_mat.shape[0]) * eps
        
        # Überprüfung und Anpassung der Lösungsmethode
        try:
            update_filtered = tf.linalg.solve(h_mat + eye_eps, g_vec)
        except tf.errors.InvalidArgumentError:  # Falls solve fehlschlägt
            # Alternativer Ansatz: Nutzung der Pseudo-Inversen für nicht-quadratische Matrizen
            pseudo_inverse = tf.linalg.pinv(h_mat + eye_eps)
            update_filtered = tf.matmul(pseudo_inverse, g_vec)
        
        full_update = tf.scatter_nd(tf.reshape(subsample_indices, [-1, 1]), update_filtered, [loop, 1])
        var_update = var - tf.reshape(full_update, var.shape)
        var.assign(var_update)
        return var_update
        
    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(NewtonOptimizer, self).get_config()
        config.update({"subsampling_rate": self.subsampling_rate})
        return config
