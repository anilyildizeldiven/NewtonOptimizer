import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

class NewtonOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, name="NewtonOptimizer", subsampling_rate=0.4, **kwargs):
        super(NewtonOptimizer, self).__init__(name, **kwargs)
        self.subsampling_rate = subsampling_rate 


    def _resource_apply_dense(self, grad, var, apply_state=None):
        grad_flat = tf.reshape(grad, [-1])
        loop = var.shape.num_elements()

        min_subsample_size = 1
        subsample_size = max(int(loop * self.subsampling_rate), min_subsample_size)

        # Shuffle and select subsample indices
        subsample_indices = tf.random.shuffle(tf.range(loop))[:subsample_size]
        subsample_indices = tf.sort(subsample_indices, direction='ASCENDING')    

        # Gather gradients for subsampled indices
        grad_flat_subsampled = tf.gather(grad_flat, subsample_indices)
        
        # Calculate Hessian for subsampled variables
        hessian_list = []
        for i in range(subsample_size):
            second_derivative = tf.gradients(grad_flat_subsampled[i], var)[0]
            hessian_list.append(tf.reshape(second_derivative, [-1]))
        
        hessian_flat = tf.stack(hessian_list, axis=1)
        hessian_filtered = tf.gather(hessian_flat, subsample_indices)
        
        n_params = tf.reduce_prod(grad_flat_subsampled.shape)
        g_vec = tf.reshape(grad_flat_subsampled, [n_params, 1])
        h_mat = tf.reshape(hessian_filtered, [n_params, n_params])
        
        # Compute average Hessian for approximation
        average_hessian = tf.reduce_mean(h_mat)
        
        # Newton's method for subsampled variables
        eps = 1e-4
        eye_eps = tf.eye(n_params) * eps
        try:
            update_filtered = tf.linalg.solve(h_mat + eye_eps, g_vec)
        except tf.errors.InvalidArgumentError:  # Falls solve fehlschlägt
            # Alternativer Ansatz: Nutzung der Pseudo-Inversen für nicht-quadratische Matrizen
            pseudo_inverse = tf.linalg.pinv(h_mat + eye_eps)
            update_filtered = tf.matmul(pseudo_inverse, g_vec)
        
        # Prepare full update for subsampled variables
        full_update_subsampled = tf.scatter_nd(tf.reshape(subsample_indices, [-1, 1]), update_filtered, [loop, 1])
        
        # Approximate update for non-sampled variables
        # Calculate difference indices (non-sampled indices)
        all_indices = tf.range(loop)
        difference_indices = tf.sets.difference(tf.expand_dims(all_indices, 0), tf.expand_dims(subsample_indices, 0))
        difference_indices = tf.reshape(difference_indices.values, [-1])
        
        # Gather gradients for non-sampled variables
        grad_flat_non_sampled = tf.gather(grad_flat, difference_indices)
        
        # Use inverse of average Hessian for approximation
        inv_average_hessian = 1 / (average_hessian + eps)
        update_non_sampled = grad_flat_non_sampled * inv_average_hessian
        
        # Ensure update_non_sampled is properly reshaped to match the required dimensions for tf.scatter_nd
        update_non_sampled_reshaped = tf.reshape(update_non_sampled, [-1, 1])  # Reshape to ensure it has a second dimension
        
        # Prepare update for non-sampled variables
        full_update_non_sampled = tf.scatter_nd(tf.reshape(difference_indices, [-1, 1]), update_non_sampled_reshaped, [loop, 1])
        
        # Combine updates
        combined_update = full_update_subsampled + full_update_non_sampled
        
        # Apply update to variable
        var_update = var - tf.reshape(combined_update, var.shape)
        var.assign(var_update)
        
        return var_update
        
        
    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(NewtonOptimizer, self).get_config()
        config.update({ "subsampling_rate": self.subsampling_rate})
        return config

    def initialize_weights(self, model):
        for var in model.trainable_variables:
            var.assign(tf.random.uniform(var.shape, -1, 1))
