import tensorflow as tf


def logistic_regression_map(x_):
    scope_args = {'initializer': tf.random_normal_initializer(stddev=1e-4)}
    with tf.variable_scope("weights", **scope_args):
        flattered = tf.contrib.layers.flatten(x_, scope='pool2flat')
        W = tf.get_variable('W', shape=[3072, 10])
        b = tf.get_variable('b', shape=[10])
        y_logits = tf.matmul(flattered, W) + b
    return y_logits


def cnn_map(x_):
    conv1 = tf.layers.conv2d(
            inputs=x_,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2, 2], 
                                    strides=2)
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                    pool_size=[2, 2], 
                                    strides=2)
    pool_flat = tf.contrib.layers.flatten(pool2, scope='pool2flat')
    dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)
    return logits


def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
            y_ = tf.placeholder(tf.int32, [None])
            y_logits = model_function(x_)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            trainer = tf.train.AdamOptimizer()
            train_op = trainer.minimize(cross_entropy_loss)
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), dimension=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    
    return model_dict
