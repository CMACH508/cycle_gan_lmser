import tensorflow as tf


def select_act_func(actfun):
    if actfun == 'tanh':
        return tf.nn.tanh
    elif actfun == 'sigmoid':
        return tf.sigmoid
    elif actfun == 'relu':
        return tf.nn.relu
    elif actfun == 'lrelu':
        return lrelu
    else:
        return lambda x: x


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def batchnorm(inputs):
    with tf.variable_scope("batchnorm"):
        variance_epsilon = 1e-5
        mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
        normalized = tf.nn.batch_normalization(
            inputs, mean, variance, None, None, variance_epsilon=variance_epsilon, name='batchnorm_op')
        return normalized


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
            # updates_collections=None, epsilon=1e-5, scale=True,
            # scope="batch_norm")

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def get_filter(name, shape):
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))


def get_bias(shape, name, conv=True):
    if conv:
        return tf.get_variable(name + '_b_conv', shape[-1], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
    else:
        return tf.get_variable(name + '_b_deconv', shape[-2], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))


def get_kernel_params(shape, name, conv=True, bias=True):
        w_c1 = tf.get_variable(name + '_w', shape, dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        if bias is False:
            return w_c1
        if conv:
            b_c1 = tf.get_variable(name + '_b_conv', [shape[-1]], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
        else:
            b_c1 = tf.get_variable(name + '_b_deconv', [shape[-2]], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
        return w_c1, b_c1


def build_conv_layer(inputs, w, b, strides=[1, 2, 2, 1], padding='SAME',
                     actfun='relu', dropout=False, keep_prob=0.9, do_norm=True, name='conv_layer'):

    with tf.variable_scope(name):
        f = select_act_func(actfun)
        conv = tf.nn.conv2d(inputs, w, strides=strides, padding=padding)
        conv = conv + b
        if do_norm:
            conv = instance_norm(conv)
        if dropout:
            conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        out = f(conv)

    return out


def build_conv_layer_act(inputs, w, b, strides=[1, 2, 2, 1], padding='SAME',
                     actfun='relu', dropout=False, keep_prob=0.9, do_norm=True, name='conv_layer'):

    with tf.variable_scope(name):
        f = select_act_func(actfun)
        conv = tf.nn.conv2d(inputs, w, strides=strides, padding=padding)
        conv = conv + b
        if do_norm:
            conv = instance_norm(conv)
        if dropout:
            conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        out = f(conv)

    return out, conv


def build_conv_layer_nb(inputs, w, strides=[1, 2, 2, 1], padding='SAME',
                     actfun='relu', dropout=False, keep_prob=0.9, do_norm=True, name='conv_layer'):

    with tf.variable_scope(name):
        f = select_act_func(actfun)
        conv = tf.nn.conv2d(inputs, w, strides=strides, padding=padding)
        # conv = conv + b
        if do_norm:
            conv = instance_norm(conv)
        if dropout:
            conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        out = f(conv)

    return out


def build_deconv_layer(inputs, w, b, batch_size, height, width, out_channel,
                       strides=[1, 2, 2, 1], padding='SAME', actfun='relu',
                       dropout=False, keep_prob=0.9, do_norm=True, name='deconv_layer'):

    with tf.variable_scope(name):
        if strides[1] == 1:
            deconv_shape = [batch_size, height, width, out_channel]
        else:
            deconv_shape = [batch_size, height * 2, width * 2, out_channel]
        f = select_act_func(actfun)
        conv = tf.nn.conv2d_transpose(inputs, w, deconv_shape, strides=strides, padding=padding)
        conv = conv + b
        if do_norm:
            conv = instance_norm(conv)
        if dropout:
            conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        out = f(conv)

    return out


def build_deconv_layer_act(inputs, w, b, batch_size, height, width, out_channel,
                       strides=[1, 2, 2, 1], padding='SAME', actfun='relu',
                       dropout=False, keep_prob=0.9, do_norm=True, name='deconv_layer'):

    with tf.variable_scope(name):
        if strides[1] == 1:
            deconv_shape = [batch_size, height, width, out_channel]
        else:
            deconv_shape = [batch_size, height * 2, width * 2, out_channel]
        f = select_act_func(actfun)
        conv = tf.nn.conv2d_transpose(inputs, w, deconv_shape, strides=strides, padding=padding)
        conv = conv + b
        if do_norm:
            conv = instance_norm(conv)
        if dropout:
            conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        out = f(conv)

    return out, conv


def build_deconv_layer_nb(inputs, w, batch_size, height, width, out_channel,
                       strides=[1, 2, 2, 1], padding='SAME', actfun='relu',
                       dropout=False, keep_prob=0.9, do_norm=True, name='deconv_layer'):

    with tf.variable_scope(name):
        if strides[1] == 1:
            deconv_shape = [batch_size, height, width, out_channel]
        else:
            deconv_shape = [batch_size, height * 2, width * 2, out_channel]
        f = select_act_func(actfun)
        conv = tf.nn.conv2d_transpose(inputs, w, deconv_shape, strides=strides, padding=padding)
        # conv = conv + b
        if do_norm:
            conv = instance_norm(conv)
        if dropout:
            conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        out = f(conv)

    return out


def calculate_vars(vars):
    vars_count = 0
    for var in vars:
        shape = var.get_shape()
        mul = 1
        for dim in shape:
            mul *= dim.value
        vars_count += mul
    return vars_count


def rot(tensor, rot=True):
    if rot:
        tensor_tr = tf.transpose(tensor, perm=[2, 0, 1, 3])
        tensor_rot = tf.image.rot90(tensor_tr, k=2)
        tensor_fi = tf.transpose(tensor_rot, perm=[1, 2, 0, 3])
        return tensor_fi
    else:
        return tensor


def get_mask_params(shape, name):
    param_shape = [shape[0], shape[1], 1]
    mask = tf.get_variable(name, param_shape, dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    mask_tile = tf.tile(mask, [1, 1, shape[2]])
    return mask_tile