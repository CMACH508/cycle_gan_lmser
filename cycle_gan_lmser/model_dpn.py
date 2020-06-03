"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf

from . import layers

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.d
IMG_HEIGHT = 256
# The width of each image.
IMG_WIDTH = 256
# The number of color channels per image.
IMG_CHANNELS = 3

POOL_SIZE = 50
ngf = 32
ndf = 64

# The amount of model variables: 8416648


def get_outputs(inputs, skip=False, neuron_share=False, cs=1, rot=False):

    images_a = inputs['images_a']
    images_b = inputs['images_b']

    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:

        prob_real_a_is_real = build_discriminator(images_a, "d_A")
        prob_real_b_is_real = build_discriminator(images_b, "d_B")

        fake_b, cycle_a, fake_a, cycle_b = build_generator(images_a, images_b, name="g_AB", skip=skip, neuron_share=neuron_share, cs=cs, rotate=rot)

        prob_fake_a_is_real = build_discriminator(fake_a, "d_A")
        prob_fake_b_is_real = build_discriminator(fake_b, "d_B")

        prob_fake_pool_a_is_real = build_discriminator(fake_pool_a, "d_A")
        prob_fake_pool_b_is_real = build_discriminator(fake_pool_b, "d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_a,
        'cycle_images_b': cycle_b,
        'fake_images_a': fake_a,
        'fake_images_b': fake_b,
    }


# build_conv_layer(inputs, w, b, strides=[1, 2, 2, 1], padding='SAME',
#                      actfun='relu', dropout=False, keep_prob=0.9, name='conv_layer'):

# build_deconv_layer(inputs, w, b, batch_size, height, width, out_channel,
#                        strides=[1, 2, 2, 1], padding='SAME', actfun='relu',
#                        dropout=False, keep_prob=0.9, do_norm=True, name='deconv_layer'):

def build_resnet_block(input_res, w1, b1, w2, b2, name="resnet", padding="REFLECT"):

    with tf.variable_scope(name):
        out_res = tf.pad(input_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.build_conv_layer(out_res, w1, b1, strides=[1, 1, 1, 1],
                                          padding='VALID', name=name + "_l1")

        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.build_conv_layer(out_res, w2, b2, strides=[1, 1, 1, 1],
                                          padding='VALID', name=name + "_l2", actfun='no')

        return tf.nn.relu(out_res + input_res), out_res + input_res


def build_resnet_block_deconv(input_res, w1, b1, w2, b2, name="resnet", padding="REFLECT"):

    with tf.variable_scope(name):
        # out_res = tf.pad(input_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.build_deconv_layer(input_res, w1, b1, batch_size=1, height=64, width=64, out_channel=ngf * 4,
                                            strides=[1, 1, 1, 1], padding='SAME', name=name + "_l1")

        # out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.build_deconv_layer(out_res, w2, b2, batch_size=1, height=64, width=64, out_channel=ngf * 4,
                                            strides=[1, 1, 1, 1], padding='SAME', actfun='no', name=name + "_l2")

        return tf.nn.relu(out_res + input_res), out_res + input_res


def build_generator(input_a, input_b, name="generator", skip=False, neuron_share=False, cs=1, rotate=False):

    with tf.variable_scope(name) as scope:
        ks_7 = 7
        ks_3 = 3
        kr = 3
        padding = "CONSTANT"
        rot = rotate

        # ========== initialize a2b kernel parameters ==========
        k_c1_shape = [ks_7, ks_7, IMG_CHANNELS, ngf]
        w_c1, b_c1 = layers.get_kernel_params(k_c1_shape, 'c1')

        k_c2_shape = [ks_3, ks_3, ngf, ngf * 2]
        w_c2, b_c2 = layers.get_kernel_params(k_c2_shape, 'c2')

        k_c3_shape = [ks_3, ks_3, ngf * 2, ngf * 4]
        w_c3, b_c3 = layers.get_kernel_params(k_c3_shape, 'c3')

        k_r_shape = [kr, kr, ngf * 4, ngf * 4]

        w_r11, b_r11 = layers.get_kernel_params(k_r_shape, 'r11')
        w_r12, b_r12 = layers.get_kernel_params(k_r_shape, 'r12')

        w_r21, b_r21 = layers.get_kernel_params(k_r_shape, 'r21')
        w_r22, b_r22 = layers.get_kernel_params(k_r_shape, 'r22')

        w_r31, b_r31 = layers.get_kernel_params(k_r_shape, 'r31')
        w_r32, b_r32 = layers.get_kernel_params(k_r_shape, 'r32')

        w_r41, b_r41 = layers.get_kernel_params(k_r_shape, 'r41')
        w_r42, b_r42 = layers.get_kernel_params(k_r_shape, 'r42')

        w_r51, b_r51 = layers.get_kernel_params(k_r_shape, 'r51')
        w_r52, b_r52 = layers.get_kernel_params(k_r_shape, 'r52')

        w_r61, b_r61 = layers.get_kernel_params(k_r_shape, 'r61')
        w_r62, b_r62 = layers.get_kernel_params(k_r_shape, 'r62')

        w_r71, b_r71 = layers.get_kernel_params(k_r_shape, 'r71')
        w_r72, b_r72 = layers.get_kernel_params(k_r_shape, 'r72')

        w_r81, b_r81 = layers.get_kernel_params(k_r_shape, 'r81')
        w_r82, b_r82 = layers.get_kernel_params(k_r_shape, 'r82')

        w_r91, b_r91 = layers.get_kernel_params(k_r_shape, 'r91')
        w_r92, b_r92 = layers.get_kernel_params(k_r_shape, 'r92')

        k_c4_shape = [ks_3, ks_3, ngf * 2, ngf * 4]
        w_c4, b_c4 = layers.get_kernel_params(k_c4_shape, 'c4', False)

        k_c5_shape = [ks_3, ks_3, ngf, ngf * 2]
        w_c5, b_c5 = layers.get_kernel_params(k_c5_shape, 'c5', False)

        k_c6_shape = [ks_7, ks_7, ngf, IMG_CHANNELS]
        w_c6, b_c6 = layers.get_kernel_params(k_c6_shape, 'c6')

        # ========== build generator a2b model ==========
        pad_input_0 = tf.pad(input_a, [[0, 0], [ks_3, ks_3], [ks_3, ks_3], [0, 0]], padding)

        o_c1, o_c1_u = layers.build_conv_layer_act(pad_input_0, w_c1, b_c1, strides=[1, 1, 1, 1], padding='VALID', name="c1")

        o_c2, o_c2_u = layers.build_conv_layer_act(o_c1, w_c2, b_c2, name="c2")

        o_c3, o_c3_u = layers.build_conv_layer_act(o_c2, w_c3, b_c3, name="c3")

        o_r1, o_r1_u = build_resnet_block(o_c3, w_r11, b_r11, w_r12, b_r12, name="r1", padding=padding)

        o_r2, o_r2_u = build_resnet_block(o_r1, w_r21, b_r21, w_r22, b_r22, name="r2", padding=padding)

        o_r3, o_r3_u = build_resnet_block(o_r2, w_r31, b_r31, w_r32, b_r32, name="r3", padding=padding)

        o_r4, o_r4_u = build_resnet_block(o_r3, w_r41, b_r41, w_r42, b_r42, name="r4", padding=padding)

        o_r5, o_r5_u = build_resnet_block(o_r4, w_r51, b_r51, w_r52, b_r52, name="r5", padding=padding)

        o_r6, o_r6_u = build_resnet_block(o_r5, w_r61, b_r61, w_r62, b_r62, name="r6", padding=padding)

        o_r7, o_r7_u = build_resnet_block(o_r6, w_r71, b_r71, w_r72, b_r72, name="r7", padding=padding)

        o_r8, o_r8_u = build_resnet_block(o_r7, w_r81, b_r81, w_r82, b_r82, name="r8", padding=padding)

        o_r9, o_r9_u = build_resnet_block(o_r8, w_r91, b_r91, w_r92, b_r92, name="r9", padding=padding)

        o_c4, o_c4_u = layers.build_deconv_layer_act(o_r9, w_c4, b_c4, batch_size=1, height=64, width=64, out_channel=ngf * 2, name="c4")

        o_c5, o_c5_u = layers.build_deconv_layer_act(o_c4, w_c5, b_c5, batch_size=1, height=128, width=128, out_channel=ngf, name="c5")

        o_c6, o_c6_u = layers.build_conv_layer_act(o_c5, w_c6, b_c6, strides=[1, 1, 1, 1], padding='SAME', do_norm=False, actfun='tanh', name="c6")

        fake_b = o_c6

        # ========== initialize b2a kernel parameters ==========
        k_c1_shape = [ks_7, ks_7, ngf, IMG_CHANNELS]
        w_b_c1, b_b_c1 = layers.get_kernel_params(k_c1_shape, 'c1_b')

        b_b_c2 = layers.get_bias(k_c2_shape, 'c2_b', False)

        b_b_c3 = layers.get_bias(k_c3_shape, 'c3_b', False)

        b_b_r11 = layers.get_bias(k_r_shape, 'r11_b')
        b_b_r12 = layers.get_bias(k_r_shape, 'r12_b')

        b_b_r21 = layers.get_bias(k_r_shape, 'r21_b')
        b_b_r22 = layers.get_bias(k_r_shape, 'r22_b')

        b_b_r31 = layers.get_bias(k_r_shape, 'r31_b')
        b_b_r32 = layers.get_bias(k_r_shape, 'r32_b')

        b_b_r41 = layers.get_bias(k_r_shape, 'r41_b')
        b_b_r42 = layers.get_bias(k_r_shape, 'r42_b')

        b_b_r51 = layers.get_bias(k_r_shape, 'r51_b')
        b_b_r52 = layers.get_bias(k_r_shape, 'r52_b')

        b_b_r61 = layers.get_bias(k_r_shape, 'r61_b')
        b_b_r62 = layers.get_bias(k_r_shape, 'r62_b')

        b_b_r71 = layers.get_bias(k_r_shape, 'r71_b')
        b_b_r72 = layers.get_bias(k_r_shape, 'r72_b')

        b_b_r81 = layers.get_bias(k_r_shape, 'r81_b')
        b_b_r82 = layers.get_bias(k_r_shape, 'r82_b')

        b_b_r91 = layers.get_bias(k_r_shape, 'r91_b')
        b_b_r92 = layers.get_bias(k_r_shape, 'r92_b')

        b_b_c4 = layers.get_bias(k_c4_shape, 'c4_b')

        b_b_c5 = layers.get_bias(k_c5_shape, 'c5_b')

        k_c6_shape = [ks_7, ks_7, IMG_CHANNELS, ngf]
        w_b_c6, b_b_c6 = layers.get_kernel_params(k_c6_shape, 'c6_b')

        # ========== build generator b2a model ==========
        pad_input_1 = tf.pad(fake_b, [[0, 0], [ks_3, ks_3], [ks_3, ks_3], [0, 0]], padding)

        o_c6_b, o_c6_b_u = layers.build_conv_layer_act(pad_input_1, layers.rot(w_b_c6, rot), b_b_c6, strides=[1, 1, 1, 1], padding='VALID', name="c6_b")

        if neuron_share:
            o_c6_b = tf.nn.relu(o_c6_b_u + cs * o_c5_u)

        o_c5_b, o_c5_b_u = layers.build_conv_layer_act(o_c6_b, layers.rot(w_c5, rot), b_b_c5, name="c5_b")

        if neuron_share:
            o_c5_b = tf.nn.relu(o_c5_b_u + cs * o_c4_u)

        o_c4_b, o_c4_b_u = layers.build_conv_layer_act(o_c5_b, layers.rot(w_c4, rot), b_b_c4, name="c4_b")

        if neuron_share:
            o_c4_b = tf.nn.relu(o_c4_b_u + cs * o_r9_u)

        o_r9_b, o_r9_b_u = build_resnet_block_deconv(o_c4_b, layers.rot(w_r92, rot), b_b_r92, layers.rot(w_r91, rot), b_b_r91, name="r9_b", padding=padding)

        if neuron_share:
            o_r9_b = tf.nn.relu(o_r9_b_u + cs * o_r8_u)

        o_r8_b, o_r8_b_u = build_resnet_block_deconv(o_r9_b, layers.rot(w_r82, rot), b_b_r82, layers.rot(w_r81, rot), b_b_r81, name="r8_b", padding=padding)

        if neuron_share:
            o_r8_b = tf.nn.relu(o_r8_b_u + cs * o_r7_u)

        o_r7_b, o_r7_b_u = build_resnet_block_deconv(o_r8_b, layers.rot(w_r72, rot), b_b_r72, layers.rot(w_r71, rot), b_b_r71, name="r7_b", padding=padding)

        if neuron_share:
            o_r7_b = tf.nn.relu(o_r7_b_u + cs * o_r6_u)

        o_r6_b, o_r6_b_u = build_resnet_block_deconv(o_r7_b, layers.rot(w_r62, rot), b_b_r62, layers.rot(w_r61, rot), b_b_r61, name="r6_b", padding=padding)

        if neuron_share:
            o_r6_b = tf.nn.relu(o_r6_b_u + cs * o_r5_u)

        o_r5_b, o_r5_b_u = build_resnet_block_deconv(o_r6_b, layers.rot(w_r52, rot), b_b_r52, layers.rot(w_r51, rot), b_b_r51, name="r5_b", padding=padding)

        if neuron_share:
            o_r5_b = tf.nn.relu(o_r5_b_u + cs * o_r4_u)

        o_r4_b, o_r4_b_u = build_resnet_block_deconv(o_r5_b, layers.rot(w_r42, rot), b_b_r42, layers.rot(w_r41, rot), b_b_r41, name="r4_b", padding=padding)

        if neuron_share:
            o_r4_b = tf.nn.relu(o_r4_b_u + cs * o_r3_u)

        o_r3_b, o_r3_b_u = build_resnet_block_deconv(o_r4_b, layers.rot(w_r32, rot), b_b_r32, layers.rot(w_r31, rot), b_b_r31, name="r3_b", padding=padding)

        if neuron_share:
            o_r3_b = tf.nn.relu(o_r3_b_u + cs * o_r2_u)

        o_r2_b, o_r2_b_u = build_resnet_block_deconv(o_r3_b, layers.rot(w_r22, rot), b_b_r22, layers.rot(w_r21, rot), b_b_r21, name="r2_b", padding=padding)

        if neuron_share:
            o_r2_b = tf.nn.relu(o_r2_b_u + cs * o_r1_u)

        o_r1_b, o_r1_b_u = build_resnet_block_deconv(o_r2_b, layers.rot(w_r12, rot), b_b_r12, layers.rot(w_r11, rot), b_b_r11, name="r1_b", padding=padding)

        if neuron_share:
            o_r1_b = tf.nn.relu(o_r1_b_u + cs * o_c3_u)

        o_c3_b, o_c3_b_u = layers.build_deconv_layer_act(o_r1_b, layers.rot(w_c3, rot), b_b_c3, batch_size=1, height=64, width=64, out_channel=ngf * 2, name="c3_b")

        if neuron_share:
            o_c3_b = tf.nn.relu(o_c3_b_u + cs * o_c2_u)

        o_c2_b, o_c2_b_u = layers.build_deconv_layer_act(o_c3_b, layers.rot(w_c2, rot), b_b_c2, batch_size=1, height=128, width=128, out_channel=ngf, name="c2_b")

        if neuron_share:
            o_c2_b = tf.nn.relu(o_c2_b_u + cs * o_c1_u)

        o_c1_b, o_c1_b_u = layers.build_conv_layer_act(o_c2_b, layers.rot(w_b_c1, rot), b_b_c1, strides=[1, 1, 1, 1], do_norm=False, actfun='tanh', name="c1_b")

        cycle_a = o_c1_b

        # ========== reuse generator b2a model ==========
        pad_input_2 = tf.pad(input_b, [[0, 0], [ks_3, ks_3], [ks_3, ks_3], [0, 0]], padding)

        o_c6_d, o_c6_d_u = layers.build_conv_layer_act(pad_input_2, layers.rot(w_b_c6, rot), b_b_c6, strides=[1, 1, 1, 1], padding='VALID', name="c6_b")

        o_c5_d, o_c5_d_u = layers.build_conv_layer_act(o_c6_d, layers.rot(w_c5, rot), b_b_c5, name="c5_b")

        o_c4_d, o_c4_d_u = layers.build_conv_layer_act(o_c5_d, layers.rot(w_c4, rot), b_b_c4, name="c4_b")

        o_r9_d, o_r9_d_u = build_resnet_block_deconv(o_c4_d, layers.rot(w_r92, rot), b_b_r92, layers.rot(w_r91, rot), b_b_r91, name="r9_b", padding=padding)

        o_r8_d, o_r8_d_u = build_resnet_block_deconv(o_r9_d, layers.rot(w_r82, rot), b_b_r82, layers.rot(w_r81, rot), b_b_r81, name="r8_b", padding=padding)

        o_r7_d, o_r7_d_u = build_resnet_block_deconv(o_r8_d, layers.rot(w_r72, rot), b_b_r72, layers.rot(w_r71, rot), b_b_r71, name="r7_b", padding=padding)

        o_r6_d, o_r6_d_u = build_resnet_block_deconv(o_r7_d, layers.rot(w_r62, rot), b_b_r62, layers.rot(w_r61, rot), b_b_r61, name="r6_b", padding=padding)

        o_r5_d, o_r5_d_u = build_resnet_block_deconv(o_r6_d, layers.rot(w_r52, rot), b_b_r52, layers.rot(w_r51, rot), b_b_r51, name="r5_b", padding=padding)

        o_r4_d, o_r4_d_u = build_resnet_block_deconv(o_r5_d, layers.rot(w_r42, rot), b_b_r42, layers.rot(w_r41, rot), b_b_r41, name="r4_b", padding=padding)

        o_r3_d, o_r3_d_u = build_resnet_block_deconv(o_r4_d, layers.rot(w_r32, rot), b_b_r32, layers.rot(w_r31, rot), b_b_r31, name="r3_b", padding=padding)

        o_r2_d, o_r2_d_u = build_resnet_block_deconv(o_r3_d, layers.rot(w_r22, rot), b_b_r22, layers.rot(w_r21, rot), b_b_r21, name="r2_b", padding=padding)

        o_r1_d, o_r1_d_u = build_resnet_block_deconv(o_r2_d, layers.rot(w_r12, rot), b_b_r12, layers.rot(w_r11, rot), b_b_r11, name="r1_b", padding=padding)

        o_c3_d, o_c3_d_u = layers.build_deconv_layer_act(o_r1_d, layers.rot(w_c3, rot), b_b_c3, batch_size=1, height=64, width=64, out_channel=ngf * 2, name="c3_b")

        o_c2_d, o_c2_d_u = layers.build_deconv_layer_act(o_c3_d, layers.rot(w_c2, rot), b_b_c2, batch_size=1, height=128, width=128, out_channel=ngf, name="c2_b")

        o_c1_d, o_c1_d_u = layers.build_conv_layer_act(o_c2_d, layers.rot(w_b_c1, rot), b_b_c1, strides=[1, 1, 1, 1], do_norm=False, actfun='tanh', name="c1_b")

        fake_a = o_c1_d

        # ========== reuse generator a2b model ==========
        pad_input = tf.pad(fake_a, [[0, 0], [ks_3, ks_3], [ks_3, ks_3], [0, 0]], padding)

        o_c1_c, o_c1_c_u = layers.build_conv_layer_act(pad_input, w_c1, b_c1, strides=[1, 1, 1, 1], padding='VALID', name="c1")

        if neuron_share:
            o_c1_c = tf.nn.relu(o_c1_c_u + cs * o_c2_d_u)

        o_c2_c, o_c2_c_u = layers.build_conv_layer_act(o_c1_c, w_c2, b_c2, name="c2")

        if neuron_share:
            o_c2_c = tf.nn.relu(o_c2_c_u + cs * o_c3_d_u)

        o_c3_c, o_c3_c_u = layers.build_conv_layer_act(o_c2_c, w_c3, b_c3, name="c3")

        if neuron_share:
            o_c3_c = tf.nn.relu(o_c3_c_u + cs * o_r1_d_u)

        o_r1_c, o_r1_c_u = build_resnet_block(o_c3_c, w_r11, b_r11, w_r12, b_r12, name="r1", padding=padding)

        if neuron_share:
            o_r1_c = tf.nn.relu(o_r1_c_u + cs * o_r2_d_u)

        o_r2_c, o_r2_c_u = build_resnet_block(o_r1_c, w_r21, b_r21, w_r22, b_r22, name="r2", padding=padding)

        if neuron_share:
            o_r2_c = tf.nn.relu(o_r2_c_u + cs * o_r3_d_u)

        o_r3_c, o_r3_c_u = build_resnet_block(o_r2_c, w_r31, b_r31, w_r32, b_r32, name="r3", padding=padding)

        if neuron_share:
            o_r3_c = tf.nn.relu(o_r3_c_u + cs * o_r4_d_u)

        o_r4_c, o_r4_c_u = build_resnet_block(o_r3_c, w_r41, b_r41, w_r42, b_r42, name="r4", padding=padding)

        if neuron_share:
            o_r4_c = tf.nn.relu(o_r4_c_u + cs * o_r5_d_u)

        o_r5_c, o_r5_c_u = build_resnet_block(o_r4_c, w_r51, b_r51, w_r52, b_r52, name="r5", padding=padding)

        if neuron_share:
            o_r5_c = tf.nn.relu(o_r5_c_u + cs * o_r6_d_u)

        o_r6_c, o_r6_c_u = build_resnet_block(o_r5_c, w_r61, b_r61, w_r62, b_r62, name="r6", padding=padding)

        if neuron_share:
            o_r6_c = tf.nn.relu(o_r6_c_u + cs * o_r7_d_u)

        o_r7_c, o_r7_c_u = build_resnet_block(o_r6_c, w_r71, b_r71, w_r72, b_r72, name="r7", padding=padding)

        if neuron_share:
            o_r7_c = tf.nn.relu(o_r7_c_u + cs * o_r8_d_u)

        o_r8_c, o_r8_c_u = build_resnet_block(o_r7_c, w_r81, b_r81, w_r82, b_r82, name="r8", padding=padding)

        if neuron_share:
            o_r8_c = tf.nn.relu(o_r8_c_u + cs * o_r9_d_u)

        o_r9_c, o_r9_c_u = build_resnet_block(o_r8_c, w_r91, b_r91, w_r92, b_r92, name="r9", padding=padding)

        if neuron_share:
            o_r9_c = tf.nn.relu(o_r9_c_u + cs * o_c4_d_u)

        o_c4_c, o_c4_c_u = layers.build_deconv_layer_act(o_r9_c, w_c4, b_c4, batch_size=1, height=64, width=64, out_channel=ngf * 2, name="c4")

        if neuron_share:
            o_c4_c = tf.nn.relu(o_c4_c_u + cs * o_c5_d_u)

        o_c5_c, o_c5_c_u = layers.build_deconv_layer_act(o_c4_c, w_c5, b_c5, batch_size=1, height=128, width=128, out_channel=ngf, name="c5")

        if neuron_share:
            o_c5_c = tf.nn.relu(o_c5_c_u + cs * o_c6_d_u)

        o_c6_c, o_c6_c_u = layers.build_conv_layer_act(o_c5_c, w_c6, b_c6, strides=[1, 1, 1, 1], padding='SAME', do_norm=False, actfun='tanh', name="c6")

        cycle_b = o_c6_c

        return fake_b, cycle_a, fake_a, cycle_b


def build_discriminator_original(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        padw = 2

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2,
                                     0.02, "VALID", "c1", do_norm=False,
                                     relufactor=0.2)

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "VALID", "c2", relufactor=0.2)

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "VALID", "c3", relufactor=0.2)

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1,
                                     0.02, "VALID", "c4", relufactor=0.2)

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [
            padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(
            pad_o_c4, 1, f, f, 1, 1, 0.02, "VALID", "c5",
            do_norm=False, do_relu=False)

        return o_c5


def build_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        ks_4 = 4
        padw = 2
        padding = "CONSTANT"

        # ========== initialize discriminator kernel parameters ==========
        k_c1_shape = [ks_4, ks_4, IMG_CHANNELS, ndf]
        w_c1, b_c1 = layers.get_kernel_params(k_c1_shape, 'c1')

        k_c2_shape = [ks_4, ks_4, ndf, ndf * 2]
        w_c2, b_c2 = layers.get_kernel_params(k_c2_shape, 'c2')

        k_c3_shape = [ks_4, ks_4, ndf * 2, ndf * 4]
        w_c3, b_c3 = layers.get_kernel_params(k_c3_shape, 'c3')

        k_c4_shape = [ks_4, ks_4, ndf * 4, ndf * 8]
        w_c4, b_c4 = layers.get_kernel_params(k_c4_shape, 'c4')

        k_c5_shape = [ks_4, ks_4, ndf * 8, 1]
        w_c5, b_c5 = layers.get_kernel_params(k_c5_shape, 'c5')

        # ========== build discriminator model ==========

        # def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
        #                    padding="VALID", name="conv2d", do_norm=True, do_relu=True,
        #                    relufactor=0):

        # def build_conv_layer(inputs, w, b, strides=[1, 2, 2, 1], padding='SAME',
        #                      actfun='relu', dropout=False, keep_prob=0.9, do_norm=True, name='conv_layer'):

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], padding)
        o_c1 = layers.build_conv_layer(pad_input, w_c1, b_c1, padding='VALID', actfun='lrelu', do_norm=False, name="c1")

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], padding)
        o_c2 = layers.build_conv_layer(pad_o_c1, w_c2, b_c2, padding='VALID', actfun='lrelu', name="c2")

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], padding)
        o_c3 = layers.build_conv_layer(pad_o_c2, w_c3, b_c3, padding='VALID', actfun='lrelu', name="c3")

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], padding)
        o_c4 = layers.build_conv_layer(pad_o_c3, w_c4, b_c4, strides=[1, 1, 1, 1], padding='VALID', actfun='lrelu', name="c4")

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], padding)
        o_c5 = layers.build_conv_layer(pad_o_c4, w_c5, b_c5, strides=[1, 1, 1, 1], padding='VALID', actfun='no', do_norm=False, name="c5")

        return o_c5


def patch_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 3])
        o_c1 = layers.general_conv2d(patch_input, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm="False",
                                     relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, ndf * 8, f, f, 2, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False,
            do_relu=False)

        return o_c5



