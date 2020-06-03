"""Code for training CycleGAN."""
from datetime import datetime
import json
import numpy as np
import os
import random
from scipy.misc import imsave
import time
import click
import tensorflow as tf
import csv

from . import cyclegan_datasets
from . import data_loader, losses, layers
# from . import model_dpn as model
# from . import model_dpn_V2 as model
from . import model_dpn as model

slim = tf.contrib.slim


class CycleGANLmser:
    """The CycleGAN module."""

    def __init__(self, pool_size, lambda_a, lambda_b, output_root_dir, to_restore, base_lr, max_step,
                 dataset_name, checkpoint_dir, do_flipping, skip, neuron_share, cs, rot, lambda_d_a, lambda_d_b):

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._pool_size = pool_size
        self._size_before_crop = 286
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 30
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip
        self._neuron_share = neuron_share
        self._cs = cs
        self._rot = rot
        self._lambda_d_a = lambda_d_a
        self._lambda_d_b = lambda_d_b

        self.fake_images_A = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS)
        )

    def model_setup(self):

        self.input_a = tf.placeholder(tf.float32,
                                      [1, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS],
                                      name="input_A")
        self.input_b = tf.placeholder(tf.float32,
                                      [1, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS],
                                      name="input_B")

        self.fake_pool_A = tf.placeholder(tf.float32,
                                          [None, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS],
                                          name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32,
                                          [None, model.IMG_WIDTH, model.IMG_HEIGHT, model.IMG_CHANNELS],
                                          name="fake_pool_B")

        self.global_step = slim.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        inputs = {
            'images_a': self.input_a, 'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A, 'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(inputs, skip=self._skip, neuron_share=self._neuron_share, cs=self._cs, rot=self._rot)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

    def compute_losses(self):

        self.cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=self.input_a, generated_images=self.cycle_images_a,
            )
        self.cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=self.input_b, generated_images=self.cycle_images_b,
            )

        self.lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        self.lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)

        self.g_loss_A = self.cycle_consistency_loss_a + self.cycle_consistency_loss_b + self.lsgan_loss_b
        self.g_loss_B = self.cycle_consistency_loss_b + self.cycle_consistency_loss_a + self.lsgan_loss_a

        self.d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
            r=self._lambda_d_a
        )
        self.d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
            r=self._lambda_d_b
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.trainable_variables()
        self.vars_count = layers.calculate_vars(self.model_vars)

        print '************'
        print 'The amount of model_variables:', self.vars_count
        print '************'

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_vars = [var for var in self.model_vars if 'g_AB' in var.name]

        self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(self.g_loss_A, var_list=g_vars)
        self.g_B_trainer = optimizer.minimize(self.g_loss_B, var_list=g_vars)

        # d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        # g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        # d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        # g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        #
        # self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_A_vars)
        # self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_B_vars)
        # self.g_A_trainer = optimizer.minimize(self.g_loss_A, var_list=g_A_vars)
        # self.g_B_trainer = optimizer.minimize(self.g_loss_B, var_list=g_B_vars)

        # for var in self.model_vars:
        #     print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", self.g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", self.g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", self.d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", self.d_loss_B)

    def save_images(self, sess, epoch):

        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['_real_A', '_real_B', '_fake_B',
                 '_fake_A', '_rec_A', '_rec_B']

        with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                inputs = sess.run(self.inputs)
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                    [self.fake_images_a, self.fake_images_b, self.cycle_images_a,self.cycle_images_b],
                    feed_dict={self.input_a: inputs['images_i'], self.input_b: inputs['images_j']})

                tensors = [inputs['images_i'], inputs['images_j'],
                           fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = str(epoch) + "_" + str(i) + name + ".png"
                    imsave(os.path.join(self._images_dir, image_name),
                           ((tensor[0] + 1) * 127.5).astype(np.uint8))
                    v_html.write("<img src=\"" + os.path.join('imgs', image_name) + "\">")
                v_html.write("<br>")

    def fake_image_pool(self, num_fakes, fake, fake_pool):

        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder
        self.inputs = data_loader.load_data(self._dataset_name, self._size_before_crop, True, self._do_flipping, True)

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver()

        max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]

        with tf.Session() as sess:
            start_time = time.time()
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                if epoch > 190:
                    saver.save(sess, os.path.join(self._output_dir, "cyclegan.ckpt"), global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if epoch < 100:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - self._base_lr * (epoch - 100) / 100
                    # curr_lr = self._base_lr * 0.9 ** ((epoch - 100) / 10)

                self.save_images(sess, epoch)

                for i in range(0, max_images):

                    inputs = sess.run(self.inputs)

                    # Optimizing the G_A network
                    # g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
                    _, fake_B_temp, summary_str, c_c_loss_a, g_loss_b \
                        = sess.run(
                        [self.g_A_trainer,
                         self.fake_images_b,
                         self.g_A_loss_summ,
                         self.cycle_consistency_loss_a,
                         self.lsgan_loss_b],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr})
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network once
                    _, summary_str, d_loss_b = sess.run([self.d_B_trainer, self.d_B_loss_summ, self.d_loss_B],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1})
                    writer.add_summary(summary_str, epoch * max_images + i)

                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str, c_c_loss_b, g_loss_a \
                        = sess.run(
                        [self.g_B_trainer,
                         self.fake_images_a,
                         self.g_B_loss_summ,
                         self.cycle_consistency_loss_b,
                         self.lsgan_loss_a
                         ],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr})
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network once
                    _, summary_str, d_loss_a = sess.run([self.d_A_trainer, self.d_A_loss_summ, self.d_loss_A],
                        feed_dict={
                            self.input_a: inputs['images_i'],
                            self.input_b: inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1})
                    writer.add_summary(summary_str, epoch * max_images + i)

                    # Call this method to make sure that all pending events have been written to disk.
                    writer.flush()
                    self.num_fake_inputs += 1
                    if i % 50 == 0:
                        print "Training epoch:", epoch, "batch:", i
                        print "G_A loss (fake B):", g_loss_b
                        print "G_B loss (fake A):", g_loss_a
                        print "D_A loss (fake A):", d_loss_a
                        print "D_B loss (fake B):", d_loss_b
                        print "c_c_A loss:", c_c_loss_a
                        print "c_c_B loss:", c_c_loss_b
                        print "time:", time.time() - start_time
                        print " "

                    # if epoch == 0:
                    #     print i
                    #     self.save_image_debug(fake_B_temp, 'fake_b_' + str(epoch) + '_' + str(i))
                    #     self.save_image_debug(fake_A_temp, 'fake_a_' + str(epoch) + '_' + str(i))

                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)

    def save_image_debug(self, image, image_id):
            image_name = image_id + ".png"
            image_debug_dir = self._images_dir + '/debug'
            if not os.path.exists(image_debug_dir):
                os.makedirs(image_debug_dir)
            imsave(os.path.join(image_debug_dir, image_name), ((image[0] + 1) * 127.5).astype(np.uint8))



    def test(self):
        """Test Function."""
        print("Testing the results")

        self.inputs = data_loader.load_data(
            self._dataset_name, self._size_before_crop, False, False, False)

        self.model_setup()
        saver = tf.train.Saver()  
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            saver.restore(sess, chkpt_fname)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self._num_imgs_to_save = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]
            self.save_images(sess, 0)

            coord.request_stop()
            coord.join(threads)


@click.command()
@click.option('--to_train',
              type=click.INT,
              default=True,
              help='Whether it is train or false.')
@click.option('--log_dir',
              type=click.STRING,
              default=None,
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='train',
              help='The name of the configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='',
              help='The name of the train/test split.')
def main(to_train, log_dir, config_filename, checkpoint_dir):

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    lambda_d_a = float(config['LAMBDA_D_A']) if 'LAMBDA_D_A' in config else 0.25
    lambda_d_b = float(config['LAMBDA_D_B']) if 'LAMBDA_D_B' in config else 0.25
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    dataset_name = str(config['dataset_name'])
    do_flipping = bool(config['do_flipping'])
    skip = bool(config['skip'])
    neuron_share = bool(config['neuron_share'])
    cs = bool(config['cs']) if 'cs' in config else 1
    rot = bool(config['rotate'])

    cycle_gan_lmser_model = CycleGANLmser(
        pool_size, lambda_a, lambda_b, log_dir, to_restore, base_lr, max_step, dataset_name,
        checkpoint_dir, do_flipping, skip, neuron_share, cs, rot, lambda_d_a, lambda_d_b)

    if to_train > 0:
        cycle_gan_lmser_model.train()
    else:
        cycle_gan_lmser_model.test()


if __name__ == '__main__':
    main()
