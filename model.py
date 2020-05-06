#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorlayer.layers import *
import tensorlayer as tl
import tensorflow as tf

def RIR_block(x, nb_filters, w_init, b_init, is_train, gamma_init, name='RIR_block'):
	conv_1 = Conv2d(x, nb_filters / 2 , (3, 3), (1, 1), act=None, padding='SAME',
					W_init=w_init, b_init=b_init, name='conv_1_1'+name)
	conv_1 = BatchNormLayer(conv_1, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							 is_train=is_train, gamma_init=gamma_init, name='conv_1_2'+name)

	conv_2 = Conv2d(conv_1, nb_filters / 2, (1, 1), (1, 1), act=None, padding='SAME',
					W_init=w_init, b_init=b_init, name='conv_2_1' + name)
	conv_2 = BatchNormLayer(conv_2, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							is_train=is_train, gamma_init=gamma_init, name='conv_2_2' + name)

	conv_3 = Conv2d(conv_2, nb_filters / 2, (1, 1), (1, 1), act=None, padding='SAME',
					W_init=w_init, b_init=b_init, name='conv_3_1' + name)
	conv_3 = BatchNormLayer(conv_3, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							is_train=is_train, gamma_init=gamma_init, name='conv_3_2' + name)

	RIR_in = ElementwiseLayer([conv_1, conv_3], tf.add)

	conv_4 = Conv2d(RIR_in, nb_filters, (3, 3), (1, 1), act=None, padding='SAME',
					W_init=w_init, b_init=b_init, name='conv_4_1' + name)
	conv_4 = BatchNormLayer(conv_4, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							is_train=is_train, gamma_init=gamma_init, name='conv_4_2' + name)

	RIR_out = ElementwiseLayer([x, conv_4], tf.add)

	return RIR_out


def encoder_block(x, nb_filters, w_init, b_init, is_train, gamma_init, name='encoder_block'):
	conv_a = Conv2d(x, nb_filters, (3, 3), (2, 2), act=None, padding='SAME',
					W_init=w_init, b_init=b_init, name='conv_a_1'+name)
	conv_a = BatchNormLayer(conv_a, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							 is_train=is_train, gamma_init=gamma_init, name='conv_a_2'+name)

	en_RIR_output = RIR_block(conv_a, nb_filters, w_init, b_init, is_train, gamma_init, name='en_RIR_block'+name)

	conv_b = Conv2d(en_RIR_output, nb_filters, (3, 3), (1, 1), act=None, padding='SAME',
					W_init=w_init, b_init=b_init, name='conv_b_1' + name)
	conv_b = BatchNormLayer(conv_b, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							is_train=is_train, gamma_init=gamma_init, name='conv_b_2' + name)

	return conv_b


def decoder_block(x, nb_filters, w_init, b_init, is_train, gamma_init, name='decoder_block'):
	deconv_a = DeConv2d(x, nb_filters, (3, 3), strides=(1, 1), padding='SAME',
					 act=None, W_init=w_init, b_init=b_init, name='deconv_a_1')
	deconv_a = BatchNormLayer(deconv_a, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							is_train=is_train, gamma_init=gamma_init, name='deconv_a_2' + name)

	de_RIR_output = RIR_block(deconv_a, nb_filters, w_init, b_init, is_train, gamma_init, name='de_RIR_block'+name)

	deconv_b = DeConv2d(de_RIR_output, nb_filters, (3, 3), strides=(2, 2), padding='SAME',
					 act=None, W_init=w_init, b_init=b_init, name='deconv_b_1' + name)
	deconv_b = BatchNormLayer(deconv_b, act=lambda x: tf.nn.leaky_relu(x, 0.2),
							is_train=is_train, gamma_init=gamma_init, name='deconv_b_2' + name)

	return deconv_b


# SOAM
def SOAM(x, w_init, b_init, is_train, gamma_init, name='SOAM'):
	batch_size, H, W, C = x.outputs.shape.as_list()

	###### convolution
	conv1 = Conv2d(x, 1, (1, 1), (1, 1), act=None, padding='SAME',
					   W_init=w_init, b_init=b_init, name='conv1' + name)
	BN_lrelu1 = BatchNormLayer(conv1, act=lambda x: tf.nn.leaky_relu(x, 0.2),
								   is_train=is_train, gamma_init=gamma_init, name='BN_lrelu1' + name)

	reshape_HW = ReshapeLayer(BN_lrelu1, shape=(batch_size, H, W), name='reshape_HW' + name)

	###### matmul
	matmul_1 = tf.matmul(reshape_HW.outputs, reshape_HW.outputs, transpose_b=True,
							 name='matmul_left1' + name)  # H*H
	matmul_2 = tf.matmul(reshape_HW.outputs, reshape_HW.outputs, transpose_a=True,
							 name='matmul_left2' + name)  # W*W

	###### Softmax
	ATT_HH = tf.nn.softmax(matmul_1, axis=-1, name='ATT_1' + name)
	ATT_WW = tf.nn.softmax(matmul_2, axis=-1, name='ATT_2' + name)

	ATT_HH_tl = InputLayer(ATT_HH, name='ATT_HH_tl' + name)
	ATT_WW_tl = InputLayer(ATT_WW, name='ATT_WW_tl' + name)

	######
	conv_ATT = Conv2d(x, C, (1, 1), (1, 1), act=None, padding='SAME',
					  W_init=w_init, b_init=b_init, name='conv_ATT' + name)
	BN_lrelu_ATT = BatchNormLayer(conv_ATT, act=lambda x: tf.nn.leaky_relu(x, 0.2),
								  is_train=is_train, gamma_init=gamma_init, name='BN_lrelu_ATT' + name)

	###### Reshape and Matmul
	ATT_re_1 = ReshapeLayer(BN_lrelu_ATT, shape=(batch_size, H, C * W), name='ATT_re_1' + name)
	ATT_re_2 = ReshapeLayer(BN_lrelu_ATT, shape=(batch_size, W, C * H), name='ATT_re_2' + name)

	matmul_3 = tf.matmul(ATT_HH, ATT_re_1.outputs, name='matmul_3' + name)
	matmul_4 = tf.matmul(ATT_WW, ATT_re_2.outputs, name='matmul_4' + name)

	matmul_3 = InputLayer(matmul_3, name='matmul_left3_tl' + name)
	matmul_4 = InputLayer(matmul_4, name='matmul_right3_tl' + name)

	###### Reshape again
	output_re_1 = ReshapeLayer(matmul_3, shape=(batch_size, H, W, C), name='output_re_1' + name)
	output_re_2 = ReshapeLayer(matmul_4, shape=(batch_size, H, W, C), name='output_re_2' + name)

	###### Add
	add_all = ElementwiseLayer([output_re_1, output_re_2], tf.add)

	conv_output = Conv2d(add_all, C, (1, 1), (1, 1), act=None, padding='SAME',
						 W_init=w_init, b_init=b_init, name='conv_output' + name)
	BNL_output = BatchNormLayer(conv_output, act=lambda x: tf.nn.leaky_relu(x, 0.2),
								is_train=is_train, gamma_init=gamma_init, name='BNL_output' + name)

	add_output = ElementwiseLayer([BN_lrelu_ATT, BNL_output], tf.add)

	return add_output, ATT_HH_tl, ATT_WW_tl


def discriminator(input_images, is_train=True, reuse=False):
	w_init = tf.random_normal_initializer(stddev=0.02)
	b_init = None
	gamma_init = tf.random_normal_initializer(1., 0.02)
	df_filters = 64

	# with tf.device('/gpu:1'):
	with tf.variable_scope("discriminator", reuse=reuse):
		# tl.layers.set_name_reuse(reuse)
		net_in = InputLayer(input_images, name='input')

		conv1_0 = Conv2d(net_in, df_filters, (3, 3), (2, 2), act=lambda x: tf.nn.leaky_relu(x, 0.2),
						padding='SAME', W_init=w_init, name='conv1_0')

		conv1_1 = encoder_block(conv1_0, df_filters * 1, w_init, b_init, is_train, gamma_init, name='conv1_1')  # 128
		conv1_2 = encoder_block(conv1_1, df_filters * 2, w_init, b_init, is_train, gamma_init, name='conv1_2')  # 64
		conv1_3 = encoder_block(conv1_2, df_filters * 4, w_init, b_init, is_train, gamma_init, name='conv1_3')  # 32
		conv1_4 = encoder_block(conv1_3, df_filters * 8, w_init, b_init, is_train, gamma_init, name='conv1_4')  # 16

		conv1_5 = Conv2d(conv1_4, 1, (3, 3), (1, 1), act=None,
						 padding='SAME', W_init=w_init, name='conv1_5')

		net_ho = FlattenLayer(conv1_5, name='output/flatten')
		net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='output/dense')
		logits = net_ho.outputs
		net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

	return net_ho, logits


def generator(x, is_train=False, reuse=False):
	w_init = tf.truncated_normal_initializer(stddev=0.01)
	b_init = tf.constant_initializer(value=0.0)
	gamma_init = tf.random_normal_initializer(1., 0.02)
	nb_filters = 64

	# with tf.device('/gpu:2'):
	with tf.variable_scope("Generator", reuse=reuse):
		# tl.layers.set_name_reuse(reuse)
		inputs = InputLayer(x, name='input') # 256

		###################################################################################
		conv1_0 = Conv2d(inputs, nb_filters * 1, (3, 3), (1, 1), act=None, padding='SAME',
						W_init=w_init, b_init=b_init, name='conv1_0')
		conv1_0 = BatchNormLayer(conv1_0, act=lambda x: tf.nn.leaky_relu(x, 0.2),
								is_train=is_train, gamma_init=gamma_init, name='conv1_0_BN')
		conv1_1 = encoder_block(conv1_0, nb_filters * 1, w_init, b_init, is_train, gamma_init, name='conv1_1') # 128
		conv1_2 = encoder_block(conv1_1, nb_filters * 2, w_init, b_init, is_train, gamma_init, name='conv1_2') # 64
		conv1_3 = encoder_block(conv1_2, nb_filters * 4, w_init, b_init, is_train, gamma_init, name='conv1_3') # 32
		conv1_4 = encoder_block(conv1_3, nb_filters * 8, w_init, b_init, is_train, gamma_init, name='conv1_4') # 16

		deconv1_4 = decoder_block(conv1_4, nb_filters * 4, w_init, b_init, is_train, gamma_init, name='deconv1_4') # 32
		residual1_1 = ElementwiseLayer([conv1_3, deconv1_4], tf.add)
		
		deconv1_3 = decoder_block(residual1_1, nb_filters * 2, w_init, b_init, is_train, gamma_init, name='deconv1_3') # 64
		residual1_2 = ElementwiseLayer([conv1_2, deconv1_3], tf.add)
		
		deconv1_2 = decoder_block(residual1_2, nb_filters * 1, w_init, b_init, is_train, gamma_init, name='deconv1_2') # 128
		residual1_3 = ElementwiseLayer([conv1_1, deconv1_2], tf.add)
		
		deconv1_1 = decoder_block(residual1_3, nb_filters * 1, w_init, b_init, is_train, gamma_init, name='deconv1_1') # 256
		residual1_4 = ElementwiseLayer([conv1_0, deconv1_1], tf.add)

		S_O_A_M1, ATT_HH1, ATT_WW1 = SOAM(residual1_4, w_init, b_init, is_train, gamma_init, name='S_O_A_M1')

		output1_1 = Conv2d(S_O_A_M1, 1, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME',
						W_init=w_init, b_init=b_init, name='output1')
		output1_2 = ElementwiseLayer([inputs, output1_1], tf.add)

		###################################################################################
		conv2_0 = Conv2d(output1_2, nb_filters * 1, (3, 3), (1, 1), act=None, padding='SAME',
						 W_init=w_init, b_init=b_init, name='conv2_0')
		conv2_0 = BatchNormLayer(conv2_0, act=lambda x: tf.nn.leaky_relu(x, 0.2),
								 is_train=is_train, gamma_init=gamma_init, name='conv2_0_BN')
		Wave_conn1_0 = ElementwiseLayer([residual1_4, conv2_0], tf.add)

		conv2_1 = encoder_block(Wave_conn1_0, nb_filters * 1, w_init, b_init, is_train, gamma_init, name='conv2_1')  # 128
		Wave_conn1_1 = ElementwiseLayer([residual1_3, conv2_1], tf.add)
		
		conv2_2 = encoder_block(Wave_conn1_1, nb_filters * 2, w_init, b_init, is_train, gamma_init, name='conv2_2')  # 64
		Wave_conn1_2 = ElementwiseLayer([residual1_2, conv2_2], tf.add)
		
		conv2_3 = encoder_block(Wave_conn1_2, nb_filters * 4, w_init, b_init, is_train, gamma_init, name='conv2_3')  # 32
		Wave_conn1_3 = ElementwiseLayer([residual1_1, conv2_3], tf.add)
		
		conv2_4 = encoder_block(Wave_conn1_3, nb_filters * 8, w_init, b_init, is_train, gamma_init, name='conv2_4')  # 16
		Wave_conn1_4 = ElementwiseLayer([conv1_4, conv2_4], tf.add)


		deconv2_4 = decoder_block(Wave_conn1_4, nb_filters * 4, w_init, b_init, is_train, gamma_init, name='deconv2_4')  # 32
		residual2_1 = ElementwiseLayer([Wave_conn1_3, deconv2_4], tf.add)

		deconv2_3 = decoder_block(residual2_1, nb_filters * 2, w_init, b_init, is_train, gamma_init, name='deconv2_3')  # 64
		residual2_2 = ElementwiseLayer([Wave_conn1_2, deconv2_3], tf.add)

		deconv2_2 = decoder_block(residual2_2, nb_filters * 1, w_init, b_init, is_train, gamma_init, name='deconv2_2')  # 128
		residual2_3 = ElementwiseLayer([Wave_conn1_1, deconv2_2], tf.add)

		deconv2_1 = decoder_block(residual2_3, nb_filters * 1, w_init, b_init, is_train, gamma_init, name='deconv2_1')  # 256
		residual2_4 = ElementwiseLayer([Wave_conn1_0, deconv2_1], tf.add)

		S_O_A_M2, ATT_HH2, ATT_WW2 = SOAM(residual2_4, w_init, b_init, is_train, gamma_init, name='S_O_A_M2')

		out = Conv2d(S_O_A_M2, 1, (3, 3), (1, 1), act=tf.nn.tanh, name='out')
		out = ElementwiseLayer([out, output1_2], tf.add)
		out.outputs = tl.act.ramp(out.outputs, v_min=-1, v_max=1)

	return out, ATT_HH1, ATT_WW1, ATT_HH2, ATT_WW2


if __name__ == "__main__":
	pass
