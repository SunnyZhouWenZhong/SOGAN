from tensorlayer.prepro import *
import numpy as np
import skimage.measure
import scipy
from time import localtime, strftime
import logging
import tensorflow as tf
import os

def distort_img(x):
	x = (x + 1.) / 2.
	x = flip_axis(x, axis=1, is_random=True)
	x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
	x = rotation(x, rg=10, is_random=True, fill_mode='constant')
	x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
	x = zoom(x, zoom_range=(0.90, 1.10), fill_mode='constant')
	x = brightness(x, gamma=0.05, is_random=True)
	x = x * 2 - 1
	return x


def to_bad_img(x, mask):
	x = (x + 1.) / 2.
	fft = scipy.fftpack.fft2(x[:, :, 0])
	fft = scipy.fftpack.fftshift(fft)
	fft = fft * mask
	fft = scipy.fftpack.ifftshift(fft)
	x = scipy.fftpack.ifft2(fft)
	x = np.abs(x)
	x = x * 2 - 1
	return x[:, :, np.newaxis]

def fft_abs_for_map_fn(x):
	x = (x + 1.) / 2.
	x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
	fft = tf.spectral.fft2d(x_complex)
	fft_abs = tf.abs(fft)
	return fft_abs

def ssim(data):
	x_good, x_bad = data
	x_good = np.squeeze(x_good)
	x_bad = np.squeeze(x_bad)
	ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
	return ssim_res


def psnr(data):
	x_good, x_bad = data
	psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
	return psnr_res

def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, _ in grad_and_vars:
			# g = tf.cast(tf.convert_to_tensor(g), dtype=tf.float32)
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)

		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

def count_trainable_params(scope):
	total_parameters = 0
	for variable in tf.trainable_variables(scope=scope):
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	return total_parameters

def logging_setup(log_dir):
	current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
	log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
	log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))

	log_all = logging.getLogger('log_all')
	log_all.setLevel(logging.DEBUG)
	log_all.addHandler(logging.FileHandler(log_all_filename))

	log_eval = logging.getLogger('log_eval')
	log_eval.setLevel(logging.INFO)
	log_eval.addHandler(logging.FileHandler(log_eval_filename))

	log_50_filename = os.path.join(log_dir, 'log_50_images_testing_{}.log'.format(current_time_str))

	log_50 = logging.getLogger('log_50')
	log_50.setLevel(logging.DEBUG)
	log_50.addHandler(logging.FileHandler(log_50_filename))

	return log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename


if __name__ == "__main__":
	pass
