import pickle
from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat, savemat
import tensorlayer as tl
import tensorflow as tf
import scipy.misc
import time
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main_train():
	mask_perc = tl.global_flag['maskperc']
	mask_name = tl.global_flag['mask']

	# =================================== BASIC CONFIGS =================================== #

	print('[*] run basic configs ... ')

	log_dir = "log_{}_{}".format(mask_name, mask_perc)
	tl.files.exists_or_mkdir(log_dir)
	log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename = logging_setup(log_dir)

	save_dir = "samples_{}_{}".format(mask_name, mask_perc)
	tl.files.exists_or_mkdir(save_dir)

	# configs
	batch_size = config.TRAIN.batch_size
	early_stopping_num = config.TRAIN.early_stopping_num
	g_alpha = config.TRAIN.g_alpha
	g_gamma = config.TRAIN.g_gamma
	g_adv = config.TRAIN.g_adv
	lr = config.TRAIN.lr
	lr_decay = config.TRAIN.lr_decay
	decay_every = config.TRAIN.decay_every
	n_epoch = config.TRAIN.n_epoch
	sample_size = config.TRAIN.sample_size

	log_config(log_all_filename, config)
	log_config(log_eval_filename, config)
	log_config(log_50_filename, config)

	# ==================================== PREPARE DATA ==================================== #

	print('[*] load data ... ')
	training_data_path = config.TRAIN.training_data_path
	val_data_path = config.TRAIN.val_data_path
	testing_data_path = config.TRAIN.testing_data_path

	with open(training_data_path, 'rb') as f:
		X_train = pickle.load(f)

	with open(val_data_path, 'rb') as f:
		X_val = pickle.load(f)

	with open(testing_data_path, 'rb') as f:
		X_test = pickle.load(f)

	print('[*] X_train shape/min/max: ', X_train.shape, X_train.min(), X_train.max())
	print('[*] X_val shape/min/max: ', X_val.shape, X_val.min(), X_val.max())
	print('[*] X_test shape/min/max: ', X_test.shape, X_test.min(), X_test.max())

	print('[*] loading mask ... ')
	if mask_name == "spiral":
		mask = \
			loadmat(
				os.path.join(config.TRAIN.mask_Spiral_path, "spiral_{}.mat".format(mask_perc)))[
				'spiral_mask']
	elif mask_name == "radial":
		mask = \
			loadmat(
				os.path.join(config.TRAIN.mask_Radial_path, "radial_{}.mat".format(mask_perc)))[
				'radial_mask']
	elif mask_name == "cartesian":
		mask = \
			loadmat(
				os.path.join(config.TRAIN.mask_Cartesian_path, "cartesian_{}.mat".format(mask_perc)))[
				'cartesian_mask']
	elif mask_name == "random":
		mask = \
			loadmat(
				os.path.join(config.TRAIN.mask_Random_path, "random_{}.mat".format(mask_perc)))[
				'random_mask']
	else:
		raise ValueError("no such mask exists: {}".format(mask_name))

	# ==================================== DEFINE MODEL ==================================== #

	print('[*] define model ... ')

	nw, nh, nz = X_train.shape[1:]


	# define placeholders
	t_image_good = tf.placeholder('float32', [batch_size, nw, nh, nz], name='good_image')
	t_image_good_samples = tf.placeholder('float32', [sample_size, nw, nh, nz], name='good_image_samples')
	t_image_bad = tf.placeholder('float32', [batch_size, nw, nh, nz], name='bad_image')
	t_image_bad_samples = tf.placeholder('float32', [sample_size, nw, nh, nz], name='bad_image_samples')
	t_gen = tf.placeholder('float32', [batch_size, nw, nh, nz], name='generated_image_for_test')
	t_gen_sample = tf.placeholder('float32', [sample_size, nw, nh, nz], name='generated_sample_image_for_test')

	# define generator network
	net, _, _, _, _ = generator(t_image_bad, is_train=True, reuse=False)
	net_test, _, _, _, _ = generator(t_image_bad, is_train=False, reuse=True)
	net_test_sample, ATT_HH1, ATT_WW1, ATT_HH2, ATT_WW2 = generator(t_image_bad_samples, is_train=False, reuse=True)

	# define discriminator network
	net_d, logits_fake = discriminator(net.outputs, is_train=True, reuse=False)
	_, logits_real = discriminator(t_image_good, is_train=True, reuse=True)

	# ==================================== DEFINE LOSS ==================================== #

	print('[*] define loss functions ... ')

	# discriminator loss
	d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
	d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
	d_loss = d_loss1 + d_loss2

	# generator loss (adversarial)
	g_adver = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')

	# generator loss (pixel-wise)
	g_MAE = tf.reduce_mean(tf.abs(t_image_good - net.outputs), name='g_MAE')


	# generator loss (gradient loss)
	good_dif1 = t_image_good[1:, :, :] - t_image_good[:-1, :, :]
	good_dif2 = t_image_good[:, 1:, :] - t_image_good[:, :-1, :]
	gen_dif1 = net.outputs[1:, :, :] - net.outputs[:-1, :, :]
	gen_dif2 = net.outputs[:, 1:, :] - net.outputs[:, :-1, :]
	dif1 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(good_dif1, gen_dif1), axis=[1, 2]))
	dif2 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(good_dif2, gen_dif2), axis=[1, 2]))
	g_TV = dif1 + dif2

	# generator loss (total)
	g_loss = g_adv * g_adver + g_alpha * g_MAE + g_gamma * g_TV

	# nmse metric for testing purpose
	nmse_a_0_1 = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen, t_image_good), axis=[1, 2, 3]))
	nmse_b_0_1 = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
	nmse_0_1 = nmse_a_0_1 / nmse_b_0_1

	nmse_a_0_1_sample = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen_sample, t_image_good_samples), axis=[1, 2, 3]))
	nmse_b_0_1_sample = tf.sqrt(tf.reduce_sum(tf.square(t_image_good_samples), axis=[1, 2, 3]))
	nmse_0_1_sample = nmse_a_0_1_sample / nmse_b_0_1_sample

	# ==================================== DEFINE TRAIN OPTS ==================================== #

	print('[*] define training options ... ')

	g_vars = tl.layers.get_variables_with_name('Generator', True, True)
	d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
	Total_parameters_g = count_trainable_params('Generator')
	Total_parameters_d = count_trainable_params('discriminator')
	Total_parameters = Total_parameters_g + Total_parameters_d

	with tf.variable_scope('learning_rate'):
		lr_v = tf.Variable(lr, trainable=False)

	g_optim = tf.train.AdamOptimizer(lr_v).minimize(g_loss, var_list=g_vars)
	d_optim = tf.train.AdamOptimizer(lr_v).minimize(d_loss, var_list=d_vars)

	# ==================================== TRAINING ==================================== #

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	sess.run(tf.global_variables_initializer())
	print("g Total training params: %.2fM" % (Total_parameters_g / 1e6))
	log_all.debug("g Total training params: %.2fM\n" % (Total_parameters_g / 1e6))
	print("d Total training params: %.2fM" % (Total_parameters_d / 1e6))
	log_all.debug("d Total training params: %.2fM\n" % (Total_parameters_d / 1e6))
	print("Total training params: %.2fM" % (Total_parameters / 1e6))
	log_all.debug("Total training params: %.2fM\n" % (Total_parameters / 1e6))

	n_training_examples = len(X_train)
	n_step_epoch = round(n_training_examples / batch_size)

	# sample testing images
	idex = tl.utils.get_random_int(min_v=0, max_v=len(X_test) - 1, number=sample_size, seed=config.TRAIN.seed)
	X_samples_good = X_test[idex]
	X_samples_bad = threading_data(X_samples_good, fn=to_bad_img, mask=mask)

	x_good_sample_rescaled = (X_samples_good + 1) / 2
	x_bad_sample_rescaled = (X_samples_bad + 1) / 2

	tl.visualize.save_images(X_samples_good,
							 [5, 10],
							 os.path.join(save_dir, "sample_image_good.png"))

	tl.visualize.save_images(X_samples_bad,
							 [5, 10],
							 os.path.join(save_dir, "sample_image_bad.png"))

	scipy.misc.imsave(os.path.join(save_dir, "mask.png"), mask * 255)


	print('[*] start training ... ')

	best_nmse = np.inf
	best_epoch = 1
	esn = early_stopping_num

	training_NMSE = []
	training_PSNR = []
	training_SSIM = []

	val_NMSE = []
	val_PSNR = []
	val_SSIM = []

	for epoch in range(0, n_epoch):

		# learning rate decay
		if epoch != 0 and (epoch % decay_every == 0):
			new_lr_decay = lr_decay ** (epoch // decay_every)
			sess.run(tf.assign(lr_v, lr * new_lr_decay))
			log = " ** new learning rate: %f" % (lr * new_lr_decay)
			print(log)
			log_all.debug(log)
		elif epoch == 0:
			log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
			print(log)
			log_all.debug(log)

		for step in range(n_step_epoch):
			step_time = time.time()
			idex = tl.utils.get_random_int(min_v=0, max_v=n_training_examples - 1, number=batch_size)
			X_good = X_train[idex]
			X_good_aug = threading_data(X_good, fn=distort_img)
			X_bad = threading_data(X_good_aug, fn=to_bad_img, mask=mask)

			errD, _ = sess.run([d_loss, d_optim], {t_image_good: X_good_aug, t_image_bad: X_bad})
			errG, errG_MAE, errG_TV, _ = sess.run([g_loss, g_MAE, g_TV, g_optim],
												  {t_image_good: X_good_aug, t_image_bad: X_bad})

			log = "Epoch[{:3}/{:3}] step={:3} d_loss={:5} g_loss={:5} g_mae={:5} g_TV={:5} took {:3}s" \
				.format(
				epoch + 1,
				n_epoch,
				step,
				round(float(errD), 3),
				round(float(errG), 3),
				round(float(errG_MAE), 3),
				round(float(errG_TV), 3),
				round(time.time() - step_time, 2))

			print(log)
			log_all.debug(log)

		# evaluation for training data
		total_nmse_training = 0
		total_ssim_training = 0
		total_psnr_training = 0
		num_training_temp = 0

		for batch in tl.iterate.minibatches(inputs=X_train, targets=X_train, batch_size=batch_size, shuffle=False):
			x_good, _ = batch
			# x_bad = threading_data(x_good, fn=to_bad_img, mask=mask)
			x_bad = threading_data(
				x_good,
				fn=to_bad_img,
				mask=mask)

			x_gen = sess.run(net_test.outputs, {t_image_bad: x_bad})

			x_good_0_1 = (x_good + 1) / 2
			x_gen_0_1 = (x_gen + 1) / 2

			nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_0_1})
			ssim_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=ssim)
			psnr_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=psnr)
			total_nmse_training += np.sum(nmse_res)
			total_ssim_training += np.sum(ssim_res)
			total_psnr_training += np.sum(psnr_res)
			num_training_temp += batch_size

		total_nmse_training /= num_training_temp
		total_ssim_training /= num_training_temp
		total_psnr_training /= num_training_temp

		training_NMSE.append(total_nmse_training)
		training_PSNR.append(total_psnr_training)
		training_SSIM.append(total_ssim_training)

		log = "Epoch: {}\nNMSE training: {:8}, SSIM training: {:8}, PSNR training: {:8}".format(
			epoch + 1,
			total_nmse_training,
			total_ssim_training,
			total_psnr_training)

		print(log)
		log_all.debug(log)
		log_eval.info(log)

		# evaluation for validation data
		total_nmse_val = 0
		total_ssim_val = 0
		total_psnr_val = 0
		num_val_temp = 0


		for batch in tl.iterate.minibatches(inputs=X_val, targets=X_val, batch_size=batch_size, shuffle=False):
			x_good, _ = batch
			# x_bad = threading_data(x_good, fn=to_bad_img, mask=mask)
			x_bad = threading_data(
				x_good,
				fn=to_bad_img,
				mask=mask)

			x_gen = sess.run(net_test.outputs, {t_image_bad: x_bad})

			x_good_0_1 = (x_good + 1) / 2
			x_gen_0_1 = (x_gen + 1) / 2

			nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_0_1})
			ssim_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=ssim)
			psnr_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=psnr)
			total_nmse_val += np.sum(nmse_res)
			total_ssim_val += np.sum(ssim_res)
			total_psnr_val += np.sum(psnr_res)
			num_val_temp += batch_size

		total_nmse_val /= num_val_temp
		total_ssim_val /= num_val_temp
		total_psnr_val /= num_val_temp

		val_NMSE.append(total_nmse_val)
		val_PSNR.append(total_psnr_val)
		val_SSIM.append(total_ssim_val)

		log = "Epoch: {}\nNMSE val: {:8}, SSIM val: {:8}, PSNR val: {:8}".format(
			epoch + 1,
			total_nmse_val,
			total_ssim_val,
			total_psnr_val)
		print(log)
		log_all.debug(log)
		log_eval.info(log)

		img = sess.run(net_test_sample.outputs, {t_image_bad_samples: X_samples_bad})
		tl.visualize.save_images(img,
								 [5, 10],
								 os.path.join(save_dir, "image_{}.png".format(epoch+1)))

		if total_nmse_val < best_nmse:
			esn = early_stopping_num  # reset early stopping num
			best_nmse = total_nmse_val
			best_epoch = epoch + 1

			tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir='checkpoint')
			print("[*] Save checkpoints SUCCESS!")
		else:
			esn -= 1

		log = "Best NMSE result: {} at {} epoch".format(best_nmse, best_epoch)
		log_eval.info(log)
		log_all.debug(log)
		print(log)

		# early stopping triggered
		if esn == 0:
			log_eval.info(log)

			log_eval.info("\ntraining_NMSE: {}".format(training_NMSE))
			log_eval.info("\ntraining_PSNR: {}".format(training_PSNR))
			log_eval.info("\ntraining_SSIM: {}".format(training_SSIM))
			log_eval.info("\nval_NMSE: {}".format(val_NMSE))
			log_eval.info("\nval_PSNR: {}".format(val_PSNR))
			log_eval.info("\nval_SSIM: {}".format(val_SSIM))

			tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', save_dir='checkpoint')

			# evluation for test data
			test_start_time = time.time()
			x_gen, ATT_HH1, ATT_WW1, ATT_HH2, ATT_WW2 = sess.run(
				[net_test_sample.outputs,
				 ATT_HH1.outputs, ATT_WW1.outputs,
				 ATT_HH2.outputs, ATT_WW2.outputs],
				{t_image_bad_samples: X_samples_bad})

			test_duration = (time.time() - test_start_time) / sample_size
			x_gen_0_1 = (x_gen + 1) / 2

			nmse_res = sess.run(nmse_0_1_sample, {t_gen_sample: x_gen_0_1, t_image_good_samples: x_good_sample_rescaled})
			ssim_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=ssim)
			psnr_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=psnr)

			log = "NMSE testing: {}\nSSIM testing: {}\nPSNR testing: {}\n\n".format(
				nmse_res,
				ssim_res,
				psnr_res)

			log_50.debug(log)

			log = "NMSE testing average: {}\nSSIM testing average: {}\nPSNR testing average: {}\n\n".format(
				np.mean(nmse_res),
				np.mean(ssim_res),
				np.mean(psnr_res))

			log_50.debug(log)

			log = "NMSE testing std: {}\nSSIM testing std: {}\nPSNR testing std: {}\n\n".format(np.std(nmse_res),
																								np.std(ssim_res),
																								np.std(psnr_res))

			log_50.debug(log)

			log = "\nAverage test time: {}\n".format(test_duration)

			log_50.debug(log)

			# evaluation for zero-filled (ZF) data
			nmse_res_zf = sess.run(nmse_0_1_sample,
								   {t_gen_sample: x_bad_sample_rescaled, t_image_good_samples: x_good_sample_rescaled})
			ssim_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=ssim)
			psnr_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=psnr)

			log = "NMSE ZF testing: {}\nSSIM ZF testing: {}\nPSNR ZF testing: {}\n\n".format(
				nmse_res_zf,
				ssim_res_zf,
				psnr_res_zf)

			log_50.debug(log)

			log = "NMSE ZF average testing: {}\nSSIM ZF average testing: {}\nPSNR ZF average testing: {}\n\n".format(
				np.mean(nmse_res_zf),
				np.mean(ssim_res_zf),
				np.mean(psnr_res_zf))

			log_50.debug(log)

			log = "NMSE ZF std testing: {}\nSSIM ZF std testing: {}\nPSNR ZF std testing: {}\n\n".format(
				np.std(nmse_res_zf),
				np.std(ssim_res_zf),
				np.std(psnr_res_zf))

			log_50.debug(log)

			# sample testing images
			tl.visualize.save_images(x_gen,
									 [5, 10],
									 os.path.join(save_dir, "final_generated_image.png"))

			print("[*] Job finished!")
			break


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--mask', type=str, default='radial', help='gaussian1d, gaussian2d, poisson2d,'
																   'spiral, radial, cartesian, random')
	parser.add_argument('--maskperc', type=int, default='20', help='10,20,30,40,50')

	args = parser.parse_args()

	tl.global_flag['mask'] = args.mask
	tl.global_flag['maskperc'] = args.maskperc

	main_train()
