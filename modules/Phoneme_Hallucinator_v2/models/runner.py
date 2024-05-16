import os
import logging
from collections import defaultdict
import numpy as np
import pickle
import tensorflow as tf
from pprint import pformat

from .utils import visualize, plot_functions, plot_img_functions

class Runner(object):
    def __init__(self, args, model):
        self.args = args
        self.sess = model.sess
        self.model = model

    def set_dataset(self, trainset, validset, testset):
        self.trainset = trainset
        self.validset = validset
        self.testset = testset

    def train(self):
        train_metrics = []
        num_batches = self.trainset.num_batches
        self.trainset.initialize()
        for i in range(num_batches):
            batch = self.trainset.next_batch()
            metric, summ, step, _ = self.model.execute(
                [self.model.metric, self.model.summ_op, 
                self.model.global_step, self.model.train_op], 
                batch)
            if (self.args.summ_freq > 0) and (i % self.args.summ_freq == 0):
                self.model.writer.add_summary(summ, step)
            train_metrics.append(metric)
        train_metrics = np.concatenate(train_metrics, axis=0)

        return np.mean(train_metrics)

    def valid(self):
        valid_metrics = []
        num_batches = self.validset.num_batches
        self.validset.initialize()
        for i in range(num_batches):
            batch = self.validset.next_batch()
            metric = self.model.execute(self.model.metric, batch)
            valid_metrics.append(metric)
        valid_metrics = np.concatenate(valid_metrics, axis=0)

        return np.mean(valid_metrics)

    def valid_mse(self):
        valid_mse = []
        num_batches = self.validset.num_batches
        self.validset.initialize()
        for i in range(num_batches):
            batch = self.validset.next_batch()
            sample = self.model.execute(self.model.sample, batch)
            mse = np.mean(np.sum(np.square(sample-batch['x']), axis=tuple(range(2,sample.ndim))), axis=1)
            valid_mse.append(mse)
        valid_mse = np.concatenate(valid_mse, axis=0)

        return np.mean(valid_mse)

    def valid_chd(self):
        pass

    def valid_emd(self):
        pass

    def test(self):
        test_metrics = []
        num_batches = self.testset.num_batches
        self.testset.initialize()
        for i in range(num_batches):
            batch = self.testset.next_batch()
            metric = self.model.execute(self.model.metric, batch)
            test_metrics.append(metric)
        test_metrics = np.concatenate(test_metrics)

        return np.mean(test_metrics)

    def test_mse(self):
        test_mse = []
        num_batches = self.testset.num_batches
        self.testset.initialize()
        for i in range(num_batches):
            batch = self.testset.next_batch()
            sample = self.model.execute(self.model.sample, batch)
            mse = np.mean(np.sum(np.square(sample-batch['x']), axis=tuple(range(2,sample.ndim))), axis=1)
            test_mse.append(mse)
        test_mse = np.concatenate(test_mse, axis=0)

        return np.mean(test_mse)

    def test_chd(self):
        pass

    def test_emd(self):
        pass

    def run(self):
        logging.info('==== start training ====')
        best_train_metric = -np.inf
        best_valid_metric = -np.inf
        best_test_metric = -np.inf
        for epoch in range(self.args.epochs):
            train_metric = self.train()
            valid_metric = self.valid()
            test_metric = self.test()
            # save
            if train_metric > best_train_metric:
                best_train_metric = train_metric
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                self.model.save()
            if test_metric > best_test_metric:
                best_test_metric = test_metric
            logging.info("Epoch %d, train: %.4f/%.4f, valid: %.4f/%.4f test: %.4f/%.4f" %
                 (epoch, train_metric, best_train_metric, 
                 valid_metric, best_valid_metric,
                 test_metric, best_test_metric))
            # evaluate
            if epoch % 100 == 0:
                logging.info('==== start evaluating ====')
                self.evaluate(folder=f'{epoch}', load=False)
            self.model.save('last')

        # finish
        logging.info('==== start evaluating ====')
        self.evaluate(load=True)

    def evaluate(self, folder='test', load=True):
        save_dir = f'{self.args.exp_dir}/evaluate/{folder}/'
        os.makedirs(save_dir, exist_ok=True)
        if load: self.model.load()
        
        # # likelihood
        if 'likel' in self.args.eval_metrics:
            valid_likel = self.valid()
            test_likel = self.test()
            logging.info(f"likelihood => valid: {valid_likel} test: {test_likel}")

        # # mse
        if 'mse' in self.args.eval_metrics:
            valid_mse = self.valid_mse()
            test_mse = self.test_mse()
            logging.info(f"mse => valid: {valid_mse} test: {test_mse}")

        if 'chd' in self.args.eval_metrics:
            valid_chd = self.valid_chd()
            test_chd = self.test_chd()
            logging.info(f"chd => valid: {valid_chd} test: {test_chd}")

        if 'emd' in self.args.eval_metrics:
            valid_emd = self.valid_emd()
            test_emd = self.test_emd()
            logging.info(f"emd => valid: {valid_emd} test: {test_emd}")

        if 'sam' in self.args.eval_metrics:
            # train set
            self.trainset.initialize()
            batch = self.trainset.next_batch()
            train_sample = self.model.execute(self.model.sample, batch)
            visualize(train_sample, batch, f'{save_dir}/train_sam')

            # valid set
            self.validset.initialize()
            batch = self.validset.next_batch()
            valid_sample = self.model.execute(self.model.sample, batch)
            visualize(valid_sample, batch, f'{save_dir}/valid_sam')

            # test set
            self.testset.initialize()
            batch = self.testset.next_batch()
            test_sample = self.model.execute(self.model.sample, batch)
            visualize(test_sample, batch, f'{save_dir}/test_sam')

        if 'fns' in self.args.eval_metrics:
            # train set
            self.trainset.initialize()
            batch = self.trainset.next_batch()
            train_mean, train_std = self.model.execute([self.model.mean, self.model.std], batch)
            plot_functions(train_mean, train_std, batch, f'{save_dir}/train_fn')

            # valid set
            self.validset.initialize()
            batch = self.validset.next_batch()
            valid_mean, valid_std = self.model.execute([self.model.mean, self.model.std], batch)
            plot_functions(valid_mean, valid_std, batch, f'{save_dir}/valid_fn')

            # test set
            self.testset.initialize()
            batch = self.testset.next_batch()
            test_mean, test_std = self.model.execute([self.model.mean, self.model.std], batch)
            plot_functions(test_mean, test_std, batch, f'{save_dir}/test_fn')

        if 'imfns' in self.args.eval_metrics:
            # train set
            self.trainset.initialize()
            batch = self.trainset.next_batch()
            train_mean, train_std = self.model.execute([self.model.mean, self.model.std], batch)
            plot_img_functions(train_mean, train_std, batch, f'{save_dir}/train_fn')

            # valid set
            self.validset.initialize()
            batch = self.validset.next_batch()
            valid_mean, valid_std = self.model.execute([self.model.mean, self.model.std], batch)
            plot_img_functions(valid_mean, valid_std, batch, f'{save_dir}/valid_fn')

            # test set
            self.testset.initialize()
            batch = self.testset.next_batch()
            test_mean, test_std = self.model.execute([self.model.mean, self.model.std], batch)
            plot_img_functions(test_mean, test_std, batch, f'{save_dir}/test_fn')
