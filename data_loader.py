#!/usr/bin/env python
#coding=utf-8

# Most of the codes are from 
# https://github.com/vshallc/PtrNets/blob/master/pointer/misc/tsp.py
import os
import re
import zipfile
import itertools
import threading
import numpy as np
from tqdm import trange, tqdm
from collections import namedtuple

import tensorflow as tf

TSP = namedtuple('TSP', ['x', 'y', 'name'])

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))

class TSPDataLoader(object):
  def __init__(self, config, rng=None):
    self.config = config
    self.rng = rng

    self.task = config.task.lower()
    self.exp_name = config.exp_name

    self.batch_size = config.batch_size
    self.min_length = config.min_data_length
    self.max_length = config.max_data_length

    self.is_train = config.is_train
    self.input_dim = config.input_dim   # dimesions of training sample:x
    self.use_terminal_symbol = config.use_terminal_symbol
    self.random_seed = config.random_seed

    self.data_num = {}
    self.data_name = {}
    self.data_set = {}
    # self.train_num == len(self.train_set) should be guaranteed
    """如果data_set的npz文件存在，直接作为数据集；不存在则用date_name里面的原始数据集，生成npz文件作为数据集"""
    self.data_num['train'] = config.train_num
    self.data_name['train'] = config.train_set
    self.data_set['train'] = config.train_npz
    self.data_name['test'] = config.test_set
    self.data_num['test'] = config.test_num
    self.data_set['test'] = config.test_npz

    self.data_dir = config.data_dir
    self.task_name = "{}_{}_{}".format(
        self.task, self.min_length, self.max_length)

    self.data = None
    self.coord = None
    self.threads = None
    self.input_ops, self.target_ops = None, None
    self.queue_ops, self.enqueue_ops = None, None
    self.x, self.y, self.seq_length, self.mask = None, None, None, None

    self._my_generate_and_save()    # 读取数据集
    self._create_input_queue()      # 多线程

  def _create_input_queue(self, queue_capacity_factor=16):
    self.input_ops, self.target_ops = {}, {}
    self.queue_ops, self.enqueue_ops = {}, {}
    self.x, self.y, self.seq_length, self.mask = {}, {}, {}, {}

    for name in self.data_num.keys():
      self.input_ops[name] = tf.placeholder(tf.float32, shape=[None, None])
      self.target_ops[name] = tf.placeholder(tf.int32, shape=[None])

      min_after_dequeue = 1000
      capacity = min_after_dequeue + 3 * self.batch_size

      self.queue_ops[name] = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=[tf.float32, tf.int32],
          shapes=[[self.max_length, self.input_dim,], [self.max_length]],
          seed=self.random_seed,
          name="random_queue_{}".format(name))
      self.enqueue_ops[name] = \
          self.queue_ops[name].enqueue([self.input_ops[name], self.target_ops[name]])

      inputs, labels = self.queue_ops[name].dequeue()

      seq_length = tf.shape(inputs)[0]
      if self.use_terminal_symbol:
        mask = tf.ones([seq_length + 1], dtype=tf.float32) # terminal symbol
      else:
        mask = tf.ones([seq_length], dtype=tf.float32)

      self.x[name], self.y[name], self.seq_length[name], self.mask[name] = \
          tf.train.batch(
              [inputs, labels, seq_length, mask],
              batch_size=self.batch_size,
              capacity=capacity,
              dynamic_pad=True,
              name="batch_and_pad")

  def run_input_queue(self, sess):
    self.threads = []
    self.coord = tf.train.Coordinator()

    for name in self.data_num.keys():
      def load_and_enqueue(sess, name, input_ops, target_ops, enqueue_ops, coord):
        idx = 0
        while not coord.should_stop():
          feed_dict = {
              input_ops[name]: self.data[name].x[idx],
              target_ops[name]: self.data[name].y[idx],
          }
          sess.run(self.enqueue_ops[name], feed_dict=feed_dict)
          idx = idx+1 if idx+1 <= len(self.data[name].x) - 1 else 0

      args = (sess, name, self.input_ops, self.target_ops, self.enqueue_ops, self.coord)
      t = threading.Thread(target=load_and_enqueue, args=args)
      t.start()
      self.threads.append(t)
      tf.logging.info("Thread for [{}] start".format(name))

  def stop_input_queue(self):
    self.coord.request_stop()
    self.coord.join(self.threads)
    tf.logging.info("All threads stopped")

  def _generate_tsp_data(self,name,path):
      try:
        data = open(self.data_name[name],'r').read().strip().split('\n\n')
      except:
        raise Exception("[!] DataSet Not Found")
      data_length = len(data)
    
      x = np.zeros([data_length, self.max_length, self.input_dim], dtype=np.float32)
      y = np.zeros([data_length, self.max_length], dtype=np.int32)

      for i in range(data_length):
          nodes = np.array([[float(ele) for ele in item.split(',')] for item in data[i].split('\n')[0].split(' ')])
          res = np.array([int(item) + 1 for item in data[i].split('\n')[1].split(' ')[:-1]])
          x[i,:len(nodes)] = nodes
          y[i,:len(res)] = res

      np.savez(path, x=x, y=y)  # save in a npz file

      self.data[name] = TSP(x=x, y=y, name=name)

  def _my_generate_and_save(self,):
      # load data from local path, save in npz file
      self.data = {}
      
      for name in self.data_num.keys():
        path = self.data_set[name]
        if not os.path.exists(path):
            self._generate_tsp_data(name, path)
        else:
            tmp = np.load(path)
            self.data[name] = TSP(x=tmp['x'], y=tmp['y'], name=name)

