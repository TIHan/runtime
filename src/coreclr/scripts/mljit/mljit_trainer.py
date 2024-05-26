import os
import re
import collections
import numpy as np
import statistics
import tensorflow as tf
import json
import itertools
import functools
import mljit_metrics
import mljit_utils
import mljit_superpmi

from collections import defaultdict
from typing import Any, List, Sequence, Tuple, Callable, Optional, Dict
from types import SimpleNamespace
from tf_agents.metrics import tf_metrics
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.trajectories import time_step, trajectory
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.typing import types
from tf_agents.networks import q_network
from tf_agents.networks import network
from tf_agents.networks import q_rnn_network
from tf_agents.utils import nest_utils
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.policies import PolicySaver, random_tf_policy, py_tf_policy
from tf_agents.utils import common as common_utils
from tf_agents.environments import tf_environment
from tf_agents import specs
from tf_agents.utils import common
from absl import logging
from dataclasses import dataclass

def create_policy_saver(agent):
    return PolicySaver(agent.policy, batch_size=1, use_nest_path_signatures=False)

def create_collect_policy_saver(agent):
    return PolicySaver(agent.collect_policy, batch_size=1, use_nest_path_signatures=False)

def save_policy(policy_saver, path):
    print(f"[mljit] Saving policy in '{path}'...")
    policy_saver.save(path)
    print(f"[mljit] Saved policy in '{path}'!")

def compute_dataset(parse, sequence_examples, train_sequence_length, batch_size, trajectory_shuffle_buffer_size):
    return tf.data.Dataset.from_tensor_slices(sequence_examples).map(parse).unbatch().batch(train_sequence_length, drop_remainder=True).cache().shuffle(trajectory_shuffle_buffer_size).batch(batch_size, drop_remainder=True)

def create_dataset(parse, sequence_examples, train_sequence_length, batch_size, trajectory_shuffle_buffer_size):
    return compute_dataset(parse, sequence_examples, train_sequence_length, batch_size, trajectory_shuffle_buffer_size)

def compute_dataset_special(parse, sequence_examples, train_sequence_length, batch_size):
    return tf.data.Dataset.from_tensor_slices(sequence_examples).map(parse).unbatch().batch(train_sequence_length, drop_remainder=True).cache().shuffle(1024 * 1024).batch(batch_size, drop_remainder=True)

def create_dataset_special(parse, sequence_examples, train_sequence_length, batch_size):
    return compute_dataset_special(parse, sequence_examples, train_sequence_length, batch_size)

def create_dataset_iter(dataset):
    return iter(dataset.repeat().prefetch(tf.data.AUTOTUNE))

def create_non_repeat_dataset_iter(dataset):
    return iter(dataset.prefetch(tf.data.AUTOTUNE))

@dataclass
class JitTrainSettings:
    train_sequence_length: int
    batch_size: int
    trajectory_shuffle_buffer_size: int

class JitTrainer:
    def __init__(self, saved_policy_path, saved_collect_policy_path, agent, create_trajectories, parse):
        self.agent = agent
        self.policy_saver = create_policy_saver(agent)
        self.collect_policy_saver = create_collect_policy_saver(agent)
        self.step = tf.compat.v1.train.get_or_create_global_step()
        self.create_trajectories = create_trajectories
        self.parse = parse
        self.saved_policy_path = saved_policy_path
        self.saved_collect_policy_path = saved_collect_policy_path

    def save_policy(self):
        save_policy(self.policy_saver, self.saved_policy_path)
        save_policy(self.collect_policy_saver, self.saved_collect_policy_path)

    def train(self, jit_metrics: mljit_metrics.JitTensorBoardMetrics, data: Sequence[mljit_superpmi.Method], num_epochs, train_settings: Optional[JitTrainSettings]=None):

        if train_settings is None:
            lookup = defaultdict(list)
            for x in data:
                lookup[len(x.log)].append(x)

            datasets = []
            for k, v in lookup.items():

                train_sequence_length = k
                batch_size = len(v)

                if k == 1:
                    train_sequence_length = 2
                    batch_size = int(len(v) / 2)
                    if batch_size < 1:
                        batch_size = 1

                sequence_examples = self.create_trajectories(v)
                datasets = datasets + [create_dataset_special(self.parse, sequence_examples, train_sequence_length, batch_size)]

            dataset = mljit_utils.functools.reduce(lambda x, y: x.concatenate(y), datasets)
            dataset = dataset.shuffle(1024 * 1024, reshuffle_each_iteration=True)
        else:
            sequence_examples = self.create_trajectories(data)
            dataset = create_dataset(self.parse, sequence_examples, train_settings.train_sequence_length, train_settings.batch_size, train_settings.trajectory_shuffle_buffer_size)

        jit_metrics.reset()

        print('[mljit] Training...')           
        count = 0
        for _ in range(num_epochs):
            for experience in dataset:
                with tf.summary.record_if(lambda: tf.math.equal(self.step % 1000, 0)):
                    self.agent.train(experience)
                jit_metrics.add_experience(experience)
            count = count + 1
            mljit_utils.print_progress_bar(count, num_epochs)
        jit_metrics.update_experience(self.step)
        jit_metrics.update_trajectories(data, self.step)