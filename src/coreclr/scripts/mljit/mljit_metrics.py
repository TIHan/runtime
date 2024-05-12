import os
import re
import collections
import numpy as np
import statistics
import tensorflow as tf
import json
import itertools
import functools

from dataclasses import dataclass
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

QUANTILE_MONITOR = (0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.5, 99.9)

def create_distribution_monitor(data: Sequence[float]) -> Dict[str, float]:
  if not data:
    return {}
  quantiles = np.percentile(data, QUANTILE_MONITOR, method='lower')
  monitor_dict = {
      f'p_{x}': y for (x, y) in zip(QUANTILE_MONITOR, quantiles)
  }
  monitor_dict['mean'] = np.mean(data)
  return monitor_dict

def create_method_trajectories_monitor(data: Sequence[Any]):
    total_trajectory_length = sum(len(res.log) for res in data)

    monitor_dict = {}
    monitor_dict['default'] = {
        'success_functions': len(data),
        'total_trajectory_length': total_trajectory_length,
    }
    rewards = list(itertools.chain.from_iterable([list(map(lambda x: x.reward, res.log)) for res in data]))
    monitor_dict['reward_distribution'] = create_distribution_monitor(rewards)
    
    return monitor_dict

class JitTensorBoardMetrics:
    def __init__(self, path):
        train_summary_writer = tf.summary.create_file_writer(path, flush_millis=10000)
        train_summary_writer.set_as_default()
        summary_log_interval = 100

        data_action_mean = tf.keras.metrics.Mean()
        data_reward_mean = tf.keras.metrics.Mean()
        num_trajectories = tf.keras.metrics.Sum()
        num_improvements = tf.keras.metrics.Sum()
        num_regressions = tf.keras.metrics.Sum()
        improvement_score = tf.keras.metrics.Sum()
        regression_score = tf.keras.metrics.Sum()

        self.train_summary_writer = train_summary_writer
        self.summary_log_interval = summary_log_interval
        self.data_action_mean = data_action_mean
        self.data_reward_mean = data_reward_mean
        self.num_trajectories = num_trajectories
        self.num_improvements = num_improvements
        self.num_regressions = num_regressions
        self.improvement_score = improvement_score
        self.regression_score = regression_score

    def reset(self):
        self.num_trajectories.reset_states()
        self.num_improvements.reset_states()
        self.num_regressions.reset_states()
        self.improvement_score.reset_states()
        self.regression_score.reset_states()

    def update_improvements_and_regressions(self, num_improvements, num_regressions, improvement_score, regression_score, step):
        if tf.math.equal(step % self.summary_log_interval, 0):
            self.num_improvements.update_state(num_improvements)
            self.num_regressions.update_state(num_regressions)
            self.improvement_score.update_state(improvement_score)
            self.regression_score.update_state(regression_score)

        if tf.summary.should_record_summaries():
            with tf.name_scope('jit/'):
                tf.summary.scalar(
                    name='num_improvements',
                    data=self.num_improvements.result(),
                    step=step)
                tf.summary.scalar(
                    name='num_regressions',
                    data=self.num_regressions.result(),
                    step=step)
                tf.summary.scalar(
                    name='improvement_score',
                    data=self.improvement_score.result(),
                    step=step)
                tf.summary.scalar(
                    name='regression_score',
                    data=self.regression_score.result(),
                    step=step)

    def update(self, data: Sequence[Any], step, experience):
        """Updates metrics and exports to Tensorboard."""
        if tf.math.equal(step % self.summary_log_interval, 0):
            is_action = ~experience.is_boundary()

            self.data_action_mean.update_state(
                experience.action, sample_weight=is_action)
            self.data_reward_mean.update_state(
                experience.reward, sample_weight=is_action)
            self.num_trajectories.update_state(experience.is_first())

        # Check earlier rather than later if we should record summaries.
        # TF also checks it, but much later. Needed to avoid looping through
        # the dict so gave the if a bigger scope
        if tf.summary.should_record_summaries():
            with tf.name_scope('default/'):
                tf.summary.scalar(
                    name='data_action_mean',
                    data=self.data_action_mean.result(),
                    step=step)
                tf.summary.scalar(
                    name='data_reward_mean',
                    data=self.data_reward_mean.result(),
                    step=step)
                tf.summary.scalar(
                    name='num_trajectories',
                    data=self.num_trajectories.result(),
                    step=step)

            monitor = create_method_trajectories_monitor(data)
            for name_scope, d in monitor.items():
                with tf.name_scope(name_scope + '/'):
                    for key, value in d.items():
                        tf.summary.scalar(name=key, data=value, step=step)

            tf.summary.histogram(name='reward', data=experience.reward, step=step)