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
import mljit_trainer

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

class JitRunner:
    def __init__(self, jit_metrics: mljit_metrics.JitTensorBoardMetrics, jit_trainer: mljit_trainer.JitTrainer, collect_data, collect_data_no_training, step_size, train_sequence_length, batch_size, trajectory_shuffle_buffer_size, num_max_steps):
        self.jit_metrics = jit_metrics
        self.jit_trainer = jit_trainer   
        self.collect_data = collect_data
        self.collect_data_no_training = collect_data_no_training
        self.step_size = step_size
        self.train_sequence_length = train_sequence_length
        self.batch_size = batch_size
        self.trajectory_shuffle_buffer_size = trajectory_shuffle_buffer_size
        self.num_max_steps = num_max_steps

    def run(self, baseline_methods):

        self.jit_trainer.save_policy()

        episode_count = 0
        while self.jit_trainer.step.numpy() < self.num_max_steps:
            print(f'[mljit] Current step: {self.jit_trainer.step.numpy()}')
            print(f'[mljit] Current episode: {episode_count}')

            print('[mljit] Collecting data...')
            methods = self.collect_data(baseline_methods)
            self.jit_trainer.train(self.jit_metrics, methods, step_size=self.step_size, train_sequence_length=self.train_sequence_length, batch_size=self.batch_size, trajectory_shuffle_buffer_size=self.trajectory_shuffle_buffer_size)
            self.jit_trainer.save_policy()

            print('[mljit] Collecting data for comparisons...')
            methods = self.collect_data_no_training(baseline_methods)

            num_improvements = 0
            num_regressions = 0
            improvement_score = 0
            regression_score = 0

            for base in baseline_methods:
                for curr in methods:
                    if base.spmi_index == curr.spmi_index:
                        if curr.perf_score < base.perf_score:
                            num_improvements = num_improvements + 1
                            improvement_score = improvement_score + (base.perf_score - curr.perf_score)
                        elif curr.perf_score > base.perf_score:
                            num_regressions = num_regressions + 1
                            regression_score = regression_score + (curr.perf_score - base.perf_score)

            self.jit_metrics.update_improvements_and_regressions(num_improvements, num_regressions, improvement_score, regression_score, self.jit_trainer.step)