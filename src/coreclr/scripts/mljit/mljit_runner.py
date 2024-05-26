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
import mljit_utils

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
    def __init__(self, jit_trainer: mljit_trainer.JitTrainer, collect_data, collect_data_no_training, num_epochs, num_episodes, train_settings):
        self.jit_trainer = jit_trainer   
        self.collect_data = collect_data
        self.collect_data_no_training = collect_data_no_training
        self.num_epochs = num_epochs
        self.train_settings = train_settings
        self.num_episodes = num_episodes

    def run(self, jit_metrics: mljit_metrics.JitTensorBoardMetrics, train_data, test_data):

        self.jit_trainer.save_policy()

        episode_count = 0
        while episode_count < self.num_episodes:
            print(f'[mljit] Current step: {self.jit_trainer.step.numpy()}')
            print(f'[mljit] Current episode: {episode_count}')

            print('[mljit] Collecting train data...')
            self.jit_trainer.train(jit_metrics, self.collect_data(train_data), num_epochs=self.num_epochs, train_settings=self.train_settings)
            self.jit_trainer.save_policy()

            print('[mljit] Collecting test data for comparisons...')
            test_methods = self.collect_data_no_training(test_data)

            num_improvements = 0
            num_regressions = 0
            improvement_score = 0
            regression_score = 0

            for curr in test_methods:
                for base in test_data:
                    if base.spmi_index == curr.spmi_index:
                        if curr.perf_score < base.perf_score:
                            num_improvements = num_improvements + 1
                            improvement_score = improvement_score + (base.perf_score - curr.perf_score)
                        elif curr.perf_score > base.perf_score:
                            num_regressions = num_regressions + 1
                            regression_score = regression_score + (curr.perf_score - base.perf_score)

            jit_metrics.update_improvements_and_regressions(num_improvements, num_regressions, improvement_score, regression_score, self.jit_trainer.step)
            episode_count = episode_count + 1