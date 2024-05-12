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
    def __init__(self, jit_metrics: mljit_metrics.JitTensorBoardMetrics, jit_trainer: mljit_trainer.JitTrainer, collect_data, num_max_steps=1000000):
        self.jit_metrics = jit_metrics
        self.jit_trainer = jit_trainer
        self.num_max_steps = num_max_steps
        self.collect_data = collect_data

    def run(self, partitioned_methods):

        self.jit_trainer.save_policy()

        episode_count = 0
        while self.jit_trainer.step.numpy() < self.num_max_steps:
            print(f'[mljit] Current step: {self.jit_trainer.step.numpy()}')
            print(f'[mljit] Current episode: {episode_count}')

          #  partition_index = episode_count % len(partitioned_methods)
          #  print(f'[mljit] Current partition index: {partition_index}')

            methods = partitioned_methods[0] # TODO: For now, it just gets the first item.
            data = self.collect_data(methods)
            self.jit_trainer.train(self.jit_metrics, data)
            self.jit_trainer.save_policy()