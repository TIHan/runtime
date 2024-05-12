import os
import re
import collections
import numpy as np
import statistics
import tensorflow as tf
import json
import itertools
import mljit_superpmi
import mljit_metrics
import mljit_trainer
import mljit_runner
import mljit_utils
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

# This was from MLGO, but not really sure what it is.
class ConstantValueNetwork(network.Network):
  """Constant value network that predicts zero per batch item."""

  def __init__(self, input_tensor_spec, constant_output_val=0, name=None):
    """Creates an instance of `ConstantValueNetwork`.

    Network supports calls with shape outer_rank + observation_spec.shape. Note
    outer_rank must be at least 1.

    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
        representing the input observations.
      constant_output_val: A constant scalar value the network will output.
      name: A string representing name of the network.

    Raises:
      ValueError: If input_tensor_spec is not an instance of network.InputSpec.
    """
    super().__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    self._constant_output_val = constant_output_val

  def call(self, inputs, step_type=None, network_state=(), training=False):
    _ = (step_type, training)
    shape = nest_utils.get_outer_shape(inputs, self._input_tensor_spec)
    constant = tf.constant(self._constant_output_val, tf.float32)
    return tf.fill(shape, constant), network_state