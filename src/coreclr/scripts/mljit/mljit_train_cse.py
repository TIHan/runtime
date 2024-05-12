import os
import re
import collections
import numpy as np
import statistics
import tensorflow as tf
import json
import itertools
import mljit_superpmi
import functools

import matplotlib
import matplotlib.pyplot as plt

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

def for_all(predicate, xs):
    return all(predicate(x) for x in xs)

def partition_yield(xs, size):
    for i in range(0, len(xs), size):
        yield list(itertools.islice(xs, i, i + size))

def partition(xs, size):
    return list(partition_yield(xs, size))

# From https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

plt.ion()

# Use 'saved_model_cli show --dir saved_policy\ --tag_set serve --signature_def action' from the command line to see the inputs/outputs of the policy.

corpus_file_path  = os.environ['DOTNET_MLJitCorpusFile']
saved_policy_path = os.environ['DOTNET_MLJitSavedPolicyPath']
saved_collect_policy_path = os.environ['DOTNET_MLJitSavedCollectPolicyPath']
log_path = os.environ['DOTNET_MLJitLogPath']

def flatten(xss):
    return [x for xs in xss for x in xs]

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
  
float32_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))

int64_layer = tf.keras.layers.Lambda(lambda x: tf.cast(tf.expand_dims(x, -1), tf.float32))

observation_spec_and_preprocessing_layers = [
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_index'), 
        int64_layer),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_cost_ex'), 
        float32_layer),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_use_count_weighted_log'), 
        float32_layer),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_def_count_weighted_log'), 
        float32_layer),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_cost_sz'), 
        float32_layer),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_use_count'), 
        int64_layer),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_def_count'), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_live_across_call', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_int', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_constant_not_shared', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_shared_constant', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_cost_is_MIN_CSE_COST', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_constant_live_across_call', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_constant_min_cost', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_cost_is_MIN_CSE_COST_live_across_call', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_GTF_MAKE_CSE', minimum=0, maximum=1), 
        int64_layer),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_num_distinct_locals'), 
        int64_layer),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_num_local_occurrences'), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_has_call', minimum=0, maximum=1), 
        int64_layer),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='log_cse_use_count_weighted_times_cost_ex'), 
        float32_layer),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='log_cse_use_count_weighted_times_num_local_occurrences'), 
        float32_layer),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_distance'), 
        float32_layer), # (max postorder num - min postorder num) / num BBs
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_containable', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_cheap_containable', minimum=0, maximum=1), 
        int64_layer),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_live_across_call_in_LSRA_ordering', minimum=0, maximum=1), 
        int64_layer),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='log_pressure_estimated_weight'), 
        int64_layer)
]

observation_spec_and_preprocessing_layers = list(map(lambda x: (x[0].name, x), observation_spec_and_preprocessing_layers))
observation_spec_and_preprocessing_layers = dict(observation_spec_and_preprocessing_layers)

def map_dict(f, my_dictionary):
   return {k: f(k, v) for k, v in my_dictionary.items()}

def map_dict_value(f, my_dictionary):
   return {k: f(v) for k, v in my_dictionary.items()}

observation_spec = map_dict_value(lambda x: x[0], observation_spec_and_preprocessing_layers)
preprocessing_layers = map_dict_value(lambda x: x[1], observation_spec_and_preprocessing_layers)

reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)
action_spec = tensor_spec.BoundedTensorSpec(
     dtype=tf.int64, shape=(), name='cse_decision', minimum=0, maximum=1)

preprocessing_combiner = tf.keras.layers.Concatenate()

def create_agent():
    actor_network = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
       # activation_fn=tf.keras.activations.softmax,
        fc_layer_params=(40, 40, 20))

    # critic_network = value_network.ValueNetwork(
    #   time_step_spec.observation,
    #   preprocessing_layers=preprocessing_layers,
    #   preprocessing_combiner=preprocessing_combiner)

    critic_network = ConstantValueNetwork(time_step_spec.observation)

    agent = ppo_agent.PPOAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_net=actor_network,
        value_net=critic_network,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003, epsilon=0.0003125),
        importance_ratio_clipping=0.2,
        lambda_value=0.0,
        discount_factor=0.0,
        entropy_regularization=0.01,
        policy_l2_reg=0.000001,
        num_epochs=1,
        value_function_l2_reg=0.0,
        value_pred_loss_coef=0.0,
        normalize_observations=False,
        normalize_rewards=False,
      #  reward_norm_clipping=1.0,
        debug_summaries=True,
        summarize_grads_and_vars=True)

    agent.initialize()
    agent.train = common_utils.function(agent.train) # Apparently, it makes 'train' faster? Who knows why...
    return agent

# ---------------------------------------

def create_sequence_example(log):

    def log_to_feature_int64(x, f):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[np.int64(f(x))]))
    
    def log_to_feature_float(x, f):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[np.float32(f(x))]))
    
    def log_to_feature_float_2(xs, f):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[np.float32(x) for x in f(xs)]))
    
    def logs_to_features_int64(xs, f):
        return [log_to_feature_int64(x, f) for x in xs]
    
    def logs_to_features_float(xs, f):
        return [log_to_feature_float(x, f) for x in xs]
    
    def logs_to_features_float_2(xs, f):
        return [log_to_feature_float_2(x, f) for x in xs]
    
    many_cse_index = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_index))

    many_cse_cost_ex = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.cse_cost_ex))

    many_cse_use_count_weighted_log = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.cse_use_count_weighted_log))

    many_cse_def_count_weighted_log = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.cse_def_count_weighted_log))

    many_cse_cost_sz = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.cse_cost_sz))

    many_cse_use_count = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_use_count))

    many_cse_def_count = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_def_count))

    many_cse_is_live_across_call = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_live_across_call))

    many_cse_is_int = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_int))

    many_cse_is_constant_not_shared = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_constant_not_shared))

    many_cse_is_shared_constant = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_shared_constant))

    many_cse_cost_is_MIN_CSE_COST = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_cost_is_MIN_CSE_COST))

    many_cse_is_constant_live_across_call = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_constant_live_across_call))

    many_cse_is_constant_min_cost = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_constant_min_cost))

    many_cse_cost_is_MIN_CSE_COST_live_across_call = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_cost_is_MIN_CSE_COST_live_across_call))

    many_cse_is_GTF_MAKE_CSE = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_GTF_MAKE_CSE))

    many_cse_num_distinct_locals = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_num_distinct_locals))

    many_cse_num_local_occurrences = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_num_local_occurrences))

    many_cse_has_call = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_has_call))

    many_log_cse_use_count_weighted_times_cost_ex = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.log_cse_use_count_weighted_times_cost_ex))

    many_log_cse_use_count_weighted_times_num_local_occurrences = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.log_cse_use_count_weighted_times_num_local_occurrences))

    many_cse_distance = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.cse_distance))

    many_cse_is_containable = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_containable))

    many_cse_is_cheap_containable = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_cheap_containable))

    many_cse_is_live_across_call_in_LSRA_ordering = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_is_live_across_call_in_LSRA_ordering))

    many_log_pressure_estimated_weight = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.log_pressure_estimated_weight))

    many_cse_decision = tf.train.FeatureList(feature=logs_to_features_int64(log, lambda x: x.cse_decision))

    many_reward = tf.train.FeatureList(feature=logs_to_features_float(log, lambda x: x.reward))

    many_CategoricalProjectionNetwork_logits = tf.train.FeatureList(feature=logs_to_features_float_2(log, lambda x: x.CategoricalProjectionNetwork_logits))

    feature_dict = {
        'cse_index': many_cse_index,
        'cse_cost_ex': many_cse_cost_ex,
        'cse_use_count_weighted_log': many_cse_use_count_weighted_log,
        'cse_def_count_weighted_log': many_cse_def_count_weighted_log,
        'cse_cost_sz': many_cse_cost_sz,
        'cse_use_count': many_cse_use_count,
        'cse_def_count': many_cse_def_count,
        'cse_is_live_across_call': many_cse_is_live_across_call,
        'cse_is_int': many_cse_is_int,
        'cse_is_constant_not_shared': many_cse_is_constant_not_shared,
        'cse_is_shared_constant': many_cse_is_shared_constant,
        'cse_cost_is_MIN_CSE_COST': many_cse_cost_is_MIN_CSE_COST,
        'cse_is_constant_live_across_call': many_cse_is_constant_live_across_call,
        'cse_is_constant_min_cost': many_cse_is_constant_min_cost,
        'cse_cost_is_MIN_CSE_COST_live_across_call': many_cse_cost_is_MIN_CSE_COST_live_across_call,
        'cse_is_GTF_MAKE_CSE': many_cse_is_GTF_MAKE_CSE,
        'cse_num_distinct_locals': many_cse_num_distinct_locals,
        'cse_num_local_occurrences': many_cse_num_local_occurrences,
        'cse_has_call': many_cse_has_call,
        'log_cse_use_count_weighted_times_cost_ex': many_log_cse_use_count_weighted_times_cost_ex,
        'log_cse_use_count_weighted_times_num_local_occurrences': many_log_cse_use_count_weighted_times_num_local_occurrences,
        'cse_distance': many_cse_distance,
        'cse_is_containable': many_cse_is_containable,
        'cse_is_cheap_containable': many_cse_is_cheap_containable,
        'cse_is_live_across_call_in_LSRA_ordering': many_cse_is_live_across_call_in_LSRA_ordering,
        'log_pressure_estimated_weight': many_log_pressure_estimated_weight,
        'cse_decision': many_cse_decision,
        'reward': many_reward,
        'CategoricalProjectionNetwork_logits': many_CategoricalProjectionNetwork_logits
    }

    feature_lists = tf.train.FeatureLists(feature_list=feature_dict)

    return tf.train.SequenceExample(context={},feature_lists=feature_lists)

def create_serialized_sequence_example(data):
   return create_sequence_example(data).SerializeToString()

# ---------------------------------------

def get_policy_info_parsing_dict() -> Dict[str, tf.io.FixedLenSequenceFeature]:
    if tensor_spec.is_discrete(action_spec):
      return {
          'CategoricalProjectionNetwork_logits':
              tf.io.FixedLenSequenceFeature(
                  shape=(action_spec.maximum - action_spec.minimum +
                         1),
                  dtype=tf.float32)
      }
    else:
      return {
          'NormalProjectionNetwork_scale':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32),
          'NormalProjectionNetwork_loc':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
      }

def process_parsed_sequence_and_get_policy_info(parsed_sequence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if tensor_spec.is_discrete(action_spec):
        policy_info = {
            'dist_params': {
                'logits': parsed_sequence['CategoricalProjectionNetwork_logits']
            }
        }
        del parsed_sequence['CategoricalProjectionNetwork_logits']
    else:
        policy_info = {
            'dist_params': {
                'scale': parsed_sequence['NormalProjectionNetwork_scale'],
                'loc': parsed_sequence['NormalProjectionNetwork_loc']
            }
        }
        del parsed_sequence['NormalProjectionNetwork_scale']
        del parsed_sequence['NormalProjectionNetwork_loc']
    return policy_info

# From MLGO.
def parse(serialized_proto):
    # We copy through all context features at each frame, so even though we know
    # they don't change from frame to frame, they are still sequence features
    # and stored in the feature list.
    context_features = {}
    # pylint: disable=g-complex-comprehension
    sequence_features = dict(
        (tensor_spec.name,
         tf.io.FixedLenSequenceFeature(
             shape=tensor_spec.shape, dtype=tensor_spec.dtype))
        for tensor_spec in time_step_spec.observation.values())
    sequence_features[
        action_spec.name] = tf.io.FixedLenSequenceFeature(
            shape=action_spec.shape,
            dtype=action_spec.dtype)
    sequence_features[
        time_step_spec.reward.name] = tf.io.FixedLenSequenceFeature(
            shape=time_step_spec.reward.shape,
            dtype=time_step_spec.reward.dtype)
    sequence_features.update(get_policy_info_parsing_dict())

    # pylint: enable=g-complex-comprehension
    with tf.name_scope('parse'):
      _, parsed_sequence = tf.io.parse_single_sequence_example(
          serialized_proto,
          context_features=context_features,
          sequence_features=sequence_features)

      # TODO(yundi): make the transformed reward configurable.
      action = parsed_sequence[action_spec.name]
      reward = tf.cast(parsed_sequence[time_step_spec.reward.name],
                       tf.float32)

      policy_info = process_parsed_sequence_and_get_policy_info(
          parsed_sequence)
      
      del parsed_sequence[time_step_spec.reward.name]
      del parsed_sequence[action_spec.name]
      full_trajectory = trajectory.from_episode(
          observation=parsed_sequence,
          action=action,
          policy_info=policy_info,
          reward=reward,
          discount=None)
      return full_trajectory
    
# ---------------------------------------

global_step = tf.compat.v1.train.get_or_create_global_step()

train_summary_writer = tf.summary.create_file_writer(log_path, flush_millis=10000)
train_summary_writer.set_as_default()
summary_log_interval = 100

data_action_mean = tf.keras.metrics.Mean()
data_reward_mean = tf.keras.metrics.Mean()
num_trajectories = tf.keras.metrics.Sum()

def update_metrics(experience, monitor_dict):
    """Updates metrics and exports to Tensorboard."""
    if tf.math.equal(global_step % summary_log_interval, 0):
      is_action = ~experience.is_boundary()

      data_action_mean.update_state(
          experience.action, sample_weight=is_action)
      data_reward_mean.update_state(
          experience.reward, sample_weight=is_action)
      num_trajectories.update_state(experience.is_first())

    # Check earlier rather than later if we should record summaries.
    # TF also checks it, but much later. Needed to avoid looping through
    # the dict so gave the if a bigger scope
    if tf.summary.should_record_summaries():
      with tf.name_scope('default/'):
        tf.summary.scalar(
            name='data_action_mean',
            data=data_action_mean.result(),
            step=global_step)
        tf.summary.scalar(
            name='data_reward_mean',
            data=data_reward_mean.result(),
            step=global_step)
        tf.summary.scalar(
            name='num_trajectories',
            data=num_trajectories.result(),
            step=global_step)

      for name_scope, d in monitor_dict.items():
        with tf.name_scope(name_scope + '/'):
          for key, value in d.items():
            tf.summary.scalar(name=key, data=value, step=global_step)

      tf.summary.histogram(
          name='reward', data=experience.reward, step=global_step)
      
def reset_metrics():
    """Reset num_trajectories."""
    num_trajectories.reset_states()

# ---------------------------------------

num_max_steps = 1000000
step_size     = 1000

def compute_dataset(sequence_examples, train_sequence_length, batch_size, trajectory_shuffle_buffer_size):
    return tf.data.Dataset.from_tensor_slices(sequence_examples).map(parse).unbatch().batch(train_sequence_length, drop_remainder=True).cache().shuffle(trajectory_shuffle_buffer_size).batch(batch_size, drop_remainder=True)

def create_dataset(sequence_examples, train_sequence_length, batch_size, trajectory_shuffle_buffer_size):
    return compute_dataset(sequence_examples, train_sequence_length, batch_size, trajectory_shuffle_buffer_size)

def create_dataset_iter(dataset):
    return iter(dataset.repeat().prefetch(tf.data.AUTOTUNE))

# Majority of this is from MLGO.
def train(agent, dataset, monitor_dict):
    reset_metrics()
    with tf.summary.record_if(lambda: tf.math.equal(
        global_step % 1000, 0)):

        print('[mljit] Training...')
        dataset_iter = create_dataset_iter(dataset)
        count = 0
        for _ in range(step_size):
            # When the data is not enough to fill in a batch, next(dataset_iter)
            # will throw StopIteration exception, logging a warning message instead
            # of killing the training when it happens.
            try:
                experience = next(dataset_iter)
                #tf.print(experience, summarize=64)
                count = count + 1
            except StopIteration:
                logging.warning(
                    ('[mljit] ERROR: Skipped training because do not have enough data to fill '
                    'in a batch, consider increase data or reduce batch size.'))
                break

            loss = agent.train(experience)
            update_metrics(experience, monitor_dict)
            printProgressBar(count, step_size)

def create_policy_saver(agent):
    return PolicySaver(agent.policy, batch_size=1, use_nest_path_signatures=False)

def create_collect_policy_saver(agent):
    return PolicySaver(agent.collect_policy, batch_size=1, use_nest_path_signatures=False)

def save_policy(policy_saver, path):
    print(f"[mljit] Saving policy in '{path}'...")
    policy_saver.save(path)
    print(f"[mljit] Saved policy in '{path}'!")

@dataclass
class LogItem:
    cse_index: any
    cse_cost_ex: any 
    cse_use_count_weighted_log: any 
    cse_def_count_weighted_log: any 
    cse_cost_sz: any 
    cse_use_count: any 
    cse_def_count: any 
    cse_is_live_across_call: any  
    cse_is_int: any 
    cse_is_constant_not_shared: any  
    cse_is_shared_constant: any 
    cse_cost_is_MIN_CSE_COST: any 
    cse_is_constant_live_across_call: any  
    cse_is_constant_min_cost: any 
    cse_cost_is_MIN_CSE_COST_live_across_call: any  
    cse_is_GTF_MAKE_CSE: any 
    cse_num_distinct_locals: any 
    cse_num_local_occurrences: any 
    cse_has_call: any 
    log_cse_use_count_weighted_times_cost_ex: any 
    log_cse_use_count_weighted_times_num_local_occurrences: any 
    cse_distance: any  
    cse_is_containable: any 
    cse_is_cheap_containable: any 
    cse_is_live_across_call_in_LSRA_ordering: any 
    log_pressure_estimated_weight: any 
    cse_decision: any 
    reward: any 
    CategoricalProjectionNetwork_logits: any

REWARD_QUANTILE_MONITOR = (0.1, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 50,
                           60, 70, 80, 90, 95, 99, 99.5, 99.9)

def build_distribution_monitor(data: Sequence[float]) -> Dict[str, float]:
  if not data:
    return {}
  quantiles = np.percentile(data, REWARD_QUANTILE_MONITOR, method='lower')
  monitor_dict = {
      f'p_{x}': y for (x, y) in zip(REWARD_QUANTILE_MONITOR, quantiles)
  }
  monitor_dict['mean'] = np.mean(data)
  return monitor_dict

def collect_data(corpus_file_path, baseline, best_state, train_kind=1):
    indices = flatten(list(map(lambda x: [x.spmi_index], baseline)))

    print('[mljit] Collecting data...')
    data = mljit_superpmi.collect_data(corpus_file_path, indices, train_kind=train_kind)
    print('[mljit] Calculating rewards...')

    for i in range(len(baseline)):
        item_base = baseline[i]

        spmi_index = item_base.spmi_index

        item_best = best_state[spmi_index]

        for item in data:
            if item.spmi_index == spmi_index:

                # Reward Method 1
                # reward = (item_best.perf_score - item.perf_score) / item_base.perf_score

                # Reward Method 2
                # reward = (1.0 - (item.perf_score / item_base.perf_score))

                # Reward Method 3
                if item.perf_score < item_base.perf_score:
                    reward = 1.0
                elif item.perf_score > item_base.perf_score:
                    reward = -1.0
                else:
                    reward = 0.0

                # Clamp
                if reward > 1.0:
                    reward = 1.0
                elif reward < -1.0:
                    reward = -1.0

                for i in range(len(item.log)):
                    item.log[i].reward = reward

                if item.perf_score < item_best.perf_score:
                    item_best = item
                elif item.perf_score == item_best.perf_score and item.num_cse_usages < item_best.num_cse_usages:
                    item_best = item

        best_state[spmi_index] = item_best

    return data

def create_trajectories(data):
    total_trajectory_length = sum(len(res.log) for res in data)

    monitor_dict = {}
    monitor_dict['default'] = {
        'success_functions': len(data),
        'total_trajectory_length': total_trajectory_length,
    }
    rewards = list(
        itertools.chain.from_iterable(
            [list(map(lambda x: x.reward, res.log)) for res in data]))
    monitor_dict[
        'reward_distribution'] = build_distribution_monitor(rewards)
    
    acc = []
    for x in data:
        acc = acc + [x.log]

    print(f'[mljit] Creating trajectories: {len(acc)}...')
    return list(map(create_serialized_sequence_example, acc)), monitor_dict

# ---------------------------------------

if not mljit_superpmi.mldump_file_exists():
    print('[mljit] Producing mldump.txt...')
    mljit_superpmi.produce_mldump_file(corpus_file_path)
    print('[mljit] Finished producing mldump.txt')

def filter_cse_methods(m):
    return m.is_valid and m.num_cse_candidates > 0 and m.perf_score > 0

baseline = mljit_superpmi.parse_mldump_file_filter(filter_cse_methods)

partitioned_baseline = partition(baseline, 2000)

eval_baseline = partitioned_baseline[0]
eval_indices = list(map(lambda x: x.spmi_index, eval_baseline))

# ---------------------------------------

# Training

eval_only = False

agent = create_agent()
policy_saver = create_policy_saver(agent)
collect_policy_saver = create_collect_policy_saver(agent)

if not eval_only:
    # Save initial policy.
    save_policy(collect_policy_saver, saved_collect_policy_path)
    save_policy(policy_saver, saved_policy_path)

def plot_results(data_step_num, data_num_improvements, data_num_regressions):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Step')
    plt.plot(data_step_num, data_num_improvements, label = "Improvements")
    plt.plot(data_step_num, data_num_regressions, label = "Regressions")
    plt.legend()
    plt.pause(0.0001)

# Compare Results
print_verbose_results = False
def compare_results(data_step_num, data_num_improvements, data_num_regressions, step_num, baseline_indices):
    print('[mljit] Collecting data for comparison...')
    policy_result = mljit_superpmi.collect_data(corpus_file_path, baseline_indices, train_kind=2) # policy
    print('[mljit] Comparing results...')

    total_perf_score_improvement_diff = 0.0
    total_perf_score_regression_diff = 0.0

    num_improvements = 0
    num_regressions = 0
    for i in range(len(policy_result)):
        curr = policy_result[i]
        for j in range(len(baseline)):
            base = baseline[j]

            if curr.spmi_index == base.spmi_index:
                best = best_state[curr.spmi_index]

                if print_verbose_results:
                    print("")
                    print(f'spmi index {curr.spmi_index}')
                if curr.perf_score < base.perf_score:       
                    num_improvements = num_improvements + 1
                    total_perf_score_improvement_diff = total_perf_score_improvement_diff + (base.perf_score - curr.perf_score)
                elif curr.perf_score > base.perf_score:
                    num_regressions = num_regressions + 1
                    total_perf_score_regression_diff = total_perf_score_regression_diff + (curr.perf_score - base.perf_score)

                if curr.perf_score < best.perf_score:
                    best_state[curr.spmi_index] = curr

                if print_verbose_results:
                    for k in range(len(curr.log)):
                        best_item = best.log[k]
                        curr_item = curr.log[k]
                        base_item = base.log[k]
                        print(f'best decision: {best_item.cse_decision}, base decision: {base_item.cse_decision}, curr decision: {curr_item.cse_decision}')
                    print(f'best score: {best.perf_score}, base score: {base.perf_score}, curr score: {curr.perf_score}')

    print('----- Evaluation Policy Results ----')
    print(f'Step:              {step_num}')
    print(f'Total Methods:     {len(baseline)}')
    print(f'Improvements:      {num_improvements}')
    print(f'Improvement Score: {total_perf_score_improvement_diff}')
    print(f'Regressions:       {num_regressions}')
    print(f'Regression Score:  {total_perf_score_regression_diff}')
    print('------------------------------------')

    data_step_num = data_step_num + [int(step_num)]
    data_num_improvements = data_num_improvements + [num_improvements]
    data_num_regressions = data_num_regressions + [num_regressions]

    plot_results(data_step_num, data_num_improvements, data_num_regressions)
    return (data_step_num, data_num_improvements, data_num_regressions)

print(f'[mljit] Setting up baseline...')
eval_baseline = mljit_superpmi.collect_data(corpus_file_path, eval_indices, train_kind=0)

best_state = dict()
for x in baseline:
    best_state[x.spmi_index] = x

data_step_num = []
data_num_improvements = []
data_num_regressions = []

will_compare_results = True

if will_compare_results:
    (new_data_step_num, new_data_num_improvements, new_data_num_regressions) = compare_results(data_step_num, data_num_improvements, data_num_regressions, 0, eval_indices)
    data_step_num = new_data_step_num
    data_num_improvements = new_data_num_improvements
    data_num_regressions = new_data_num_regressions

if not eval_only:
    episode_count = 0
    best_ratio = -1000000.0
    while global_step.numpy() < num_max_steps:
        print(f'[mljit] Best ratio: {best_ratio}')
        print(f'[mljit] Current step: {global_step.numpy()}')
        print(f'[mljit] Current episode: {episode_count}')

        partition_index = episode_count % len(partitioned_baseline)
        print(f'[mljit] Current partition index: {partition_index}')

        baseline_methods = eval_baseline#partitioned_baseline[partition_index]

        data = collect_data(corpus_file_path, baseline_methods, best_state)
        sequence_examples, monitor_dict = create_trajectories(data)
        dataset = create_dataset(sequence_examples, train_sequence_length=16, batch_size=256, trajectory_shuffle_buffer_size=512)
        train(agent, dataset, monitor_dict)

        if not eval_only:
            save_policy(policy_saver, saved_policy_path)
            save_policy(collect_policy_saver, saved_collect_policy_path)

        if will_compare_results:
            (new_data_step_num, new_data_num_improvements, new_data_num_regressions) = compare_results(data_step_num, data_num_improvements, data_num_regressions, global_step.numpy(), eval_indices)
            num_improvements = new_data_num_improvements[len(new_data_num_improvements) - 1]
            num_regressions = new_data_num_regressions[len(new_data_num_regressions) - 1]

            if not eval_only:
                ratio = float(num_improvements)
                if num_regressions != 0:
                    ratio = float(num_improvements) / float(num_regressions)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_policy_saver = create_policy_saver(agent)
                    save_policy(best_policy_saver, os.path.join(saved_policy_path, '../best_policy'))
            data_step_num = new_data_step_num
            data_num_improvements = new_data_num_improvements
            data_num_regressions = new_data_num_regressions

        episode_count = episode_count + 1

# ---------------------------------------

print(f'[mljit] Finished! Best ratio: {best_ratio}')
plt.ioff()
plt.show()