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
import mljit_tf
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
from tf_agents.policies import PolicySaver, random_tf_policy, py_tf_policy, fixed_policy
from tf_agents.utils import common as common_utils
from tf_agents.environments import tf_environment
from tf_agents import specs
from tf_agents.utils import common
from absl import logging
from tf_agents.policies import policy_loader

# Use 'saved_model_cli show --dir saved_policy\ --tag_set serve --signature_def action' from the command line to see the inputs/outputs of the policy.

corpus_file_path          = os.environ['DOTNET_MLJitCorpusFile']
saved_policy_path         = os.environ['DOTNET_MLJitSavedPolicyPath']
saved_collect_policy_path = os.environ['DOTNET_MLJitSavedCollectPolicyPath']
log_path                  = os.environ['DOTNET_MLJitLogPath']

warmstart_policy_path = os.path.join(log_path, 'warmstart_policy/')

@dataclass
class CseLogItem:
    cse_index: int
    cse_cost_ex: float
    cse_use_count_weighted_log: float
    cse_def_count_weighted_log: float
    cse_cost_sz: float
    cse_use_count: int
    cse_def_count: int
    cse_is_live_across_call: int
    cse_is_int: int
    cse_is_constant_not_shared: int
    cse_is_shared_constant: int
    cse_cost_is_MIN_CSE_COST: int
    cse_is_constant_live_across_call: int
    cse_is_constant_min_cost: int
    cse_cost_is_MIN_CSE_COST_live_across_call: int
    cse_is_GTF_MAKE_CSE: int
    cse_num_distinct_locals: int
    cse_num_local_occurrences: int
    cse_has_call: int
    log_cse_use_count_weighted_times_cost_ex: float
    log_cse_use_count_weighted_times_num_local_occurrences: float
    cse_distance: float
    cse_is_containable: int
    cse_is_cheap_containable: int
    cse_is_live_across_call_in_LSRA_ordering: int
    log_pressure_estimated_weight: int
    CategoricalProjectionNetwork_logits: Sequence[float]
    cse_decision: int
    reward: float
  
float32_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))
int64_layer = tf.keras.layers.Lambda(lambda x: tf.cast(tf.expand_dims(x, -1), tf.float32))

observation_spec_and_preprocessing_layers = [
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_index', minimum=0, maximum=1), 
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

observation_spec = mljit_utils.map_dict_value(lambda x: x[0], observation_spec_and_preprocessing_layers)
preprocessing_layers = mljit_utils.map_dict_value(lambda x: x[1], observation_spec_and_preprocessing_layers)

reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)
action_spec = tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_decision', minimum=0, maximum=1)

preprocessing_combiner = tf.keras.layers.Concatenate()

def create_bc_agent(use_actor_network=False):
    if use_actor_network:
        network = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec=time_step_spec.observation,
            output_tensor_spec=action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(40, 40, 20),
            dropout_layer_params=(0.2, 0.2, 0.2))
    else:
        network = q_network.QNetwork(
            input_tensor_spec=time_step_spec.observation,
            action_spec=action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(40, 40, 20),
            dropout_layer_params=(0.2, 0.2, 0.2))

    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        cloning_network=network,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.0003125),
        num_outer_dims=2,
        debug_summaries=True,
        summarize_grads_and_vars=True)

    agent.initialize()
    agent.train = common_utils.function(agent.train) # Apparently, it makes 'train' faster? Who knows why...
    return agent

def create_ppo_agent(use_real_critic=False):
    actor_network = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=(40, 40, 20))

    if use_real_critic:
        learning_rate = 0.001
    else:
        learning_rate = 0.001#0.00003

    epsilon                   = 0.0003125
    entropy_regularization    = 0.01
    importance_ratio_clipping = 0.2
    policy_l2_reg             = 0.000001

    if use_real_critic:
        critic_network = value_network.ValueNetwork(
            time_step_spec.observation,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner)

        agent = ppo_agent.PPOAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_net=actor_network,
            value_net=critic_network,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
            importance_ratio_clipping=importance_ratio_clipping,
            entropy_regularization=entropy_regularization,
            policy_l2_reg=policy_l2_reg,
            num_epochs=1,
            normalize_observations=True,
            normalize_rewards=True,
            debug_summaries=True,
            summarize_grads_and_vars=True)
    else:
        critic_network = mljit_tf.ConstantValueNetwork(time_step_spec.observation)

        agent = ppo_agent.PPOAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_net=actor_network,
            value_net=critic_network,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
            importance_ratio_clipping=importance_ratio_clipping,
            entropy_regularization=entropy_regularization,
            policy_l2_reg=policy_l2_reg,
            num_epochs=1,
            normalize_observations=True,
            normalize_rewards=True,
            debug_summaries=True,
            summarize_grads_and_vars=True,

            # This is needed for the constant value network.
            value_function_l2_reg=0.0,
            value_pred_loss_coef=0.0,
            lambda_value=0.0,
            discount_factor=0.0)

    agent.initialize()
    agent.train = common_utils.function(agent.train) # Apparently, it makes 'train' faster? Who knows why...
    return agent

# ---------------------------------------

def create_sequence_example(log, use_behavioral_cloning):

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

    if use_behavioral_cloning:
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
            'reward': many_reward
        }

    else:
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

def create_serialized_sequence_example(data, use_behavioral_cloning):
   return create_sequence_example(data, use_behavioral_cloning).SerializeToString()

# ---------------------------------------

def get_policy_info_parsing_dict(use_behavioral_cloning) -> Dict[str, tf.io.FixedLenSequenceFeature]:
    if use_behavioral_cloning:
        return {}
    
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

def process_parsed_sequence_and_get_policy_info(parsed_sequence: Dict[str, Any], use_behavioral_cloning) -> Dict[str, Dict[str, Any]]:
    if use_behavioral_cloning:
        return {}
    
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
def parse(serialized_proto, use_behavioral_cloning):
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
    sequence_features.update(get_policy_info_parsing_dict(use_behavioral_cloning))

    # pylint: enable=g-complex-comprehension
    with tf.name_scope('parse'):
      _, parsed_sequence = tf.io.parse_single_sequence_example(
          serialized_proto,
          context_features=context_features,
          sequence_features=sequence_features)

      # TODO(yundi): make the transformed reward configurable.
      action = parsed_sequence[action_spec.name]
      reward = tf.cast(parsed_sequence[time_step_spec.reward.name], tf.float32)

      policy_info = process_parsed_sequence_and_get_policy_info(parsed_sequence, use_behavioral_cloning)
      
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

def collect_data(corpus_file_path, baseline, train_kind):
    if baseline is None:
        return []
    else:
        data = mljit_superpmi.collect_data(corpus_file_path, list(map(lambda x: x.spmi_index, baseline)), train_kind=train_kind)

        # Calculate and update rewards.
        # Only do this for exploration.
        if train_kind == 2:
            baseline_dict = dict()

            for x in baseline:
                baseline_dict[x.spmi_index] = x

            for item in data:
                item_base = baseline_dict[item.spmi_index]
                reward = (1.0 - (item.perf_score / item_base.perf_score))
                for log_item in item.log:
                    log_item.reward = reward

        return data
    
def create_trajectories(data: Sequence[Any], use_behavioral_cloning):
    print(f'[mljit] Bundling...')
    acc = []
    for x in data:
        acc = acc + [x.log]

    print(f'[mljit] Creating trajectories: {len(acc)}...')
    return list(map(lambda x: create_serialized_sequence_example(x, use_behavioral_cloning), acc))

def create_one_trajectory(data: Sequence[Any], use_behavioral_cloning):
    print(f'[mljit] Bundling...')
    acc = []
    for x in data:
        acc = acc + x.log
    acc = [acc]

    print(f'[mljit] Creating one trajectory...')
    return list(map(lambda x: create_serialized_sequence_example(x, use_behavioral_cloning), acc))

# ---------------------------------------

if not mljit_superpmi.mldump_file_exists():
    print('[mljit] Producing mldump.txt...')
    mljit_superpmi.produce_mldump_file(corpus_file_path)
    print('[mljit] Finished producing mldump.txt')

def filter_cse_methods(m):
    return m.num_cse_candidates > 0 and m.perf_score > 0

baseline = mljit_superpmi.parse_mldump_file_filter(filter_cse_methods)

# ---------------------------------------

# Training

def collect_data_no_training(x):
    return collect_data(corpus_file_path, x, train_kind=1)

def create_trajectories_for_bc(x):
    return create_one_trajectory(x, use_behavioral_cloning=True)

def parse_for_bc(x):
    return parse(x, use_behavioral_cloning=True)

def collect_data_for_bc(x):
    return collect_data(corpus_file_path, x, train_kind=0)

def create_trajectories_for_ppo(x):
    return create_one_trajectory(x, use_behavioral_cloning=False)

def parse_for_ppo(x):
    return parse(x, use_behavioral_cloning=False)

def collect_data_for_ppo(x):
    return collect_data(corpus_file_path, x, train_kind=2)

build_warmstart = False
use_warmstart = False

def create_default_log_item():
    return CseLogItem(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [0.0, 0.0], 0, 0)

if build_warmstart:
    # BC Training

    agent = create_bc_agent(use_actor_network=True)

    jit_metrics = mljit_metrics.JitTensorBoardMetrics(log_path)
    jit_trainer = mljit_trainer.JitTrainer(create_default_log_item,
                                           saved_policy_path, 
                                           saved_collect_policy_path, 
                                           agent, 
                                           create_trajectories=create_trajectories_for_bc, 
                                           parse=parse_for_bc)
    
    train_settings = mljit_trainer.JitTrainSettings(train_sequence_length=1, batch_size=64, trajectory_shuffle_buffer_size=1024)
    jit_runner  = mljit_runner.JitRunner(jit_trainer,
                                         collect_data=collect_data_for_bc, 
                                         collect_data_no_training=collect_data_no_training, 
                                         num_epochs=100000,
                                         num_episodes=1,
                                         train_settings=train_settings)

    jit_runner.run(jit_metrics, train_data=baseline, test_data=[])

    mljit_trainer.save_policy(jit_trainer.policy_saver, warmstart_policy_path)
else:
    # PPO Training

    agent = create_ppo_agent(use_real_critic=False)

    if use_warmstart:
        agent.policy.update(policy_loader.load(warmstart_policy_path))
    #else:
        # In the beginning, force the policy to be incentivised to return 'false' for CSE decisions.
        # Anecdotally, the policy trains better when it starts making 'false' decisions more than 'true' decisions.
        # agent.collect_policy.update(fixed_policy.FixedPolicy(tf.constant(0, dtype=tf.int64), agent.time_step_spec, agent.action_spec))

    jit_metrics = mljit_metrics.JitTensorBoardMetrics(log_path)
    jit_trainer = mljit_trainer.JitTrainer(create_default_log_item,
                                           saved_policy_path, 
                                           saved_collect_policy_path, 
                                           agent, 
                                           create_trajectories=create_trajectories_for_ppo, 
                                           parse=parse_for_ppo)
    
    train_settings = mljit_trainer.JitTrainSettings(train_sequence_length=16, batch_size=256, trajectory_shuffle_buffer_size=1024)
    jit_runner  = mljit_runner.JitRunner(jit_trainer, 
                                         collect_data=collect_data_for_ppo, 
                                         collect_data_no_training=collect_data_no_training, 
                                         num_epochs=100, 
                                         num_episodes=100,
                                         train_settings=train_settings)
    
    partitioned = mljit_utils.partition(baseline, 10000)

    jit_runner.run(jit_metrics, train_data=baseline, test_data=partitioned[1][:1000])

# ---------------------------------------

print(f'[mljit] Finished!')