import os
import re
import collections
import numpy as np
import statistics
import tensorflow as tf
from typing import Any, List, Sequence, Tuple, Callable, Optional, Dict
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
from tf_agents.policies import PolicySaver
from tf_agents.utils import common as common_utils
from absl import logging

# # Prevents a warning
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

saved_policy_path = os.environ['DOTNET_MLJIT_SAVED_POLICY_PATH']

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

observation_spec_and_preprocessing_layers = [
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_cost_ex'), 
        tf.keras.layers.Rescaling(scale=1000, offset=0, dtype=tf.int64)),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_use_count_weighted_log'), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_def_count_weighted_log'), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_cost_sz'), 
        tf.keras.layers.Rescaling(scale=1000, offset=0, dtype=tf.int64)),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_use_count'), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_def_count'), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_live_across_call', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_int', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_constant_not_shared', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_shared_constant', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_cost_is_MIN_CSE_COST', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_constant_live_across_call', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_constant_min_cost', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_cost_is_MIN_CSE_COST_live_across_call', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_GTF_MAKE_CSE', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_num_distinct_locals'), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='cse_num_local_occurrences'), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_has_call', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='log_cse_use_count_weighted_times_cost_ex'), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='log_cse_use_count_weighted_times_num_local_occurrences'), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.float32, shape=(), name='cse_distance'), 
        tf.keras.layers.Rescaling(scale=1000, offset=0, dtype=tf.int64)), # (max postorder num - min postorder num) / num BBs
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_containable', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_cheap_containable', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tensor_spec.BoundedTensorSpec(dtype=tf.int64, shape=(), name='cse_is_live_across_call_in_LSRA_ordering', minimum=0, maximum=1), 
        tf.keras.layers.Identity()),
    (tf.TensorSpec(dtype=tf.int64, shape=(), name='log_pressure_estimated_weight'), 
        tf.keras.layers.Identity())
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

preprocessing_combiner = tf.keras.layers.Add()

actor_network = actor_distribution_network.ActorDistributionNetwork(
    input_tensor_spec=time_step_spec.observation,
    output_tensor_spec=action_spec,
    preprocessing_layers=preprocessing_layers,
    preprocessing_combiner=preprocessing_combiner,
    
    # Settings below match MLGO, most of the settings are actually the default values of ActorDistributionNetwork.
    fc_layer_params=(40, 40, 20),
    dropout_layer_params=None,
    activation_fn=tf.keras.activations.relu)

# value_network = value_network.ValueNetwork(
#   time_step_spec.observation,
#   preprocessing_layers=preprocessing_layers,
#   preprocessing_combiner=preprocessing_combiner)

value_network = ConstantValueNetwork(time_step_spec.observation)

agent = ppo_agent.PPOAgent(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    actor_net=actor_network,
    value_net=value_network,

    # Settings below match MLGO, most of the settings are actually the default values of PPOAgent.
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003, epsilon=0.0003125),
    importance_ratio_clipping=0.2,
    lambda_value=0.0,
    discount_factor=0.0,
    entropy_regularization=0.003,
    policy_l2_reg=0.000001,
    value_function_l2_reg=0.0,
    shared_vars_l2_reg=0.0,
    value_pred_loss_coef=0.0,
    num_epochs=1,
    use_gae=False,
    use_td_lambda_return=False,
    normalize_rewards=False,
    reward_norm_clipping=10.0,
    normalize_observations=False,
    log_prob_clipping=0.0,
    kl_cutoff_factor=2.0,
    kl_cutoff_coef=1000.0,
    initial_adaptive_kl_beta=1.0,
    adaptive_kl_target=0.01,
    adaptive_kl_tolerance=0.3,
    gradient_clipping=None,
    value_clipping=None,
    check_numerics=False,
    compute_value_and_advantage_in_train=True,
    update_normalizers_in_train=True,
    debug_summaries=True,
    summarize_grads_and_vars=True)

agent.initialize()
agent.train = common_utils.function(agent.train) # Apparently, it makes 'train' faster? Who knows why...


# ---------------------------------------

def step_action(step_type: int, reward: float, discount: float, observation: list):
    return agent.collect_policy.action(time_step.TimeStep(
        step_type=tf.constant(step_type, dtype=tf.int32),
        reward=tf.constant(reward, dtype=tf.float32),
        discount=tf.constant(discount, dtype=tf.float32),
        observation=observation
    ))

# TODO: not working
# action_step = step_action(0, 0.0, 0.0, [
#    tf.constant(0.0, dtype=tf.float32, name='cse_cost_ex'),
#    tf.constant(0, dtype=tf.int64, name='cse_use_count_weighted_log')
# ])

# ---------------------------------------

many_cse_cost_ex = tf.train.FeatureList(feature=[
    tf.train.Feature(float_list=tf.train.FloatList(
        value=[0.0]))
])

many_cse_use_count_weighted_log = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_def_count_weighted_log = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_cost_sz = tf.train.FeatureList(feature=[
    tf.train.Feature(float_list=tf.train.FloatList(
        value=[0.0]))
])

many_cse_use_count = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_def_count = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_live_across_call = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_int = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_constant_not_shared = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_constant_not_shared = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_shared_constant = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_cost_is_MIN_CSE_COST = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_constant_live_across_call = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_constant_min_cost = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_cost_is_MIN_CSE_COST_live_across_call = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_GTF_MAKE_CSE = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_num_distinct_locals = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_num_local_occurrences = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_has_call = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_log_cse_use_count_weighted_times_cost_ex = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_log_cse_use_count_weighted_times_num_local_occurrences = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_distance = tf.train.FeatureList(feature=[
    tf.train.Feature(float_list=tf.train.FloatList(
        value=[0.0]))
])

many_cse_is_containable = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_cheap_containable = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_is_live_across_call_in_LSRA_ordering = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_log_pressure_estimated_weight = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_cse_decision = tf.train.FeatureList(feature=[
    tf.train.Feature(int64_list=tf.train.Int64List(
        value=[0]))
])

many_reward = tf.train.FeatureList(feature=[
    tf.train.Feature(float_list=tf.train.FloatList(
        value=[0.0]))
])

many_logits = tf.train.FeatureList(feature=[
    tf.train.Feature(float_list=tf.train.FloatList(
        value=[0.0, 0.0]))
])

feature_dict = {
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
  'CategoricalProjectionNetwork_logits': many_logits
}

feature_lists = tf.train.FeatureLists(feature_list=feature_dict)

example = tf.train.SequenceExample(context={},feature_lists=feature_lists)

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
          reward=reward)
      return full_trajectory
    
# ---------------------------------------

# Settings from MLGO.
num_iterations                 = 300
batch_size                     = 256
train_sequence_length          = 16 # We have to have 2 or more for PPOAgent to work.
trajectory_shuffle_buffer_size = 1024

def compute_dataset(sequence_examples):
    return tf.data.Dataset.from_tensor_slices(sequence_examples).map(parse).unbatch().batch(train_sequence_length, drop_remainder=True).cache().shuffle(trajectory_shuffle_buffer_size).batch(batch_size, drop_remainder=True)

def create_dataset_iter(sequence_examples):
    return iter(compute_dataset(sequence_examples).repeat().prefetch(tf.data.AUTOTUNE))

# Majority of this is from MLGO.
def train(sequence_examples):
    dataset_iter = create_dataset_iter(sequence_examples)
    for _ in range(num_iterations):
        # When the data is not enough to fill in a batch, next(dataset_iter)
        # will throw StopIteration exception, logging a warning message instead
        # of killing the training when it happens.
        try:
            experience = next(dataset_iter)
        except StopIteration:
            logging.warning(
                ('Warning: skip training because do not have enough data to fill '
                'in a batch, consider increase data or reduce batch size.'))
            break

        agent.train(experience)

# ---------------------------------------

test_sequence_examples = [example.SerializeToString() for _ in range(train_sequence_length * batch_size)]
train(test_sequence_examples)

# saved_model_cli show --dir saved_model\ --tag_set serve --signature_def input
# save policy
policy_saver = PolicySaver(agent.policy, batch_size=1, use_nest_path_signatures=False)

print(f"[mljit] Saving model in '{saved_policy_path}'...")
policy_saver.save(saved_policy_path)
print(f"[mljit] Saved model in '{saved_policy_path}'!")