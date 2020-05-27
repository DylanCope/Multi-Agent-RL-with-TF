# Multi-Agent Reinforcement Learning with TF-Agents

In this notebook we're going to be implementing reinforcement learning (RL) agents to play games against one another. Before reading this it is advised to be familiar with the [TF-Agents](https://github.com/tensorflow/agents) and Deep Q-Learning; [this tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb) will bring you up to speed.

## Introduction

TF-Agents is a framework for designing and experimenting with RL algorithms. It provides a collection of useful abstractions such as agents, replay buffers, and drivers. However, the code is quite rigidly tied to the single-agent view, which is explained by the *extrinsically motivated* agent in the diagram below.

In this view, the environment provides observations and rewards to the agent. Under the assumption that there is only one agent this makes sense, however, when we have many agents in the same space we would like to have agent-specific observations and rewards. In order to rectify this we first need to think of agents as *intrinsically motivated*, which is to say that their rewards are a function of their observations and internal state. Secondly, the agent is only *partially observing* the environment, and the window into the environment is a function of the agent's total state. This total state can include "physical" properties of the agent such as position, but it also includes internal state. For example, an agent could have an internal `is_sleeping` parameter that multiplies their observations by zero to simulate a lack of light.

## Implementing the IMAgent

In order to implement this with TF-Agents we are going to define an `IMAgent` (Intrinsically Motivated Agent) class by overriding the `DqnAgent` class. In the standard TF-Agents DQN pipeline the agent is trained by alternating between data collection and training updates to the Q-Network. Data collection is done with a special `collect_policy` which behaves differently to the main policy for the sake of managing the exploitation-exploration trade-off. Usually, the environment and the agent are separated. The environment generates a `TimeStep` containing the observation and reward information which is then passed to `policy.action`. This produces a `PolicyStep` that contains an action to step the environment. 

<img src="./im_rl_agent.png" width="600px" display="block" margin-left="auto" margin-right="auto"/>

This provides us with two approaches to our problem. We could make the enviroment aware of which agent it is producing the `TimeStep` for, or we could have each agent ingest an agent-independent time step that is then augmented internally. Here we argue that the latter is a more natural decomposition as it keeps the agent-specific code with the agent class. 


```python
from functools import partial
from pathlib import Path
import random
from time import time
from typing import Tuple, List, Callable

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tf_agents.agents import DqnAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs import TensorSpec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Trajectory

print('Physical Devices:\n', tf.config.list_physical_devices(), '\n\n')

OUTPUTS_DIR = f'./outputs/{int(10000000 * time())}'
print('Output Directory:', OUTPUTS_DIR)
```

    Physical Devices:
     [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')] 
    
    
    Output Directory: ./outputs/15905145823503500
    


```python
class IMAgent(DqnAgent):

    def __init__(self,
                 env: TFPyEnvironment,
                 observation_spec: TensorSpec = None,
                 action_spec: TensorSpec = None,
                 reward_fn: Callable = lambda time_step: time_step.reward,
                 action_fn: Callable = lambda action: action,
                 name: str='IMAgent',
                 q_network=None,
                 # training params
                 replay_buffer_max_length: int = 1000,
                 learning_rate: float = 1e-5,
                 training_batch_size: int = 8,
                 training_parallel_calls: int = 3,
                 training_prefetch_buffer_size: int = 3,
                 training_num_steps: int = 2,
                 **dqn_kwargs):

        self._env = env
        self._reward_fn = reward_fn
        self._name = name
        self._observation_spec = observation_spec or self._env.observation_spec()
        self._action_spec = action_spec or self._env.action_spec()
        self._action_fn = action_fn

        q_network = q_network or self._build_q_net()

        env_ts_spec = self._env.time_step_spec()
        time_step_spec = TimeStep(
            step_type=env_ts_spec.step_type,
            reward=env_ts_spec.reward,
            discount=env_ts_spec.discount,
            observation=q_network.input_tensor_spec
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        super().__init__(time_step_spec,
                         self._action_spec,
                         q_network,
                         optimizer,
                         name=name,
                         **dqn_kwargs)

        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size)

        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self.collect_data_spec,
            batch_size=self._env.batch_size,
            max_length=replay_buffer_max_length)

        dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=training_parallel_calls,
            sample_batch_size=training_batch_size,
            num_steps=training_num_steps
        ).prefetch(training_prefetch_buffer_size)

        self._training_data_iter = iter(dataset)

    def _build_q_net(self):
        qrnn = QRnnNetwork(input_tensor_spec=self._observation_spec,
                           action_spec=self._action_spec,
                           name=f'{self._name}QRNN')

        qrnn.create_variables()
        qrnn.summary()

        return qrnn

    def reset_state(self):
        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size)
        
    def _observation_fn(self, observation: tf.Tensor) -> tf.Tensor:
        """
            Takes a tensor with specification self._env.observation_spec
            and extracts a tensor with specification self._observation_spec.
            
            For example, consider an agent within an NxN maze environment. 
            The env could expose the entire NxN integer matrix as an observation
            but we would prefer the agent to only see a 3x3 window around their
            current location. To do this we can override this method.
            
            This allows us to have different agents acting in the same environment
            with different observations.
        """
        return observation

    def _augment_time_step(self, time_step: TimeStep) -> TimeStep:

        reward = self._reward_fn(time_step)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        if reward.shape != time_step.reward.shape:
            reward = tf.reshape(reward, time_step.reward.shape)
            
        observation = self._observation_fn(time_step.observation)

        return TimeStep(
            step_type=time_step.step_type,
            reward=reward,
            discount=time_step.discount,
            observation=observation
        )

    def _current_time_step(self) -> TimeStep:
        time_step = self._env.current_time_step()
        time_step = self._augment_time_step(time_step)
        return time_step

    def _step_environment(self, action) -> TimeStep:
        action = self._action_fn(action)
        time_step = self._env.step(action)
        time_step = self._augment_time_step(time_step)
        return time_step

    def act(self, collect=False) -> Trajectory:
        time_step = self._current_time_step()

        if collect:
            policy_step = self.collect_policy.action(
                time_step, policy_state=self._policy_state)
        else:
            policy_step = self.policy.action(
                time_step, policy_state=self._policy_state)

        self._policy_state = policy_step.state
        next_time_step = self._step_environment(policy_step.action)
        traj = trajectory.from_transition(time_step, policy_step, next_time_step)

        if collect:
            self._replay_buffer.add_batch(traj)

        return traj

    def train_iteration(self) -> LossInfo:
        experience, buffer_info = next(self._training_data_iter)
        return self.train(experience)
```

## Tic-Tac-Toe Example

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Tic_tac_toe.svg/1200px-Tic_tac_toe.svg.png" width="300px"/>

In order to test this we can utlise the [already-implemented Tic-Tac-Toe environment](https://github.com/tensorflow/agents/blob/master/tf_agents/environments/examples/tic_tac_toe_environment.py) in TF-Agents (At the time of writing this script has not been added to the pip distribution so I have manually copied it across). The environment represents the problem on a 3x3 matrix where a 0 represents an empty slot, a 1 represents a play by player 1, and a 2 represents a play by player 2. However, as TF-Agents is not focused on the multi-agent case, their implementation has the second player act randomly. To change this we will override the step function.

The only additional change that we need to make is to the action specification, where we need to provide the value that is being placed (i.e. which player is making the move).


```python
from tic_tac_toe_environment import TicTacToeEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType

class TicTacToeMultiAgentEnv(TicTacToeEnvironment):
    
    def action_spec(self):
        position_spec = BoundedArraySpec((1,), np.int32, minimum=0, maximum=8)
        value_spec = BoundedArraySpec((1,), np.int32, minimum=1, maximum=2)
        return {
            'position': position_spec,
            'value': value_spec
        }
    
    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        index_flat = np.array(range(9)) == action['position']
        index = index_flat.reshape(self._states.shape) == True
        if self._states[index] != 0:
            return TimeStep(StepType.LAST, 
                            TicTacToeEnvironment.REWARD_ILLEGAL_MOVE,
                            self._discount, 
                            self._states)

        self._states[index] = action['value']

        is_final, reward = self._check_states(self._states)
        
        if np.all(self._states == 0):
            step_type = StepType.FIRST
        elif is_final:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return TimeStep(step_type, reward, self._discount, self._states)
```


```python
def print_tic_tac_toe(state):
    table_str = '''
    {} | {} | {}
    - + - + -
    {} | {} | {}
    - + - + -
    {} | {} | {}
    '''.format(*tuple(state.flatten()))
    table_str = table_str.replace('0', ' ')
    table_str = table_str.replace('1', 'X')
    table_str = table_str.replace('2', 'O')
    print(table_str)
```


```python
tic_tac_toe_env = TicTacToeMultiAgentEnv()

ts = tic_tac_toe_env.reset()
print('Reward:', ts.reward, 'Board:')
print_tic_tac_toe(ts.observation)

random.seed(1)
player = 1
while not ts.is_last():
    action = {
        'position': np.asarray(random.randint(0, 8)),
        'value': player
    }
    ts = tic_tac_toe_env.step(action)
    print('Player:', player, 'Action:', action['position'],
          'Reward:', ts.reward, 'Board:')
    print_tic_tac_toe(ts.observation)
    player = 1 + player % 2
```

    Reward: 0.0 Board:
    
          |   |  
        - + - + -
          |   |  
        - + - + -
          |   |  
        
    Player: 1 Action: 2 Reward: 0.0 Board:
    
          |   | X
        - + - + -
          |   |  
        - + - + -
          |   |  
        
    Player: 2 Action: 1 Reward: 0.0 Board:
    
          | O | X
        - + - + -
          |   |  
        - + - + -
          |   |  
        
    Player: 1 Action: 4 Reward: 0.0 Board:
    
          | O | X
        - + - + -
          | X |  
        - + - + -
          |   |  
        
    Player: 2 Action: 1 Reward: -0.001 Board:
    
          | O | X
        - + - + -
          | X |  
        - + - + -
          |   |  
        
    


```python
def ttt_action_fn(player, action):
    return {'position': action, 'value': player}

tf_ttt_env = TFPyEnvironment(tic_tac_toe_env)

player_1 = IMAgent(
    tf_ttt_env,
    action_spec = tic_tac_toe_env.action_spec()['position'],
    action_fn = partial(ttt_action_fn, 1),
    name='Player1'
)

player_2 = IMAgent(
    tf_ttt_env,
    action_spec = tic_tac_toe_env.action_spec()['position'],
    action_fn = partial(ttt_action_fn, 2),
    reward_fn = lambda ts: -1.0 if ts.reward == 1.0 else ts.reward,
    name='Player2'
)
```

    Model: "Player1QRNN"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    EncodingNetwork (EncodingNet multiple                  3790      
    _________________________________________________________________
    dynamic_unroll_3 (DynamicUnr multiple                  12960     
    _________________________________________________________________
    Player1QRNN/dense (Dense)    multiple                  3075      
    _________________________________________________________________
    Player1QRNN/dense (Dense)    multiple                  3040      
    _________________________________________________________________
    num_action_project/dense (De multiple                  369       
    =================================================================
    Total params: 23,234
    Trainable params: 23,234
    Non-trainable params: 0
    _________________________________________________________________
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-140-f71a51eddc48> in <module>
          8     action_spec = tic_tac_toe_env.action_spec()['position'],
          9     action_fn = partial(ttt_action_fn, 1),
    ---> 10     name='Player1'
         11 )
         12 
    

    <ipython-input-108-6f6da4a896bc> in __init__(self, env, observation_spec, action_spec, reward_fn, action_fn, name, q_network, replay_buffer_max_length, learning_rate, training_batch_size, training_parallel_calls, training_prefetch_buffer_size, training_num_steps, **dqn_kwargs)
         41                          optimizer,
         42                          name=name,
    ---> 43                          **dqn_kwargs)
         44 
         45         self._policy_state = self.policy.get_initial_state(
    

    ~\Miniconda3\envs\xai-it\lib\site-packages\gin\config.py in wrapper(*args, **kwargs)
       1030         scope_info = " in scope '{}'".format(scope_str) if scope_str else ''
       1031         err_str = err_str.format(name, fn, scope_info)
    -> 1032         utils.augment_exception_message_and_reraise(e, err_str)
       1033 
       1034     return wrapper
    

    ~\Miniconda3\envs\xai-it\lib\site-packages\gin\utils.py in augment_exception_message_and_reraise(exception, message)
         47   if six.PY3:
         48     ExceptionProxy.__qualname__ = type(exception).__qualname__
    ---> 49     six.raise_from(proxy.with_traceback(exception.__traceback__), None)
         50   else:
         51     six.reraise(proxy, None, sys.exc_info()[2])
    

    ~\Miniconda3\envs\xai-it\lib\site-packages\six.py in raise_from(value, from_value)
    

    ~\Miniconda3\envs\xai-it\lib\site-packages\gin\config.py in wrapper(*args, **kwargs)
       1007 
       1008       try:
    -> 1009         return fn(*new_args, **new_kwargs)
       1010       except Exception as e:  # pylint: disable=broad-except
       1011         err_str = ''
    

    ~\Miniconda3\envs\xai-it\lib\site-packages\tf_agents\agents\dqn\dqn_agent.py in __init__(self, time_step_spec, action_spec, q_network, optimizer, observation_and_action_constraint_splitter, epsilon_greedy, n_step_update, boltzmann_temperature, emit_log_probability, target_q_network, target_update_tau, target_update_period, td_errors_loss_fn, gamma, reward_scale_factor, gradient_clipping, debug_summaries, summarize_grads_and_vars, train_step_counter, name)
        206     tf.Module.__init__(self, name=name)
        207 
    --> 208     self._check_action_spec(action_spec)
        209 
        210     if epsilon_greedy is not None and boltzmann_temperature is not None:
    

    ~\Miniconda3\envs\xai-it\lib\site-packages\tf_agents\agents\dqn\dqn_agent.py in _check_action_spec(self, action_spec)
        264 
        265     # TODO(oars): Get DQN working with more than one dim in the actions.
    --> 266     if len(flat_action_spec) > 1 or flat_action_spec[0].shape.rank > 1:
        267       raise ValueError('Only one dimensional actions are supported now.')
        268 
    

    AttributeError: 'tuple' object has no attribute 'rank'
      In call to configurable 'DqnAgent' (<function DqnAgent.__init__ at 0x00000240FC493E18>)



```python

```
