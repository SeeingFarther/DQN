# RL Project

This repository contains code for a Reinforcement Learning project. The project includes an implementation of various RL algorithms and allows for easy customization of hyperparameters and agent configurations through command-line arguments.

## Installation

Since `argparse` is part of the standard Python library, it should already be installed. However, if it’s not, you can install it using the following command:

pip install argparse

## Usage

To run the main script with different hyperparameter values, use the `python main.py` command along with the appropriate flags.

### Hyperparameters:

#### Replay Buffer Size
To change the replay buffer size, use the `--buffer` flag. For example, to run with a replay buffer size of 5000000:

python main.py --buffer 5000000


#### Batch Size
To change the batch size, use the `--batch` flag. For example, to run with a batch size of 32:

python main.py --learning_starts 50000


#### Learning Frequency
To change the updating weights frequency, use the `--learning_freq` flag. For example, to run with an update frequency of 4:

python main.py --learning_freq 4


#### History Length
To change the length of history used, use the `--history_len` flag. For example, to run with a history length of 5000:

python main.py --history_len 5000


#### Update Target Frequency
To change the update target frequency, use the `--update_freq` flag. For example, to run with an update frequency of 4:

python main.py --update_freq 4


#### Network Build
To change the network build, use the `--func` flag. Choose between the following values:

- To run the normal DQN network: `dqn`
- To run the double DQN network: `ddqn`
- To run the dueling DQN network: `duelingdqn`

For example, to run with the double DQN network:

python main.py --func ddqn


#### Gamma Size
To change the gamma size, use the `--gamma` flag. For example, to run with a gamma size of 0.9:

python main.py --gamma 0.9


#### Alpha Size
To change the alpha size, use the `--alpha` flag. For example, to run with an alpha size of 0.9:

python main.py --alpha 0.9

#### Agent Type
To change the agent, use the `--agent` flag. Choose between the following values:

- To run the greedy agent: `greedy`
- To run the noisy agent: `noisy`
- To run the softmax agent: `softmax`

For example, to run with the softmax agent:

python main.py --agent softmax


#### Beta Value for Softmax Agent
To change the beta value for the softmax agent, use the `--beta` flag. For example, to run with a beta value of 0.002:

python main.py --beta 0.002


#### Standard Deviation Value for Noisy Agent
To change the standard deviation value for the noisy agent, use the `--std` flag. For example, to run with a standard deviation of 0.002:

python main.py --std 0.002


#### Epsilon
To change epsilon, use the `--eps` flag. For example, to run with an epsilon value of 0.002:

python main.py --eps 0.002


### Multiple Arguments
You can combine multiple arguments together. For example, to run with the double DQN network and softmax agent with beta 9:

python main.py --func ddqn --agent softmax --beta 9


Feel free to explore and customize the hyperparameters and agent configurations for your experiments. Happy Reinforcement Learning!
