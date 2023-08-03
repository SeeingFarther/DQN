import gym
import torch.optim as optim
import argparse
from dqn_model import DQN, DuelingDQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

FUNC = DQN
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
AGENT = 'greedy'
STD = 0.1
BETA = 30
DDQN_FLAG = False
DuelingDQN_FLAG = False

def main(env, num_timesteps, args):
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=args.learning_rate, alpha=args.alpha, eps=args.eps),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=FUNC,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=args.buffer,
        batch_size=args.batch,
        gamma=args.gamma,
        learning_starts=args.learning_starts,
        learning_freq=args.learning_freq,
        frame_history_len=args.history_len,
        target_update_freq=args.update_freq,
        agent=args.agent,
        std=args.std,
        beta=args.beta,
        ddqn_flag=DDQN_FLAG,
        dueling_dqn_flag=DuelingDQN_FLAG
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN arguments')
    parser.add_argument('--buffer', type=int, default=REPLAY_BUFFER_SIZE, help='Size of the replay buffer')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Size of the batch')
    parser.add_argument('--learning_starts', type=int, default=LEARNING_STARTS, help='Timestep when learning starts')
    parser.add_argument('--learning_freq', type=int, default=LEARNING_FREQ, help='Updating weights frequency')
    parser.add_argument('--history_len', type=int, default=FRAME_HISTORY_LEN, help='Length of history used')
    parser.add_argument('--update_freq', type=int, default=TARGER_UPDATE_FREQ, help='Update target frequency')
    parser.add_argument('--func', type=str, default='dqn', help='Kind of network: dqn, ddqn or duelingdqn')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='Value of the gamma')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='Value of the alpha')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Value of learning rate')
    parser.add_argument('--agent', type=str, default='greedy', help='Kind of agent: greedy, noisy or softmax')
    parser.add_argument('--beta', type=float, default=BETA, help='Beta value used with softmax')
    parser.add_argument('--std', type=float, default=STD, help='Standard deviation for the noisy agent')
    parser.add_argument('--eps', type=float, default=EPS, help='epsilon')
    args = parser.parse_args()

    # Check for func
    if args.func == 'dqn':
        pass
    elif args.func == 'ddqn':
        DDQN_FLAG = True
    elif args.func == 'duelingdqn':
        FUNC = DuelingDQN
        DuelingDQN_FLAG = True
    else:
        raise ValueError('Invalid func')

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, task.max_timesteps, args)
