import os


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)

        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None,
                        'action_dim': None, 'if_discrete': None}
        # the name of environment. Be used to set 'cwd'.
        self.env_name = env_args['env_name']
        # vector dimension (feature number) of state
        self.state_dim = env_args['state_dim']
        # vector dimension (feature number) of action
        self.action_dim = env_args['action_dim']
        # discrete or continuous action space
        self.if_discrete = env_args['if_discrete']

        self.agent_class = agent_class  # agent = agent_class(...)

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.net_dims = (64, 32)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        # num of transitions sampled from replay buffer.
        self.batch_size = int(128)
        # collect horizon_len step while exploring, then update network
        self.horizon_len = int(2000)
        # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
        self.buffer_size = None
        # repeatedly update network using ReplayBuffer to keep critic's loss small
        self.repeat_times = 8.0

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        # number of times that get episodic cumulative return
        self.eval_times = int(32)
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps

    def init_before_training(self):
        # set cwd (current working directory) for saving model
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)
