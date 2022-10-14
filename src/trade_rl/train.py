from trade_rl.meta.data_processor import DataProcessor
from trade_rl.agents.drl_agent import DRLAgent
import numpy as np


def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):

    # process data using unified data processor
    DP = DataProcessor(data_source, start_date,
                       end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                       technical_indicator_list,
                                                       if_vix, cache=True)

    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}

    # build environment using processed data
    env_instance = env(config=data_config)

    # read parameters and load agents
    current_working_dir = kwargs.get(
        'current_working_dir', './'+str(model_name))

    if drl_lib == 'elegantrl':
        break_step = kwargs.get('break_step', 1e6)
        erl_params = kwargs.get('erl_params')

        agent = DRLAgent(env=env,
                         price_array=price_array,
                         tech_array=tech_array,
                         turbulence_array=turbulence_array)

        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(model=model,
                                          cwd=current_working_dir,
                                          total_timesteps=break_step)
