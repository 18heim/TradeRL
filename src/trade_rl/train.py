from trade_rl.meta.data_processor import DataProcessor
from trade_rl.agents.drl_agent import DRLAgent
import numpy as np


def train(drl_lib, env_class, model_name, **kwargs):
    # read parameters and load agents
    cwd = kwargs.get("cwd", "./" + str(model_name))  # cwd: current_working_dir
    DP = DataProcessor(**kwargs)
    price_array, tech_array, turbulence_array = DP.run(kwargs["ticker_list"],
                                                        kwargs["technical_indicator_list"], 
                                                        kwargs["if_vix"], cache=True)

    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}

    #build environment using processed data
    env_instance = env_class(config=data_config)

    if drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")

        agent = DRLAgent(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training finished!")
        trained_model.save(cwd)
        print("Trained model saved in " + str(cwd))

    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
