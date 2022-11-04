from pathlib import Path
from typing import Any


from trade_rl.agents.drl_agent import DRLAgent
from trade_rl.meta.data_processor import DataProcessor

from omegaconf import DictConfig


def train(drl_lib: str,
          processor_config: DictConfig,
          data_config: DictConfig,
          cwd: Path,
          env_class: Any,
          model_params: DictConfig,
          initial_capital: float,
          max_trade: float):
    """Training RL Model."""
    # read parameters and load agents
    cwd = cwd / model_params["model_name"]
    DP = DataProcessor(**processor_config)
    data_config['price_array'], data_config['tech_array'], data_config['turbulence_array'] = DP.run(data_config["ticker_list"],
                                                                                                    data_config["technical_indicator_list"],
                                                                                                    data_config["if_vix"], cache=True)

    # build environment using processed data
    env_instance = env_class(data_config=data_config,
                             initial_capital=initial_capital,
                             max_trade=max_trade)

    if drl_lib == "stable_baselines3":
        total_timesteps = model_params.get("total_timesteps", 1e6)
        agent_params = model_params.get("agent_params")

        agent = DRLAgent(env=env_instance)
        model = agent.get_model(model_params["model_name"],
                                model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_params["model_name"], total_timesteps=total_timesteps
        )
        print("Training finished!")
        trained_model.save(cwd)
        print("Trained model saved in " + str(cwd))

    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
