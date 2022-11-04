# import DRL agents

from trade_rl.meta.data_processor import DataProcessor
from trade_rl.agents.drl_agent import DRLAgent

from omegaconf import DictConfig
from pathlib import Path
from typing import Any


def test(drl_lib: str,
         processor_config: DictConfig,
         data_config: DictConfig,
         cwd: Path,
         env_class: Any,
         model_name: str,
         initial_capital: float):
    """Testing RL Model."""
    # read parameters and load agents
    cwd = cwd / model_name
    DP = DataProcessor(**processor_config)
    data_config['price_array'], \
        data_config['tech_array'], data_config['turbulence_array'] = \
        DP.run(data_config["ticker_list"],
               data_config["technical_indicator_list"],
               data_config["if_vix"], cache=True)

    # build environment using processed data
    data_config["if_train"] = False
    env_instance = env_class(data_config=data_config,
                             initial_capital=initial_capital)

    # run prediction episode.
    if drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent.DRL_prediction_load_from_file(
            model_name=model_name,
            cwd=cwd,
            environment=env_instance,
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
