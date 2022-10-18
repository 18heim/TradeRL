# import DRL agents

from trade_rl.meta.data_processor import DataProcessor
from trade_rl.agents.drl_agent import DRLAgent
import numpy as np


def test(start_date, end_date, ticker_list, data_source, time_interval,
            technical_indicator_list, drl_lib, env_class, model_name, if_vix=True,
            **kwargs):
  
    #process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                        technical_indicator_list, 
                                                        if_vix, cache=True)
    

    #build environment using processed data
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env_class(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    current_working_dir = kwargs.get("current_working_dir", "./" + str(model_name))

    if drl_lib == "stable_baseline3":
        episode_total_assets = DRLAgent.DRL_prediction_load_from_file(
            model_name=model_name,
            cwd=current_working_dir,
            environment=env_instance,
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
