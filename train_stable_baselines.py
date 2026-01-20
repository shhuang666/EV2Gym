# this file is used to evalaute the performance of the ev2gym environment with various stable baselines algorithms.

from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
# from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml
import importlib
import sys

def load_function_from_module(function_spec):
    """
    Load a function from a module specification.
    
    Args:
        function_spec: Either a simple function name (for built-in functions)
                      or 'module_path:function_name' for custom functions.
                      Examples:
                        - 'profit_maximization' (built-in)
                        - 'my_state:custom_state_function' (custom)
                        - 'path.to.module:my_function' (custom with nested path)
    
    Returns:
        The function object
    """
    if ':' in function_spec:
        # Custom function: module_path:function_name
        module_path, function_name = function_spec.split(':', 1)
        
        # Add current directory to sys.path if not already there
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the function from the module
            if hasattr(module, function_name):
                return getattr(module, function_name)
            else:
                raise AttributeError(f"Module '{module_path}' has no function '{function_name}'")
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_path}': {e}")
    else:
        # Return the spec as-is, will be looked up in the built-in dictionaries
        return function_spec

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ppo")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--train_steps', type=int, default=20_000) 
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--config_file', type=str, default="ev2gym/example_config_files/V2GProfitPlusLoads.yaml")
    parser.add_argument('--reward_function', type=str, default=None)
    parser.add_argument('--state_function', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()
    algorithm = args.algorithm
    device = args.device
    run_name = args.run_name
    config_file = args.config_file
    reward_function_arg = args.reward_function
    state_function_arg = args.state_function
    use_wandb = not args.no_wandb

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    # Determine reward and state functions based on config file (defaults)
    if config_file == "ev2gym/example_config_files/V2GProfitMax.yaml":
        reward_function = profit_maximization
        state_function = V2G_profit_max
        group_name = f'{config["number_of_charging_stations"]}cs_V2GProfitMax'
    elif config_file == "ev2gym/example_config_files/PublicPST.yaml":
        reward_function = SquaredTrackingErrorReward
        state_function = PublicPST
        group_name = f'{config["number_of_charging_stations"]}cs_PublicPST'
    elif config_file == "ev2gym/example_config_files/V2GProfitPlusLoads.yaml":
        reward_function = ProfitMax_TrPenalty_UserIncentives
        state_function = V2G_profit_max_loads
        group_name = f'{config["number_of_charging_stations"]}cs_V2GProfitPlusLoads'
    else:
        # Default for custom config files
        reward_function = profit_maximization
        state_function = V2G_profit_max
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        group_name = f'{config["number_of_charging_stations"]}cs_{config_name}'
    
    # Override with command-line arguments if provided
    if reward_function_arg is not None:
        reward_function = load_function_from_module(reward_function_arg)
    
    if state_function_arg is not None:
        state_function = load_function_from_module(state_function_arg)
                
    run_name += f'{algorithm}_{reward_function.__name__}_{state_function.__name__}'

    if use_wandb:
        run = wandb.init(project='ev2gym',
                         sync_tensorboard=True,
                         group=group_name,
                         name=run_name,
                         save_code=True,
                         )

    gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'verbose': False,
                              'save_plots': False,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              })

    env = gym.make('evs-v0')

    eval_log_dir = "./eval_logs/" + group_name + "_" + run_name + "/"
    save_path = f"./saved_models/{group_name}/{run_name}/"
    
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(f"./saved_models/{group_name}", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    print(f'Model will be saved at: {save_path}')

    eval_callback = EvalCallback(env,
                                 best_model_save_path=save_path,
                                 log_path=eval_log_dir,
                                 eval_freq=config['simulation_length']*30,
                                 n_eval_episodes=50,
                                 deterministic=True)

    if algorithm == "ddpg":
        model = DDPG("MlpPolicy", env, verbose=1,
                    learning_rate = 1e-3,
                    buffer_size = 1_000_000,  # 1e6
                    learning_starts = 100,
                    batch_size = 100,
                    tau = 0.005,
                    gamma = 0.99,                     
                     device=device, tensorboard_log="./logs/")
    elif algorithm == "td3":
        model = TD3("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "sac":
        model = SAC("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "a2c":
        model = A2C("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "ppo":
        model = PPO("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    # elif algorithm == "tqc":
    #     model = TQC("MlpPolicy", env, verbose=1,
    #                 device=device, tensorboard_log="./logs/")
    # elif algorithm == "trpo":
    #     model = TRPO("MlpPolicy", env, verbose=1,
    #                  device=device, tensorboard_log="./logs/")
    # elif algorithm == "ars":
    #     model = ARS("MlpPolicy", env, verbose=1,
    #                 device=device, tensorboard_log="./logs/")
    # elif algorithm == "rppo":
    #     model = RecurrentPPO("MlpLstmPolicy", env, verbose=1,
    #                          device=device, tensorboard_log="./logs/")
    else:
        raise ValueError("Unknown algorithm")

    callbacks = [eval_callback]
    if use_wandb:
        callbacks.insert(0, WandbCallback(verbose=2))
    
    model.learn(total_timesteps=args.train_steps,
                progress_bar=True,
                callback=callbacks)
    # model.save(f"./saved_models/{group_name}/{run_name}.last")
    print(f'Finished training {algorithm} algorithm, {run_name} saving model at {save_path}_last.pt')

    model.save(f"{save_path}/last_model.zip")    
    
    #load the best model
    model = model.load(f"{save_path}/best_model.zip", env=env)

    env = model.get_env()
    obs = env.reset()

    stats = []
    for i in range(96*100):

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # env.render()
        # VecEnv resets automatically
        if done:
            stats.append(info)
            obs = env.reset()

    # print average stats
    print("=====================================================")
    print(f' Average stats for {algorithm} algorithm, {len(stats)} episodes')
    print("total_ev_served: ", sum(
        [i[0]['total_ev_served'] for i in stats])/len(stats))
    print("total_profits: ", sum(
        [i[0]['total_profits'] for i in stats])/len(stats))
    print("total_energy_charged: ", sum(
        [i[0]['total_energy_charged'] for i in stats])/len(stats))
    print("total_energy_discharged: ", sum(
        [i[0]['total_energy_discharged'] for i in stats])/len(stats))
    print("average_user_satisfaction: ", sum(
        [i[0]['average_user_satisfaction'] for i in stats])/len(stats))
    print("power_tracker_violation: ", sum(
        [i[0]['power_tracker_violation'] for i in stats])/len(stats))
    print("tracking_error: ", sum(
        [i[0]['tracking_error'] for i in stats])/len(stats))
    print("energy_user_satisfaction: ", sum(
        [i[0]['energy_user_satisfaction'] for i in stats])/len(stats))
    print("total_transformer_overload: ", sum(
        [i[0]['total_transformer_overload'] for i in stats])/len(stats))
    print("reward: ", sum([i[0]['episode']['r'] for i in stats])/len(stats))

    if use_wandb:
        run.log({
            "test/total_ev_served": sum([i[0]['total_ev_served'] for i in stats])/len(stats),
            "test/total_profits": sum([i[0]['total_profits'] for i in stats])/len(stats),
            "test/total_energy_charged": sum([i[0]['total_energy_charged'] for i in stats])/len(stats),
            "test/total_energy_discharged": sum([i[0]['total_energy_discharged'] for i in stats])/len(stats),
            "test/average_user_satisfaction": sum([i[0]['average_user_satisfaction'] for i in stats])/len
            (stats),
            "test/power_tracker_violation": sum([i[0]['power_tracker_violation'] for i in stats])/len(stats),
            "test/tracking_error": sum([i[0]['tracking_error'] for i in stats])/len(stats),
            "test/energy_user_satisfaction": sum([i[0]['energy_user_satisfaction'] for i in stats])/len
            (stats),
            "test/total_transformer_overload": sum([i[0]['total_transformer_overload'] for i in stats])/len
            (stats),
            "test/reward": sum([i[0]['episode']['r'] for i in stats])/len(stats),
        })

        run.finish()
