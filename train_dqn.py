"""
Deep RL : DQN (stable-baselines3) sur le même environnement que le Q-learning.

- Observation continue normalisée (4 valeurs dans [0, 1]), 3 actions
- Réseau : MLP 4 -> 64 -> 64 -> 3 (ReLU)
- Logs TensorBoard : runs/dqn
    * episode/total_reward : récompense totale NON actualisée de chaque épisode (x = n° d'épisode)
    * eval/total_reward    : moyenne sur 5 épisodes joués en greedy, tous les 2500 pas
    * rollout/ep_rew_mean, train/loss, ... : logs standards de SB3

Usage : python train_dqn.py [--timesteps 150000]
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from lane_change_env import LaneChangeEnv, greedy_rollouts

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_EVERY_STEPS, EVAL_EPISODES, EVAL_SEED = 2_500, 5, 500_000


class EpisodeLogger(BaseCallback):
    """Écrit la récompense totale de chaque épisode dans TensorBoard et évalue
    périodiquement la politique greedy sur un environnement séparé."""

    def __init__(self, log_dir, eval_env):
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        self.eval_env = eval_env
        self.history = []
        self.eval_history = []
        self.last_eval = 0

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            ep = len(self.history)
            row = {"episode": ep, "timestep": self.num_timesteps,
                   **{k: v for k, v in info.items()
                      if k in ("total_reward", "travel_time", "arrived", "lane_changes", "useless_lane_changes",
                               "boundary_collisions", "vehicle_collisions")}}
            self.history.append(row)
            self.writer.add_scalar("episode/total_reward", row["total_reward"], ep)
            self.writer.add_scalar("episode/travel_time", row["travel_time"], ep)
            self.writer.add_scalar("episode/lane_changes", row["lane_changes"], ep)
            self.writer.add_scalar("episode/useless_lane_changes", row["useless_lane_changes"], ep)
            self.writer.add_scalar("episode/arrived", float(row["arrived"]), ep)
            if self.num_timesteps - self.last_eval >= EVAL_EVERY_STEPS:
                self.last_eval = self.num_timesteps
                policy = lambda o: int(self.model.predict(o, deterministic=True)[0])  # noqa: E731
                mean_r, arr = greedy_rollouts(self.eval_env, policy, EVAL_EPISODES, EVAL_SEED)
                self.writer.add_scalar("eval/total_reward", mean_r, ep)
                self.writer.add_scalar("eval/arrival_rate", arr, ep)
                self.eval_history.append({"episode": ep, "timestep": self.num_timesteps,
                                          "eval_total_reward": mean_r, "eval_arrival_rate": arr})
        return True

    def _on_training_end(self):
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for d in ("models", "plots", "results", "runs"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)

    env = Monitor(LaneChangeEnv())
    env.reset(seed=args.seed)

    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.005,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=os.path.join(HERE, "runs"),
        seed=args.seed,
        verbose=1,
    )

    eval_env = LaneChangeEnv()
    callback = EpisodeLogger(os.path.join(HERE, "runs", "dqn_episodes"), eval_env)
    model.learn(total_timesteps=args.timesteps, callback=callback,
                tb_log_name="dqn", log_interval=50)
    model.save(os.path.join(HERE, "models", "dqn_lanechange"))
    env.close()
    eval_env.close()

    df = pd.DataFrame(callback.history)
    df.to_csv(os.path.join(HERE, "results", "dqn_training.csv"), index=False)
    ev = pd.DataFrame(callback.eval_history)
    ev.to_csv(os.path.join(HERE, "results", "dqn_eval_during_training.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df.episode, df.total_reward, alpha=0.3, label="récompense totale / épisode")
    plt.plot(df.episode, df.total_reward.rolling(50, min_periods=1).mean(),
             label="moyenne glissante (50)")
    plt.plot(ev.episode, ev.eval_total_reward, "k.-", label="politique greedy (moy. 5 épisodes)")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale (non actualisée)")
    plt.title("DQN - convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "plots", "dqn_training_reward.png"), dpi=120)
    print("Modèle -> models/dqn_lanechange.zip | courbe -> plots/dqn_training_reward.png")


if __name__ == "__main__":
    main()
