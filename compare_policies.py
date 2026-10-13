"""
Comparaison Q-learning vs DQN vs modèle de changement de voie par défaut de SUMO.

Les trois politiques sont évaluées dans le MÊME environnement (lane_change_env.py) et sur les
MÊMES 50 départs aléatoires (seeds jamais vues à l'entraînement), en mode greedy.
La Q-table d'origine (Qlearning.py) est utilisée avec son propre état et ses propres actions :
    état 1 si un obstacle est à moins de 30 m devant, sinon 0
    action 1 = passer sur l'autre voie

Sorties :
    results/evaluation_all_policies.csv, results/evaluation_all_policies_summary.csv
    plots/compare_travel_time.png, plots/compare_travel_time_episodes.png, plots/compare_speed.png

Usage : python compare_policies.py [--episodes 50]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from lane_change_env import LaneChangeEnv, KEEP, LEFT, RIGHT, GAP_RANGE, MAX_STEPS

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["Q-learning", "DQN", "SUMO_default"]
COLORS = {"Q-learning": "#c0392b", "DQN": "#5b2a86", "SUMO_default": "#27ae60"}


def find(*candidates):
    for c in candidates:
        path = os.path.join(HERE, c)
        if os.path.exists(path):
            return path
    raise SystemExit(f"Fichier introuvable : {candidates}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    for d in ("results", "plots"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)

    q = np.load(find("Qlearning_q_table.npy", "models/Qlearning_q_table.npy"))
    dqn = DQN.load(find("models/dqn_lanechange.zip"), device="cpu")

    def qlearning(env, obs):
        s = 1 if obs[1] * GAP_RANGE < 30 else 0
        if int(np.argmax(q[s])) == 0:
            return KEEP
        return LEFT if env.lane == 0 else RIGHT

    policies = {
        "Q-learning": (qlearning, False),
        "DQN": (lambda env, obs: int(dqn.predict(obs, deterministic=True)[0]), False),
        "SUMO_default": (lambda env, obs: KEEP, True),  # actions ignorées : SUMO décide
    }

    rows = []
    for name, (policy, sumo_default) in policies.items():
        env = LaneChangeEnv(sumo_default=sumo_default)
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=1_000_000 + ep)
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(policy(env, obs))
                done = terminated or truncated
            rows.append({"policy": name, "episode": ep + 1,
                         "travel_time": info["travel_time"] if info["arrived"] else float(MAX_STEPS),
                         "lane_changes": info["lane_changes"],
                         "useless_lane_changes": info["useless_lane_changes"],
                         "arrived": int(info["arrived"]),
                         "collisions": info["boundary_collisions"] + info["vehicle_collisions"],
                         "mean_speed": round(info["mean_speed"], 3),
                         "total_reward": round(info["total_reward"], 3)})
        env.close()
        print(f"{name} : {args.episodes} épisodes joués")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "evaluation_all_policies.csv"), index=False)
    summary = (df.groupby("policy")
                 .agg(travel_time=("travel_time", "mean"), travel_time_std=("travel_time", "std"),
                      lane_changes=("lane_changes", "mean"),
                      useless_lane_changes=("useless_lane_changes", "mean"),
                      arrival_rate=("arrived", "mean"), collisions=("collisions", "mean"),
                      mean_speed=("mean_speed", "mean"), total_reward=("total_reward", "mean"))
                 .reindex(ORDER).round(2))
    summary.to_csv(os.path.join(HERE, "results", "evaluation_all_policies_summary.csv"))
    print(summary.T.to_string())

    labels = ["Q-learning", "DQN", "SUMO default"]
    for metric, ylabel, fname in [("travel_time", "Travel time (s)", "compare_travel_time.png"),
                                  ("mean_speed", "Mean speed (m/s)", "compare_speed.png")]:
        fig, ax = plt.subplots(figsize=(7, 5.2))
        data = [df[df.policy == p][metric] for p in ORDER]
        bars = ax.bar(labels, [d.mean() for d in data], yerr=[d.std() for d in data], capsize=8,
                      color=[COLORS[p] for p in ORDER], width=0.6)
        for b, d in zip(bars, data):
            ax.text(b.get_x() + b.get_width() / 2, d.mean() / 2, f"{d.mean():.1f}",
                    ha="center", color="white", fontsize=15, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} - mean ± std, {args.episodes} episodes")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "plots", fname), dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for p, label in zip(ORDER, labels):
        sub = df[df.policy == p]
        ax.plot(sub.episode, sub.travel_time, "o-", ms=3, lw=1.2, color=COLORS[p], label=label)
    ax.set_xlabel("Evaluation episode (same random start for all policies)")
    ax.set_ylabel("Travel time (s)")
    ax.set_title("Travel time per evaluation episode")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", "compare_travel_time_episodes.png"), dpi=160)
    print("CSV -> results/, figures -> plots/")


if __name__ == "__main__":
    main()
