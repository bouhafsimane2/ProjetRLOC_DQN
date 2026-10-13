"""
Test du modèle DQN téléchargé (sans interface graphique).

Compare le DQN au modèle de changement de voie par défaut de SUMO sur les mêmes
départs aléatoires, affiche un tableau et sauvegarde un CSV dans results/.

Usage :
    python test_model.py
    python test_model.py --episodes 20 --model models/dqn_lanechange.zip
"""

import argparse
import os

import pandas as pd
from stable_baselines3 import DQN

from lane_change_env import LaneChangeEnv, KEEP, MAX_STEPS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(HERE, "models", "dqn_lanechange.zip"))
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Modèle introuvable : {args.model}")
    model = DQN.load(args.model, device="cpu")
    print(f"Modèle chargé : {args.model}")

    policies = {
        "DQN": (lambda o: int(model.predict(o, deterministic=True)[0]), False),
        "SUMO_default": (lambda o: KEEP, True),  # actions ignorées : SUMO décide
    }
    rows = []
    for name, (policy, sumo_default) in policies.items():
        env = LaneChangeEnv(sumo_default=sumo_default)
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=1_000_000 + ep)
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(policy(obs))
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
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "test_dqn_vs_sumo.csv"), index=False)

    summary = df.groupby("policy").agg(
        temps_parcours_s=("travel_time", "mean"),
        changements_voie=("lane_changes", "mean"),
        changements_inutiles=("useless_lane_changes", "mean"),
        taux_arrivee=("arrived", "mean"),
        collisions=("collisions", "mean"),
        vitesse_moy_ms=("mean_speed", "mean"),
        recompense_totale=("total_reward", "mean")).round(2)
    print()
    print(summary.T.to_string())
    print()

    dqn, sumo = summary.loc["DQN"], summary.loc["SUMO_default"]
    ok = dqn.taux_arrivee == 1.0 and dqn.collisions == 0
    print("RÉSULTAT :", "✅ le modèle fonctionne" if ok else "❌ le modèle a des échecs",
          f"(arrivées {dqn.taux_arrivee:.0%}, collisions {dqn.collisions}, "
          f"changements inutiles {dqn.changements_inutiles}, "
          f"temps {dqn.temps_parcours_s:.1f} s contre {sumo.temps_parcours_s:.1f} s pour SUMO)")
    print("Détails -> results/test_dqn_vs_sumo.csv")


if __name__ == "__main__":
    main()
