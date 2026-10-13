"""
Visualisation du DQN (ou du modèle SUMO par défaut) dans sumo-gui.

Usage :
    python simulate_gui.py --policy dqn
    python simulate_gui.py --policy sumo          # modèle par défaut de SUMO
    python simulate_gui.py --policy dqn --random-start --seed 3
"""

import argparse
import os

from stable_baselines3 import DQN

from lane_change_env import LaneChangeEnv, ACTION_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["dqn", "sumo"], default="dqn")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=int, default=200, help="délai sumo-gui (ms)")
    args = parser.parse_args()

    if args.policy == "dqn":
        model = DQN.load(os.path.join(HERE, "models", "dqn_lanechange.zip"), device="cpu")
        policy = lambda obs: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
    else:
        policy = lambda obs: 0  # noqa: E731

    env = LaneChangeEnv(gui=True, sumo_default=(args.policy == "sumo"),
                        randomize_start=args.random_start, gui_delay=args.delay)
    obs, _ = env.reset(seed=args.seed)
    done = False
    while not done:
        a = policy(obs)
        obs, r, terminated, truncated, info = env.step(a)
        print(f"t={info['steps']:3d} | action={ACTION_NAMES[a]:5s} | reward={r:6.2f} | "
              f"voie={env.lane} | pos={env.pos:7.1f} m")
        done = terminated or truncated
    env.close()
    print(f"Fin : arrivé={info['arrived']} | temps={info['travel_time']:.0f} s | "
          f"changements de voie={info['lane_changes']} | "
          f"collisions={info['boundary_collisions'] + info['vehicle_collisions']} | "
          f"récompense totale={info['total_reward']:.2f}")


if __name__ == "__main__":
    main()
