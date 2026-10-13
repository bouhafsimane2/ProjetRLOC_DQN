"""
Enregistre une vidéo MP4 de sumo-gui pendant que le DQN (ou le modèle SUMO par défaut) conduit.

SUMO avance par pas de 0.1 s pour une vidéo fluide, mais l'agent ne décide qu'une fois
par seconde (toutes les 10 sous-étapes), exactement comme pendant l'entraînement.

Usage :
    python record_video.py                         # DQN, départ fixe -> videos/dqn_demo.mp4
    python record_video.py --policy sumo           # modèle SUMO par défaut -> videos/sumo_default_demo.mp4
    python record_video.py --random-start --seed 3
Nécessite : pip install imageio imageio-ffmpeg
"""

import argparse
import os
import shutil
import tempfile

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN

import lane_change_env as lce
from lane_change_env import LaneChangeEnv, EGO_ID, KEEP, ACTION_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSTEPS = 10          # 10 x 0.1 s = 1 décision par seconde
VIEW = "View #0"
W, H = 1280, 720


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["dqn", "sumo"], default="dqn")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zoom", type=float, default=900, help="zoom sumo-gui (plus grand = plus proche)")
    parser.add_argument("--fps", type=int, default=20, help="20 fps = vidéo 2x plus rapide que le temps réel")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(HERE, "videos", f"{'dqn' if args.policy == 'dqn' else 'sumo_default'}_demo.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.policy == "dqn":
        model = DQN.load(os.path.join(HERE, "models", "dqn_lanechange.zip"), device="cpu")
        policy = lambda obs: int(model.predict(obs, deterministic=True)[0])  # noqa: E731
        label = "DQN agent"
    else:
        policy = lambda obs: KEEP  # noqa: E731
        label = "SUMO default lane-change model"

    lce.STEP_LENGTH = 1.0 / SUBSTEPS
    env = LaneChangeEnv(gui=True, sumo_default=(args.policy == "sumo"),
                        randomize_start=args.random_start, gui_delay=0,
                        max_steps=lce.MAX_STEPS * SUBSTEPS,
                        extra_sumo_args=["--window-size", f"{W},{H + 120}"])
    obs, _ = env.reset(seed=args.seed)
    c = env.conn
    c.gui.trackVehicle(VIEW, EGO_ID)
    c.gui.setZoom(VIEW, args.zoom)

    tmp = tempfile.mkdtemp(prefix="sumo_frames_")
    frames, captions = [], []
    action_name, k, done = "keep", 0, False
    f_big, f_small = font(30, bold=True), font(24)
    try:
        while not done:
            if k % SUBSTEPS == 0:
                a = policy(obs)
                action_name = ACTION_NAMES[a] if args.policy == "dqn" else "SUMO decides"
            else:
                a = KEEP
            path = os.path.join(tmp, f"f{k:05d}.png")
            c.gui.screenshot(VIEW, path, W, H)   # enregistrée à la fin du prochain pas
            frames.append(path)
            captions.append(f"t = {k / SUBSTEPS:5.1f} s   |   speed = {env.speed:4.1f} m/s   |   "
                            f"lane = {env.lane}   |   action: {action_name}")
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            k += 1
    finally:
        env.close()

    result = (f"Arrived: {info['arrived']}   |   travel time = {info['travel_time']:.1f} s   |   "
              f"lane changes = {info['lane_changes']}   |   collisions = "
              f"{info['boundary_collisions'] + info['vehicle_collisions']}")
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    last = None
    for path, cap in zip(frames, captions):
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB").resize((W, H))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 92], fill=(40, 16, 62))
        d.text((20, 10), label, font=f_big, fill=(255, 255, 255))
        d.text((20, 54), cap, font=f_small, fill=(230, 220, 245))
        writer.append_data(__import__("numpy").asarray(img))
        last = img
    if last is not None:  # image de fin avec le résultat (2 s)
        d = ImageDraw.Draw(last)
        d.rectangle([0, H - 70, W, H], fill=(40, 16, 62))
        d.text((20, H - 55), result, font=f_small, fill=(255, 255, 255))
        for _ in range(args.fps * 2):
            writer.append_data(__import__("numpy").asarray(last))
    writer.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print(result)
    print("Vidéo ->", out)


if __name__ == "__main__":
    main()
