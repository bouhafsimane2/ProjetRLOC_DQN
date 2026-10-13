"""
Environnement SUMO commun (Gymnasium) pour la décision de changement de voie.

Scénario (cf. énoncé) : autoroute à 2 voies, véhicules rouges IMMOBILES servant
d'obstacles, véhicule autonome jaune (AV) dont la vitesse est gérée par SUMO.
L'agent ne décide que du changement de voie :

    0 = garder la voie, 1 = voie de gauche, 2 = voie de droite

Convention SUMO : l'index de voie 0 est la voie la plus à DROITE,
donc "gauche" = index + 1 et "droite" = index - 1.

Observation (continue, dans [0, 1]) :
    [vitesse / v_max, gap_courant / R, gap_gauche / R, gap_droite / R]
    gap = distance libre devant l'AV dans la voie considérée (plafonnée à R).
    Une voie inexistante (bord de route) ou occupée à côté de l'AV vaut 0 :
    c'est un "mur" contre lequel un changement de voie provoquerait une collision.

Récompense par pas (1 pas = 1 s) :
    + v / v_max - 0.2           progression moins un coût de temps
                                (maximiser la vitesse = minimiser le temps de parcours)
    - 1.0                       si l'AV a effectivement changé de voie (plus qu'un pas de progression :
                                un changement n'est rentable que s'il évite un obstacle)
    - 50  + fin d'épisode       collision avec le bord de la route ou un obstacle
                                (toujours pire que rester bloqué jusqu'à la fin : 200 x 0.2 = 40)
    + 10  + fin d'épisode       arrivée au bout de la route
L'épisode est tronqué après MAX_STEPS pas (AV bloqué derrière un obstacle).
"""

import os
import itertools

import numpy as np
import gymnasium as gym
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    import sys
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402
import sumolib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUMO_CFG = os.path.join(HERE, "data", "obstacles.sumocfg")

EGO_ID = "vehAgent"
ROUTE_ID = "r_0"
EGO_TYPE = "obstacle"          # même vType que dans run.py fourni
STEP_LENGTH = 1.0              # s
MAX_STEPS = 200                # parcours libre ~ 80 pas
GAP_RANGE = 100.0              # m, portée de perception
TIME_PENALTY = 0.2
LANE_CHANGE_COST = 1.0
USELESS_GAP = GAP_RANGE        # changement de voie sans aucun obstacle visible devant (100 m libres) = inutile
COLLISION_PENALTY = 50.0
ARRIVAL_BONUS = 10.0

KEEP, LEFT, RIGHT = 0, 1, 2
ACTION_NAMES = {KEEP: "keep", LEFT: "left", RIGHT: "right"}

# Discrétisation pour le Q-learning (bornes des gaps en mètres)
GAP_BINS = [1.0, 20.0, 60.0]   # -> 4 classes : bloqué/mur | proche | moyen | libre
N_GAP_BINS = len(GAP_BINS) + 1
N_DISCRETE_STATES = N_GAP_BINS ** 3

_conn_counter = itertools.count()


def sumo_binary(gui=False):
    name = "sumo-gui" if gui else "sumo"
    try:
        return sumolib.checkBinary(name)
    except Exception:
        return name


def discretize(obs):
    """Observation continue -> index d'état discret (Q-learning).

    Chaque gap (courant, gauche, droite) est classé dans 4 intervalles :
      0 : <= 1 m   (bord de route ou véhicule à côté : changement impossible)
      1 : 1-20 m   (obstacle proche, l'AV freine)
      2 : 20-60 m  (obstacle en vue)
      3 : > 60 m   (voie libre)
    => 4^3 = 64 états.
    """
    gaps_m = np.asarray(obs[1:4], dtype=np.float64) * GAP_RANGE
    bins = np.digitize(gaps_m, GAP_BINS)
    return int(bins[0] * N_GAP_BINS ** 2 + bins[1] * N_GAP_BINS + bins[2])


def greedy_rollouts(env, policy, n_episodes, seed0):
    """Joue n épisodes avec une politique déterministe ; renvoie (récompense totale moyenne, taux d'arrivée)."""
    rewards, arrived = [], []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + k)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        rewards.append(info["total_reward"])
        arrived.append(info["arrived"])
    return float(np.mean(rewards)), float(np.mean(arrived))


class LaneChangeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui=False, sumo_default=False, randomize_start=True,
                 max_steps=MAX_STEPS, gui_delay=100, extra_sumo_args=()):
        """
        gui            : lance sumo-gui au lieu de sumo
        sumo_default   : l'AV utilise le modèle de changement de voie par défaut de SUMO
                         (les actions de l'agent sont ignorées) -> baseline
        randomize_start: position / voie / vitesse de départ aléatoires (sinon départ fixe)
        """
        super().__init__()
        self.gui = gui
        self.sumo_default = sumo_default
        self.randomize_start = randomize_start
        self.max_steps = max_steps
        self.gui_delay = gui_delay
        self.extra_sumo_args = list(extra_sumo_args)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)

        self.label = f"lcenv_{os.getpid()}_{next(_conn_counter)}"
        self.conn = None
        self.obstacles = {}  # lane index -> liste triée de (back, front)
        self.v_max = 13.89
        self.ego_len = 5.0
        self.n_lanes = 2

    # ------------------------------------------------------------------ SUMO
    def _start_sumo(self):
        """Lance SUMO au premier reset, puis recharge simplement la simulation
        (traci.load) : beaucoup plus rapide que de relancer un processus à chaque épisode."""
        args = [
            "-c", SUMO_CFG,
            "--step-length", str(STEP_LENGTH),
            "--collision.action", "warn",
            "--collision.check-junctions", "true",
            "--time-to-teleport", "-1",
            "--xml-validation", "never",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--seed", str(int(self.np_random.integers(0, 2**31 - 1))),
        ]
        if self.gui:
            args += ["--start", "--quit-on-end", "--delay", str(self.gui_delay)]
        args += self.extra_sumo_args
        if self.conn is None:
            traci.start([sumo_binary(self.gui)] + args, label=self.label)
            self.conn = traci.getConnection(self.label)
        else:
            self.conn.load(args)

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    # ----------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._start_sumo()
        c = self.conn

        # 1) insérer les obstacles puis les immobiliser (comme dans run.py)
        c.simulationStep()
        for v in c.vehicle.getIDList():
            c.vehicle.setLaneChangeMode(v, 0)
            c.vehicle.setSpeed(v, 0)

        # 2) insérer l'AV
        if self.randomize_start:
            depart_pos = float(self.np_random.uniform(0.0, 250.0))
            depart_lane = int(self.np_random.integers(0, 2))
            depart_speed = float(self.np_random.uniform(0.0, 13.0))
        else:
            depart_pos, depart_lane, depart_speed = 0.0, 0, 13.0
        c.vehicle.add(EGO_ID, ROUTE_ID, typeID=EGO_TYPE, depart="now",
                      departPos=f"{depart_pos:.2f}", departLane=str(depart_lane),
                      departSpeed=f"{depart_speed:.2f}")
        c.vehicle.setColor(EGO_ID, (255, 255, 0, 255))
        if not self.sumo_default:
            c.vehicle.setLaneChangeMode(EGO_ID, 0)  # désactive le modèle LC de SUMO
        for _ in range(20):
            c.simulationStep()
            if EGO_ID in c.vehicle.getIDList():
                break
        else:
            raise RuntimeError("L'AV n'a pas pu être inséré dans SUMO")

        # 3) les obstacles sont immobiles : on mémorise leurs positions une fois pour toutes
        self.obstacles = {}
        for v in c.vehicle.getIDList():
            if v == EGO_ID:
                continue
            front = c.vehicle.getLanePosition(v)
            length = c.vehicle.getLength(v)
            self.obstacles.setdefault(c.vehicle.getLaneIndex(v), []).append((front - length, front))
        for lst in self.obstacles.values():
            lst.sort()

        edge = c.vehicle.getRoadID(EGO_ID)
        self.n_lanes = c.edge.getLaneNumber(edge)
        self.v_max = c.lane.getMaxSpeed(c.vehicle.getLaneID(EGO_ID))
        self.ego_len = c.vehicle.getLength(EGO_ID)
        self.edge_len = c.lane.getLength(c.vehicle.getLaneID(EGO_ID))

        self.steps = 0
        self.total_reward = 0.0
        self.lane_changes = 0
        self.useless_lane_changes = 0
        self.boundary_collisions = 0
        self.vehicle_collisions = 0
        self.speed_sum = 0.0
        self.start_pos = c.vehicle.getLanePosition(EGO_ID)
        self.lane = c.vehicle.getLaneIndex(EGO_ID)
        self.pos = self.start_pos
        self.speed = c.vehicle.getSpeed(EGO_ID)
        return self._obs(), self._info(arrived=False)

    # ----------------------------------------------------------- observation
    def _gap(self, lane):
        """Distance libre devant l'AV dans `lane` (0 si voie inexistante ou occupée à côté)."""
        if lane < 0 or lane >= self.n_lanes:
            return 0.0
        ego_front, ego_back = self.pos, self.pos - self.ego_len
        gap = GAP_RANGE
        for back, front in self.obstacles.get(lane, []):
            if front <= ego_back:
                continue                     # obstacle derrière l'AV
            gap = min(gap, max(0.0, back - ego_front))  # 0 si chevauchement latéral
            break                            # liste triée : premier obstacle devant
        return gap

    def _obs(self):
        return np.array([
            min(self.speed / self.v_max, 1.0),
            self._gap(self.lane) / GAP_RANGE,
            self._gap(self.lane + 1) / GAP_RANGE,   # gauche
            self._gap(self.lane - 1) / GAP_RANGE,   # droite
        ], dtype=np.float32)

    def _info(self, arrived):
        return {
            "steps": self.steps,
            "travel_time": self.steps * STEP_LENGTH,
            "arrived": arrived,
            "lane_changes": self.lane_changes,
            "useless_lane_changes": self.useless_lane_changes,
            "boundary_collisions": self.boundary_collisions,
            "vehicle_collisions": self.vehicle_collisions,
            "distance": self.pos - self.start_pos,
            "mean_speed": self.speed_sum / max(self.steps, 1),
            "total_reward": self.total_reward,
        }

    # ------------------------------------------------------------------ step
    def step(self, action):
        c = self.conn
        action = int(action)
        reward = 0.0
        terminated = False
        arrived = False
        prev_lane = self.lane
        gap_before = self._gap(self.lane)

        # --- action de l'agent (ignorée si le modèle SUMO par défaut conduit) ---
        if not self.sumo_default and action != KEEP:
            target = self.lane + 1 if action == LEFT else self.lane - 1
            if 0 <= target < self.n_lanes:
                c.vehicle.changeLane(EGO_ID, target, STEP_LENGTH)
            else:
                # collision avec le bord de la route
                self.boundary_collisions += 1
                self.steps += 1
                reward = -COLLISION_PENALTY
                self.total_reward += reward
                return self._obs(), reward, True, False, self._info(arrived=False)

        c.simulationStep()
        self.steps += 1

        if EGO_ID not in c.vehicle.getIDList():
            # l'AV a quitté le réseau : fin de route atteinte
            arrived = True
            terminated = True
            last_step = max(self.edge_len - self.pos, 0.0)
            self.speed_sum += min(last_step / STEP_LENGTH, self.v_max)
            self.pos = self.edge_len
            reward = min(last_step / (self.v_max * STEP_LENGTH), 1.0) - TIME_PENALTY + ARRIVAL_BONUS
        else:
            self.lane = c.vehicle.getLaneIndex(EGO_ID)
            self.pos = c.vehicle.getLanePosition(EGO_ID)
            self.speed = c.vehicle.getSpeed(EGO_ID)
            self.speed_sum += self.speed
            reward = min(self.speed / self.v_max, 1.0) - TIME_PENALTY
            if self.lane != prev_lane:
                self.lane_changes += 1
                if gap_before >= USELESS_GAP:
                    self.useless_lane_changes += 1
                reward -= LANE_CHANGE_COST
            if EGO_ID in c.simulation.getCollidingVehiclesIDList():
                self.vehicle_collisions += 1
                reward -= COLLISION_PENALTY
                terminated = True

        truncated = (not terminated) and self.steps >= self.max_steps
        self.total_reward += reward
        return self._obs(), reward, terminated, truncated, self._info(arrived)
