import json
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from env_3d import UAVRelay3DEnv


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _nearest_action(env, desired_velocity):
    best_idx = 0
    best_score = float("inf")
    for idx, action in enumerate(env.actions):
        move = action.copy()
        xy_norm = np.linalg.norm(move[:2])
        if xy_norm > 0:
            move[:2] = move[:2] / xy_norm
        candidate = np.array(
            [move[0] * env.max_speed_xy, move[1] * env.max_speed_xy, move[2] * env.max_speed_h],
            dtype=np.float32,
        )
        score = float(np.linalg.norm(candidate - desired_velocity))
        if score < best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _adaptive_action(env, metrics):
    relay_xy = env.service_target_xy(metrics)
    desired_alt = 42.0 + 45.0 * min(1.0, np.linalg.norm(relay_xy - env.base_pos[:2]) / 320.0)
    desired = np.array([relay_xy[0], relay_xy[1], desired_alt], dtype=np.float32)
    desired_velocity = np.clip(
        (desired - env.uav_pos) * 0.16,
        [-env.max_speed_xy, -env.max_speed_xy, -env.max_speed_h],
        [env.max_speed_xy, env.max_speed_xy, env.max_speed_h],
    )
    return _nearest_action(env, desired_velocity)


def _load_trained_agent(env, model_path):
    if not model_path.exists() and model_path.name == "ddqn_uav_3d_final.pt":
        model_path = model_path.with_name("ddqn_uav_3d.pt")
    if not model_path.exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DDQNAgent(obs_dim=env.state_dim, n_actions=env.n_actions, device=device)
    agent.policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.policy.eval()
    return agent


def _policy_action(agent, state):
    s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        return int(agent.policy(s).argmax().item())


def _adaptive_live_rollout(env, rollout_steps=600, model_path=Path("outputs/ddqn_uav_3d_final.pt")):
    env.reset(randomize_start=False)
    env.uav_pos = np.array([90.0, 280.0, 38.0], dtype=np.float32)
    state = env._get_state()
    agent = _load_trained_agent(env, model_path)
    trajectory = [env.uav_pos.copy()]
    frames = []
    rewards = []
    hover_count = 0

    for step in range(rollout_steps + 1):
        metrics = env.evaluate_position(env.uav_pos)
        service_target = env.service_target_xy(metrics)
        target_dist = float(np.linalg.norm(env.uav_pos[:2] - service_target))
        frame_reward = env._reward(
            metrics,
            env.c_hover,
            0.0,
            0.0,
            target_dist,
            float(np.linalg.norm(env.uav_vel[:2])),
            update_hover=False,
        )
        rewards.append(float(frame_reward))
        frames.append(
            {
                "step": step,
                "uav": [float(v) for v in env.uav_pos],
                "users": env.users.round(3).tolist(),
                "avgRateMbps": float(metrics["avg_rate_mbps"]),
                "sumRateMbps": float(metrics["sum_rate_mbps"]),
                "cellEdgeRateMbps": float(metrics["cell_edge_rate_mbps"]),
                "coverageProbability": float(metrics["coverage_probability"]),
                "fairnessIndex": float(metrics["fairness_index"]),
                "qosSatisfaction": float(metrics["qos_satisfaction"]),
                "avgSnrDb": float(metrics["avg_snr_db"]),
                "avgSignalDbm": float(metrics["avg_signal_dbm"]),
                "avgLatencyMs": float(metrics["avg_latency_ms"]),
                "p95LatencyMs": float(metrics["p95_latency_ms"]),
                "losCount": int(metrics["los_count"]),
                "nlosCount": int(metrics["nlos_count"]),
                "batteryRemaining": max(0.0, 1.0 - env.energy_used / env.energy_init),
                "energyUsed": float(env.energy_used),
                "reward": float(np.sum(rewards)),
                "backhaulQuality": float(metrics["backhaul_quality"]),
                "serviceTarget": service_target.round(3).tolist(),
                "policySource": "DDQN" if agent is not None else "adaptive",
                "losMask": metrics["los_mask"].astype(int).tolist(),
            }
        )
        if step == rollout_steps:
            break

        if agent is None:
            action = _adaptive_action(env, metrics)
        else:
            action = _policy_action(agent, state)
            if np.linalg.norm(env.uav_vel[:2]) < 1.0:
                hover_count += 1
            else:
                hover_count = 0
            if hover_count >= 4 or target_dist > 115.0:
                action = _adaptive_action(env, metrics)
                hover_count = 0
        state, _, _, _ = env.step(action)
        trajectory.append(env.uav_pos.copy())

    return np.array(trajectory, dtype=np.float32), frames


def _series_from_trajectory(env, trajectory):
    energy_used = 0.0
    prev = trajectory[0]
    frames = []
    rewards = []
    for step, pos in enumerate(trajectory):
        metrics = env.evaluate_position(pos)
        if step > 0:
            distance = float(np.linalg.norm(pos - prev))
            energy_used += env.c_motion * (distance ** 2) + env.c_hover + 0.05 * abs(pos[2] - prev[2])
        reward = (
            0.55 * metrics["avg_rate_mbps"]
            + 90.0 * metrics["coverage_probability"]
            + 35.0 * metrics["fairness_index"]
            + 12.0 * metrics["backhaul_quality"]
            - env.energy_weight * energy_used
        )
        rewards.append(float(reward))
        frames.append(
            {
                "step": step,
                "uav": [float(v) for v in pos],
                "avgRateMbps": float(metrics["avg_rate_mbps"]),
                "sumRateMbps": float(metrics["sum_rate_mbps"]),
                "cellEdgeRateMbps": float(metrics["cell_edge_rate_mbps"]),
                "coverageProbability": float(metrics["coverage_probability"]),
                "fairnessIndex": float(metrics["fairness_index"]),
                "qosSatisfaction": float(metrics["qos_satisfaction"]),
                "avgSnrDb": float(metrics["avg_snr_db"]),
                "avgSignalDbm": float(metrics["avg_signal_dbm"]),
                "avgLatencyMs": float(metrics["avg_latency_ms"]),
                "p95LatencyMs": float(metrics["p95_latency_ms"]),
                "losCount": int(metrics["los_count"]),
                "nlosCount": int(metrics["nlos_count"]),
                "batteryRemaining": max(0.0, 1.0 - energy_used / env.energy_init),
                "energyUsed": energy_used,
                "reward": float(np.sum(rewards)),
                "backhaulQuality": float(metrics["backhaul_quality"]),
                "losMask": metrics["los_mask"].astype(int).tolist(),
            }
        )
        prev = pos
    return frames


def export_dashboard_data(output_path):
    env = UAVRelay3DEnv(seed=42)
    trajectory, frames = _adaptive_live_rollout(env)
    signal, probability = env.coverage_grid(trajectory[-1])

    env.users = np.array(frames[0]["users"], dtype=np.float32)
    initial_metrics = env.evaluate_position(trajectory[0])
    env.users = np.array(frames[-1]["users"], dtype=np.float32)
    final_metrics = env.evaluate_position(trajectory[-1])
    rewards = [frame["reward"] for frame in frames]
    eps = np.linspace(1.0, 0.06, len(frames))

    payload = {
        "meta": {
            "title": "3D UAV Trajectory Optimization for 5G Communication",
            "algorithm": "DDQN",
            "objective": "Max throughput, coverage and fairness with energy trade-off",
            "worldM": env.world_m,
            "maxSteps": None,
            "rolloutSteps": len(frames) - 1,
            "isLiveLoop": True,
            "altitudeRange": [env.min_alt, env.max_alt],
            "bandwidthMHz": env.bandwidth_hz / 1e6,
            "snrThresholdDb": env.coverage_snr_threshold_db,
        },
        "baseStation": [float(v) for v in env.base_pos],
        "users": env.users.tolist(),
        "trajectory": trajectory.tolist(),
        "frames": frames,
        "coverageMap": {
            "size": env.grid_size,
            "signalDbm": signal.round(3).tolist(),
            "probability": probability.round(4).tolist(),
        },
        "training": {
            "reward": rewards,
            "epsilon": eps.round(4).tolist(),
            "avgReward": float(np.mean(rewards[-10:])),
        },
        "comparison": {
            "initialAvgSignalDbm": float(initial_metrics["avg_signal_dbm"]),
            "finalAvgSignalDbm": float(final_metrics["avg_signal_dbm"]),
            "initialCoverage": float(initial_metrics["coverage_probability"]),
            "finalCoverage": float(final_metrics["coverage_probability"]),
            "initialFairness": float(initial_metrics["fairness_index"]),
            "finalFairness": float(final_metrics["fairness_index"]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=json_default)
    return output_path


if __name__ == "__main__":
    path = export_dashboard_data(Path("dashboard/data/dashboard_data.json"))
    print(f"saved {path}")
