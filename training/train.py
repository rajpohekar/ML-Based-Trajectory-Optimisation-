import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from ddqn_agent import DDQNAgent
from env_3d import UAVRelay3DEnv


def run_training(episodes, output_dir):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = UAVRelay3DEnv(seed=seed)
    agent = DDQNAgent(obs_dim=env.state_dim, n_actions=env.n_actions, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon, epsilon_end, decay = 1.0, 0.01, 0.998
    rewards = []
    losses = []
    best_reward = -1e18
    best_trajectory = None

    for ep in range(1, episodes + 1):
        state = env.reset(randomize_start=True)
        done = False
        ep_reward = 0.0
        trajectory = [env.uav_pos.copy().tolist()]
        step = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            if step % 4 == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
            state = next_state
            ep_reward += reward
            step += 1
            trajectory.append(env.uav_pos.copy().tolist())

        rewards.append(ep_reward)
        epsilon = max(epsilon_end, epsilon * decay)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_trajectory = trajectory
            torch.save(agent.policy.state_dict(), output_dir / "ddqn_uav_3d.pt")

        if ep % 25 == 0:
            print(f"episode={ep}/{episodes} reward={ep_reward:.2f} avg25={np.mean(rewards[-25:]):.2f} eps={epsilon:.3f}", flush=True)

    torch.save(agent.policy.state_dict(), output_dir / "ddqn_uav_3d_final.pt")
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": episodes,
                "rewards": rewards,
                "losses": losses,
                "best_reward": best_reward,
                "best_trajectory": best_trajectory,
                "device": device,
            },
            f,
            indent=2,
        )

    return output_dir / "training_history.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    path = run_training(args.episodes, args.output_dir)
    print(f"saved {path}")
