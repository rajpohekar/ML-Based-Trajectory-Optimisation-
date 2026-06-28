import math
import random

import numpy as np


class UAVRelay3DEnv:
    """Single-UAV 3D relay environment for energy-aware 5G coverage."""

    def __init__(
        self,
        grid_size=40,
        meters_per_cell=10,
        n_users=60,
        base_pos=(50.0, 50.0, 25.0),
        max_steps=140,
        seed=42,
    ):
        self.grid_size = grid_size
        self.meters_per_cell = meters_per_cell
        self.world_m = grid_size * meters_per_cell
        self.n_users = n_users
        self.max_steps = max_steps
        self.base_pos = np.array(base_pos, dtype=np.float32)
        self.min_alt = 15.0
        self.max_alt = 120.0
        self.step_xy = 10.0
        self.step_h = 5.0
        self.max_speed_xy = 14.0
        self.max_speed_h = 6.0
        self.velocity_smoothing = 0.34
        self.bandwidth_hz = 20e6
        self.tx_power_dbm = 30.0
        self.noise_dbm = -100.0
        self.coverage_snr_threshold_db = 8.0
        self.min_rate_mbps = 120.0
        self.latency_target_ms = 18.0
        self.energy_init = 12000.0
        self.c_motion = 0.08
        self.c_hover = 8.5
        self.energy_weight = 0.015
        self.n_dynamic_clusters = 3
        self.cluster_reform_interval = 90
        self.scatter_reform_interval = 180
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.actions = self._build_actions()
        self.n_actions = len(self.actions)
        self.cluster_centers = self._spawn_cluster_centers()
        self.initial_cluster_centers = self.cluster_centers.copy()
        self.cluster_velocities = self._spawn_cluster_velocities()
        self.cluster_assignments = self._spawn_cluster_assignments()
        self.users = self._spawn_users()
        self.initial_users = self.users.copy()
        self.user_vel = self._spawn_user_velocities()
        self.user_targets = self._spawn_user_targets()
        self.user_demands = self.rng.uniform(0.7, 1.3, self.n_users).astype(np.float32)
        self.state_dim = 15
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(0, self.world_m, grid_size),
            np.linspace(0, self.world_m, grid_size),
        )
        self.reset(randomize_start=False)

    def _build_actions(self):
        moves = []
        for dx, dy in [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ]:
            for dh in [0, 1, -1]:
                moves.append(np.array([dx, dy, dh], dtype=np.float32))
        return moves

    def _spawn_cluster_centers(self):
        anchors = np.array(
            [
                [self.world_m * 0.74, self.world_m * 0.72],
                [self.world_m * 0.62, self.world_m * 0.36],
                [self.world_m * 0.34, self.world_m * 0.78],
            ],
            dtype=np.float32,
        )
        jitter = self.rng.normal(0.0, self.world_m * 0.035, anchors.shape)
        return np.clip(anchors + jitter, 60, self.world_m - 60).astype(np.float32)

    def _spawn_cluster_velocities(self):
        angles = self.rng.uniform(0, 2 * np.pi, self.n_dynamic_clusters)
        speeds = self.rng.uniform(0.35, 0.9, self.n_dynamic_clusters)
        return np.column_stack([np.cos(angles) * speeds, np.sin(angles) * speeds]).astype(np.float32)

    def _spawn_cluster_assignments(self):
        clustered = int(self.n_users * 0.75)
        assignments = np.full(self.n_users, -1, dtype=np.int32)
        weights = self.rng.dirichlet(np.ones(self.n_dynamic_clusters) * 1.5)
        assignments[:clustered] = self.rng.choice(self.n_dynamic_clusters, clustered, p=weights)
        self.rng.shuffle(assignments)
        return assignments

    def _spawn_users(self):
        users = []
        for assignment in self.cluster_assignments:
            if assignment >= 0:
                center = self.cluster_centers[assignment]
                x = self.rng.normal(center[0], self.world_m * 0.065)
                y = self.rng.normal(center[1], self.world_m * 0.07)
                users.append([np.clip(x, 0, self.world_m), np.clip(y, 0, self.world_m)])
            else:
                users.append(self.rng.uniform(0, self.world_m, 2))
        return np.array(users, dtype=np.float32)

    def _spawn_user_velocities(self):
        speeds = self.rng.uniform(0.35, 1.8, self.n_users)
        angles = self.rng.uniform(0, 2 * np.pi, self.n_users)
        return np.column_stack([np.cos(angles) * speeds, np.sin(angles) * speeds]).astype(np.float32)

    def _spawn_user_targets(self):
        targets = []
        for assignment in self.cluster_assignments:
            if assignment >= 0:
                center = self.cluster_centers[assignment]
                x = self.rng.normal(center[0], self.world_m * 0.075)
                y = self.rng.normal(center[1], self.world_m * 0.08)
                targets.append([np.clip(x, 0, self.world_m), np.clip(y, 0, self.world_m)])
            else:
                targets.append(self.rng.uniform(0, self.world_m, 2))
        return np.array(targets, dtype=np.float32)

    def reset(self, randomize_start=True):
        self.time_step = 0
        self.energy_used = 0.0
        self.cluster_centers = self.initial_cluster_centers.copy()
        self.cluster_velocities = self._spawn_cluster_velocities()
        self.cluster_assignments = self._spawn_cluster_assignments()
        self.initial_users = self._spawn_users()
        self.users = self.initial_users.copy()
        self.user_vel = self._spawn_user_velocities()
        self.user_targets = self._spawn_user_targets()
        self.uav_vel = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.hover_steps = 0
        if randomize_start:
            self.uav_pos = np.array(
                [
                    self.rng.uniform(80, self.world_m - 80),
                    self.rng.uniform(80, self.world_m - 80),
                    self.rng.uniform(35, 75),
                ],
                dtype=np.float32,
            )
        else:
            self.uav_pos = np.array([self.world_m / 2, self.world_m / 2, 55.0], dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        mean_user = self.users.mean(axis=0)
        cluster_velocity = self.user_vel.mean(axis=0)
        metrics = self.evaluate_position(self.uav_pos)
        return np.array(
            [
                self.uav_pos[0] / self.world_m,
                self.uav_pos[1] / self.world_m,
                self.uav_pos[2] / self.max_alt,
                self.uav_vel[0] / self.max_speed_xy,
                self.uav_vel[1] / self.max_speed_xy,
                self.uav_vel[2] / self.max_speed_h,
                self.base_pos[0] / self.world_m,
                self.base_pos[1] / self.world_m,
                mean_user[0] / self.world_m,
                mean_user[1] / self.world_m,
                cluster_velocity[0] / 2.0,
                cluster_velocity[1] / 2.0,
                metrics["coverage_probability"],
                metrics["qos_satisfaction"],
                max(0.0, 1.0 - self.energy_used / self.energy_init),
            ],
            dtype=np.float32,
        )

    def step(self, action_idx):
        prev = self.uav_pos.copy()
        previous_target = self.service_target_xy()
        previous_target_dist = float(np.linalg.norm(self.uav_pos[:2] - previous_target))
        move = self.actions[action_idx].copy()
        xy_norm = np.linalg.norm(move[:2])
        if xy_norm > 0:
            move[:2] = move[:2] / xy_norm

        target_vel = np.array(
            [move[0] * self.max_speed_xy, move[1] * self.max_speed_xy, move[2] * self.max_speed_h],
            dtype=np.float32,
        )
        self.uav_vel = (1.0 - self.velocity_smoothing) * self.uav_vel + self.velocity_smoothing * target_vel
        self.uav_pos[:2] = np.clip(self.uav_pos[:2] + self.uav_vel[:2], 0, self.world_m)
        self.uav_pos[2] = np.clip(self.uav_pos[2] + self.uav_vel[2], self.min_alt, self.max_alt)
        self.time_step += 1
        self._move_users()

        distance = np.linalg.norm(self.uav_pos - prev)
        action_change = np.linalg.norm(move - self.prev_action)
        smoothness_cost = float(action_change + 0.025 * np.linalg.norm(self.uav_vel))
        step_energy = self.c_motion * (distance ** 2) + self.c_hover + 0.05 * abs(self.uav_pos[2] - prev[2])
        self.energy_used += step_energy

        metrics = self.evaluate_position(self.uav_pos)
        current_target = self.service_target_xy(metrics)
        target_dist = float(np.linalg.norm(self.uav_pos[:2] - current_target))
        progress = previous_target_dist - target_dist
        xy_speed = float(np.linalg.norm(self.uav_vel[:2]))
        reward = self._reward(metrics, step_energy, smoothness_cost, progress, target_dist, xy_speed)
        self.prev_action = move
        done = self.time_step >= self.max_steps or self.energy_used >= self.energy_init
        return self._get_state(), float(reward), done, metrics

    def _move_users(self):
        self.cluster_centers += self.cluster_velocities
        for cluster_idx in range(self.n_dynamic_clusters):
            for axis in range(2):
                if self.cluster_centers[cluster_idx, axis] < 60:
                    self.cluster_centers[cluster_idx, axis] = 60
                    self.cluster_velocities[cluster_idx, axis] *= -1
                if self.cluster_centers[cluster_idx, axis] > self.world_m - 60:
                    self.cluster_centers[cluster_idx, axis] = self.world_m - 60
                    self.cluster_velocities[cluster_idx, axis] *= -1

        # Update user velocities towards their targets with noise
        target_vec = self.user_targets - self.users
        target_dist = np.linalg.norm(target_vec, axis=1, keepdims=True) + 1e-6
        desired = target_vec / target_dist
        noise = self.rng.normal(0.0, 0.18, self.user_vel.shape)
        self.user_vel = (0.88 * self.user_vel + 0.32 * desired + noise).astype(np.float32)
        speeds = np.linalg.norm(self.user_vel, axis=1, keepdims=True) + 1e-6
        self.user_vel = self.user_vel / speeds * np.clip(speeds, 0.25, 2.2)
        self.users = np.clip(self.users + self.user_vel, 0, self.world_m)

        reached = np.linalg.norm(self.user_targets - self.users, axis=1) < 18.0
        if np.any(reached):
            new_targets = self._spawn_user_targets()
            self.user_targets[reached] = new_targets[reached]

        bounced_low = self.users <= 0
        bounced_high = self.users >= self.world_m
        self.user_vel[bounced_low | bounced_high] *= -0.65

        if self.time_step % self.cluster_reform_interval == 0:
            self._reform_clusters(scatter=False)
        if self.time_step % self.scatter_reform_interval == self.scatter_reform_interval // 2:
            self._reform_clusters(scatter=True)

        if self.time_step % 15 == 0:
            self.cluster_velocities += self.rng.normal(0, 0.12, self.cluster_velocities.shape)
            speeds = np.linalg.norm(self.cluster_velocities, axis=1, keepdims=True) + 1e-6
            self.cluster_velocities = self.cluster_velocities / speeds * np.clip(speeds, 0.35, 0.95)

    def _reform_clusters(self, scatter=False):
        if scatter:
            self.cluster_centers = self.rng.uniform(70, self.world_m - 70, self.cluster_centers.shape).astype(np.float32)
            scattered = self.rng.choice(self.n_users, size=max(8, self.n_users // 4), replace=False)
            self.cluster_assignments[scattered] = -1
        else:
            self.cluster_centers += self.rng.normal(0.0, self.world_m * 0.12, self.cluster_centers.shape)
            self.cluster_centers = np.clip(self.cluster_centers, 60, self.world_m - 60).astype(np.float32)
            reassigned = self.rng.choice(self.n_users, size=max(12, self.n_users // 3), replace=False)
            self.cluster_assignments[reassigned] = self.rng.integers(0, self.n_dynamic_clusters, len(reassigned))
        self.cluster_velocities = self._spawn_cluster_velocities()
        self.user_targets = self._spawn_user_targets()
 
    def service_target_xy(self, metrics=None):
        cluster = self._dominant_cluster_center()
        predicted_cluster = cluster + self.user_vel.mean(axis=0) * 8.0
        if metrics is not None:
            weak_mask = (metrics["rates_mbps"] < self.min_rate_mbps) | (
                metrics["snr_db"] < self.coverage_snr_threshold_db + 10.0
            )
            if np.any(weak_mask):
                predicted_cluster = 0.52 * predicted_cluster + 0.48 * self.users[weak_mask].mean(axis=0)
        relay_xy = self.base_pos[:2] * 0.48 + predicted_cluster * 0.52
        return np.clip(relay_xy, 0, self.world_m).astype(np.float32)

    def _dominant_cluster_center(self):
        best_center = self.users.mean(axis=0)
        best_score = -1
        for cluster_idx in range(self.n_dynamic_clusters):
            mask = self.cluster_assignments == cluster_idx
            if np.any(mask):
                score = int(np.sum(mask))
                if score > best_score:
                    best_score = score
                    best_center = self.users[mask].mean(axis=0)
        return best_center.astype(np.float32)

    def _reward(self, metrics, step_energy, smoothness_cost, progress=0.0, target_dist=0.0, xy_speed=0.0, update_hover=True):
        backhaul = metrics["backhaul_quality"]
        backhaul_penalty = max(0.0, 0.72 - backhaul)
        latency_penalty = max(0.0, metrics["p95_latency_ms"] - self.latency_target_ms)
        rate_penalty = max(0.0, self.min_rate_mbps - metrics["cell_edge_rate_mbps"])
        if update_hover:
            if xy_speed < 1.5 and target_dist > 28.0:
                self.hover_steps += 1
            else:
                self.hover_steps = 0
            hover_streak = self.hover_steps
        else:
            hover_streak = 0
        hover_penalty = max(0.0, 3.2 - xy_speed) + 0.8 * hover_streak
        reward = (
            0.42 * metrics["avg_rate_mbps"]
            + 0.035 * metrics["sum_rate_mbps"]
            + 115.0 * metrics["coverage_probability"]
            + 48.0 * metrics["fairness_index"]
            + 70.0 * metrics["qos_satisfaction"]
            + 42.0 * backhaul
            + 30.0 * progress
            - 0.22 * target_dist
            - 2.8 * latency_penalty
            - 0.18 * rate_penalty
            - 90.0 * backhaul_penalty
            - 32.0 * hover_penalty
            - 5.5 * smoothness_cost
            - self.energy_weight * step_energy
        )
        return reward

    def los_probability(self, horizontal_distance, altitude):
        theta = np.degrees(np.arctan2(altitude, horizontal_distance + 1e-6))
        a, b = 9.61, 0.16
        return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))

    def path_loss_db(self, uav_pos, points_xy):
        points = np.column_stack([points_xy, np.zeros(len(points_xy), dtype=np.float32)])
        d3 = np.linalg.norm(points - uav_pos[None, :], axis=1) + 1e-6
        horizontal = np.linalg.norm(points_xy - uav_pos[:2][None, :], axis=1)
        p_los = self.los_probability(horizontal, uav_pos[2])
        fc_ghz = 3.5
        fspl = 32.4 + 20.0 * np.log10(fc_ghz) + 20.0 * np.log10(d3 / 1000.0 + 1e-9)
        eta_los, eta_nlos = 1.0, 20.0
        return fspl + p_los * eta_los + (1.0 - p_los) * eta_nlos

    def evaluate_position(self, uav_pos):
        pl_db = self.path_loss_db(uav_pos, self.users)
        rx_dbm = self.tx_power_dbm - pl_db
        snr_db = rx_dbm - self.noise_dbm
        snr_linear = 10.0 ** (snr_db / 10.0)
        rates_mbps = (self.bandwidth_hz * np.log2(1.0 + snr_linear)) / 1e6
        weighted_rates = rates_mbps * self.user_demands
        distances = np.linalg.norm(self.users - uav_pos[:2][None, :], axis=1)
        propagation_ms = (np.sqrt(distances ** 2 + uav_pos[2] ** 2) / 3e8) * 1000.0
        queue_ms = 3.5 + 75.0 / (rates_mbps + 5.0)
        backhaul_dist = np.linalg.norm(uav_pos - self.base_pos)
        backhaul_quality = float(1.0 / (1.0 + (backhaul_dist / 260.0) ** 2))
        backhaul_latency_ms = 2.0 + 16.0 * (1.0 - backhaul_quality)
        latency_ms = propagation_ms + queue_ms + backhaul_latency_ms
        coverage = float(np.mean(snr_db >= self.coverage_snr_threshold_db))
        fairness = float((np.sum(weighted_rates) ** 2) / (self.n_users * np.sum(weighted_rates ** 2) + 1e-9))
        qos_mask = (
            (snr_db >= self.coverage_snr_threshold_db)
            & (rates_mbps >= self.min_rate_mbps)
            & (latency_ms <= self.latency_target_ms)
        )
        qos_satisfaction = float(np.mean(qos_mask))

        horizontal = np.linalg.norm(self.users - uav_pos[:2][None, :], axis=1)
        los_prob = self.los_probability(horizontal, uav_pos[2])
        los_mask = los_prob >= 0.5
        return {
            "avg_rate_mbps": float(np.mean(weighted_rates)),
            "sum_rate_mbps": float(np.sum(weighted_rates)),
            "cell_edge_rate_mbps": float(np.percentile(weighted_rates, 5)),
            "coverage_probability": coverage,
            "fairness_index": fairness,
            "qos_satisfaction": qos_satisfaction,
            "avg_snr_db": float(np.mean(snr_db)),
            "avg_signal_dbm": float(np.mean(rx_dbm)),
            "avg_latency_ms": float(np.mean(latency_ms)),
            "p95_latency_ms": float(np.percentile(latency_ms, 95)),
            "los_count": int(np.sum(los_mask)),
            "nlos_count": int(self.n_users - np.sum(los_mask)),
            "los_mask": los_mask,
            "rates_mbps": weighted_rates,
            "snr_db": snr_db,
            "signal_dbm": rx_dbm,
            "latency_ms": latency_ms,
            "backhaul_quality": backhaul_quality,
        }

    def coverage_grid(self, uav_pos):
        points = np.column_stack([self.grid_x.ravel(), self.grid_y.ravel()]).astype(np.float32)
        pl_db = self.path_loss_db(uav_pos, points)
        rx_dbm = self.tx_power_dbm - pl_db
        signal = rx_dbm.reshape(self.grid_size, self.grid_size)

        base_xy = self.base_pos[:2]
        d_base = np.linalg.norm(points - base_xy[None, :], axis=1).reshape(self.grid_size, self.grid_size)
        base_signal = -35.0 - 23.0 * np.log10(d_base + 8.0)
        combined = np.maximum(signal, base_signal)
        probability = 1.0 / (1.0 + np.exp(-(combined + 78.0) / 5.5))
        return combined, probability

    def heuristic_trajectory(self):
        sim_users = self.users.copy()
        sim_vel = self.user_vel.copy()
        sim_targets = self.user_targets.copy()
        pos = np.array([90.0, 280.0, 38.0], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        points = []
        for t in range(self.max_steps + 1):
            self.users = sim_users
            cluster = sim_users.mean(axis=0)
            predicted_cluster = cluster + sim_vel.mean(axis=0) * 7.0
            relay_xy = self.base_pos[:2] * 0.38 + predicted_cluster * 0.62
            desired_alt = 42.0 + 42.0 * min(1.0, np.linalg.norm(relay_xy - self.base_pos[:2]) / 310.0)
            desired = np.array([relay_xy[0], relay_xy[1], desired_alt], dtype=np.float32)
            target_vel = np.clip((desired - pos) * 0.09, [-self.max_speed_xy, -self.max_speed_xy, -self.max_speed_h], [self.max_speed_xy, self.max_speed_xy, self.max_speed_h])
            vel = 0.82 * vel + 0.18 * target_vel
            pos = np.array(
                [
                    np.clip(pos[0] + vel[0], 0, self.world_m),
                    np.clip(pos[1] + vel[1], 0, self.world_m),
                    np.clip(pos[2] + vel[2], self.min_alt, self.max_alt),
                ],
                dtype=np.float32,
            )
            points.append(pos.copy().tolist())

            target_vec = sim_targets - sim_users
            target_dist = np.linalg.norm(target_vec, axis=1, keepdims=True) + 1e-6
            desired_user = target_vec / target_dist
            sim_vel = (0.88 * sim_vel + 0.32 * desired_user).astype(np.float32)
            speeds = np.linalg.norm(sim_vel, axis=1, keepdims=True) + 1e-6
            sim_vel = sim_vel / speeds * np.clip(speeds, 0.25, 2.2)
            sim_users = np.clip(sim_users + sim_vel, 0, self.world_m)
        self.users = self.initial_users.copy()
        return np.array(points, dtype=np.float32)
