from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import coerce_action_range, create_single_env, load_config
from utils import resolve_env_name


def draw_city(seed: int, config_path: str, output_dir: Path) -> None:
    config = load_config(config_path)
    city_cfg = config.setdefault("env", {}).setdefault("mujoco", {}).setdefault("city", {})
    city_cfg["seed"] = int(seed)
    config["env"]["mujoco"]["multimodal"]["enabled"] = False
    action_range = coerce_action_range(config["env"]["action_range"])
    env = create_single_env(config, action_range, dict(config.get("reward", {}) or {}), resolve_env_name(config, "mujoco"), render_mode=None)
    try:
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180)
        for obstacle in getattr(env, "obstacles", []):
            aabb_min = obstacle.get("aabb_min")
            aabb_max = obstacle.get("aabb_max")
            if aabb_min is None or aabb_max is None:
                continue
            x0, y0 = float(aabb_min[0]), float(aabb_min[1])
            x1, y1 = float(aabb_max[0]), float(aabb_max[1])
            height = float(aabb_max[2] - aabb_min[2])
            color_value = min(max(height / max(float(env.z_ceil - env.z_floor), 1.0), 0.0), 1.0)
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor=plt.cm.viridis(color_value),
                    edgecolor="#2f2f2f",
                    linewidth=0.25,
                    alpha=0.9,
                )
            )
        ax.set_xlim(float(env.world_x_range[0]), float(env.world_x_range[1]))
        ax.set_ylim(float(env.world_y_range[0]), float(env.world_y_range[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"City layout seed {seed}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, linewidth=0.2, alpha=0.25)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"city{seed}_layout.png")
        fig.savefig(output_dir / f"city{seed}_layout.pdf")
        plt.close(fig)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw top-down city layouts for revision figures.")
    parser.add_argument("--config", default="config/HW_GA_SAC/config_mujoco_HW_GA_SAC_seed1.yaml")
    parser.add_argument("--city-seeds", default="1,2,3,4,42")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/revision_minimal/figures/city_layouts"))
    args = parser.parse_args()

    for seed_text in args.city_seeds.split(","):
        seed = int(seed_text.strip())
        print(f"[city] drawing city seed {seed}")
        draw_city(seed, args.config, args.output_dir)
    print(f"[city] wrote layouts to {args.output_dir}")


if __name__ == "__main__":
    main()
