"""
Throughput benchmark for ARCEngine.

Measures actions/sec for a representative agent-style workload across:
  - simple_maze: small level, few sprites, mostly static walls (sys_static path)
  - synthetic_dense: 12 dynamic non-static sprites (worst-case render path)

Each scenario is run under several configurations:
  - raw=False, RenderMode.ALL   — what humans get over the wire (every frame, JSON-safe)
  - raw=True,  RenderMode.ALL   — what current "headless" agent code probably uses
  - raw=True,  RenderMode.FINAL — only the last frame of each action (one-obs agent)
  - raw=True,  RenderMode.NONE  — pure simulation (no rendering at all)

Reports actions/sec for each, plus speedup vs the baseline (raw=False, ALL).

Run with:
    uv run python -m bench.throughput
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arcengine import (
    ActionInput,
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    Level,
    RenderMode,
    Sprite,
)
from examples.simple_maze import SimpleMaze


def _action_seq(n: int) -> list[ActionInput]:
    actions = [
        ActionInput(id=GameAction.ACTION1),
        ActionInput(id=GameAction.ACTION2),
        ActionInput(id=GameAction.ACTION3),
        ActionInput(id=GameAction.ACTION4),
    ]
    return [actions[i % 4] for i in range(n)]


def time_actions(
    game: ARCBaseGame,
    actions: list[ActionInput],
    raw: bool,
    render_mode: RenderMode,
    trials: int = 3,
) -> float:
    for a in actions[: min(50, len(actions))]:
        game.perform_action(a, raw=raw, render_mode=render_mode)
    game.full_reset()

    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for a in actions:
            game.perform_action(a, raw=raw, render_mode=render_mode)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
        game.full_reset()
    return best


class DenseGame(ARCBaseGame):
    def step(self) -> None:
        sprites = [s for s in self.current_level.get_sprites() if "dyn" in s.tags]
        if sprites:
            s = sprites[0]
            dx = 1 if self.action.id == GameAction.ACTION4 else (-1 if self.action.id == GameAction.ACTION3 else 0)
            dy = 1 if self.action.id == GameAction.ACTION2 else (-1 if self.action.id == GameAction.ACTION1 else 0)
            self.try_move_sprite(s, dx, dy)
        self.complete_action()


def _make_dense_level(n_dynamic: int = 12, n_static_walls: int = 16) -> Level:
    sprites: list[Sprite] = []
    rng = np.random.default_rng(0)
    for i in range(n_static_walls):
        pix = rng.integers(0, 10, size=(4, 4), dtype=np.int8)
        pix[0, 0] = -1
        sprites.append(
            Sprite(
                pixels=pix, name=f"wall_{i}",
                x=(i % 8) * 6, y=(i // 8) * 6 + 30, layer=0,
                blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"],
            )
        )
    for i in range(n_dynamic):
        pix = rng.integers(0, 10, size=(3, 3), dtype=np.int8)
        sprites.append(
            Sprite(
                pixels=pix, name=f"dyn_{i}",
                x=(i % 6) * 4 + 2, y=(i // 6) * 4 + 2, layer=1,
                blocking=BlockingMode.BOUNDING_BOX, tags=["dyn"],
            )
        )
    return Level(sprites=sprites, grid_size=(64, 64))


def make_dense_game() -> DenseGame:
    return DenseGame(game_id="dense", levels=[_make_dense_level()], camera=Camera())


SCENARIOS = [
    ("simple_maze",     SimpleMaze),
    ("synthetic_dense", make_dense_game),
]

CONFIGS: list[tuple[str, bool, RenderMode]] = [
    ("raw=False ALL  ",  False, RenderMode.ALL),
    ("raw=True  ALL  ",  True,  RenderMode.ALL),
    ("raw=True  FINAL",  True,  RenderMode.FINAL),
    ("raw=True  NONE ",  True,  RenderMode.NONE),
]


def main() -> None:
    n_actions = 2000
    print(f"Throughput benchmark — {n_actions} actions per scenario, best-of-3 trials\n")

    for name, factory in SCENARIOS:
        print(f"  {name}")
        baseline_dt: float | None = None
        for label, raw, mode in CONFIGS:
            g = factory()
            dt = time_actions(g, _action_seq(n_actions), raw=raw, render_mode=mode)
            aps = n_actions / dt
            if baseline_dt is None:
                baseline_dt = dt
                speedup_str = "—"
            else:
                speedup_str = f"{baseline_dt / dt:.1f}x"
            print(f"    {label}  {aps:>10,.0f} a/s   ({speedup_str} vs baseline)")
        print()


if __name__ == "__main__":
    main()
