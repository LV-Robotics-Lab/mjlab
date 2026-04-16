"""Plot recovery success-rate curves from play_record CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tyro

plt.rcParams.update({
  "font.size": 13,
  "axes.labelsize": 14,
  "axes.titlesize": 16,
  "xtick.labelsize": 13,
  "ytick.labelsize": 13,
  "legend.fontsize": 12,
  "legend.title_fontsize": 12,
})


@dataclass(frozen=True)
class PlotConfig:
  input_csv: str = "logs/play_record/recovery_success_rates.csv"
  output_png: str = "logs/play_record/recovery_success_rates.png"
  title: str = "Recovery Curves"


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
  if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
  with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
  if len(rows) == 0:
    raise RuntimeError(f"CSV has no data rows: {csv_path}")
  return rows


def main() -> None:
  cfg = tyro.cli(PlotConfig)
  csv_path = Path(cfg.input_csv)
  png_path = Path(cfg.output_png)
  rows = _read_rows(csv_path)

  grouped: dict[str, list[tuple[float, float]]] = {}
  for row in rows:
    direction = row["direction"]
    speed = float(row["speed_mps"])
    success_rate = float(row["success_rate"])
    grouped.setdefault(direction, []).append((speed, success_rate))

  color_map = {
    "front": "tab:blue",
    "left": "tab:orange",
    "back": "tab:green",
  }

  plt.figure(figsize=(8, 5))
  for direction, points in grouped.items():
    points = sorted(points, key=lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.plot(
      xs,
      ys,
      marker="o",
      linewidth=2.0,
      color=color_map.get(direction, None),
      label=direction,
    )

  plt.xlabel("Push Speed (m/s)")
  plt.ylabel("Success Rate")
  plt.ylim(0.0, 1.05)
  ax = plt.gca()
  ax.spines["top"].set_visible(False)
  plt.grid(True, linestyle="--", alpha=0.4)
  plt.legend(title="Direction")
  plt.title(cfg.title)
  plt.tight_layout()

  png_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(png_path, dpi=150)
  print(f"[DONE] Figure saved to: {png_path}")


if __name__ == "__main__":
  main()
