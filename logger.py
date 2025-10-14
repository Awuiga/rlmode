"""Training logger that records equity, rewards, and Sharpe ratio to TensorBoard and CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Utility for tracking RL training metrics and producing final reports."""

    def __init__(
        self,
        log_dir: Path,
        csv_path: Path,
        report_path: Path,
        *,
        flush_every: int = 100,
    ) -> None:
        self.log_dir = log_dir
        self.csv_path = csv_path
        self.report_path = report_path
        self.flush_every = flush_every

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self._buffer: List[dict] = []
        self._step_counter = 0

    def log(self, step: int, equity: float, reward: float, sharpe: float) -> None:
        """Record metrics for a training step."""
        record = {
            "step": step,
            "equity": float(equity),
            "reward": float(reward),
            "sharpe": float(sharpe),
        }
        self._buffer.append(record)
        self._step_counter += 1

        self.writer.add_scalar("equity", equity, step)
        self.writer.add_scalar("reward", reward, step)
        self.writer.add_scalar("sharpe", sharpe, step)

        if self._step_counter % self.flush_every == 0:
            self._flush()

    def close(self) -> None:
        """Finalize logging, flush buffers, and create visual report."""
        self._flush()
        self.writer.flush()
        self.writer.close()

        if not self.csv_path.exists():
            return

        df = pd.read_csv(self.csv_path)
        if df.empty:
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(df["step"], df["equity"], label="Equity")
        axes[0].set_ylabel("Equity")
        axes[0].legend()

        axes[1].plot(df["step"], df["reward"], label="Reward", color="tab:orange")
        axes[1].set_ylabel("Reward")
        axes[1].legend()

        axes[2].plot(df["step"], df["sharpe"], label="Sharpe", color="tab:green")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Sharpe")
        axes[2].legend()

        fig.tight_layout()

        png_path = self.report_path.with_suffix(".png")
        fig.savefig(png_path, dpi=200)

        pdf_path = self.report_path.with_suffix(".pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)

        plt.close(fig)

    def _flush(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        if self.csv_path.exists():
            df_prev = pd.read_csv(self.csv_path)
            df = pd.concat([df_prev, df], ignore_index=True)
        df.to_csv(self.csv_path, index=False)
        self._buffer.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Example usage of TrainingLogger.")
    parser.add_argument("--logdir", type=Path, default=Path("runs/logger_demo"))
    parser.add_argument("--csv", type=Path, default=Path("logs/metrics.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/training_report"))
    args = parser.parse_args()

    logger = TrainingLogger(args.logdir, args.csv, args.report)
    try:
        for step in range(1, 501):
            equity = 100_000 + step * 50 + 5000 * (0.5 - step / 1000)
            reward = (step % 10) - 5
            sharpe = 0.5 + 0.001 * step
            logger.log(step, equity, reward, sharpe)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
