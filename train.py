import logging
from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

from src.system import PoissonSolver
from src.callback import LogPressureCallback


def get_normalization_values(trn_path: str, rho: float = 1, dt: float = 0.01, h: float = 1.0):
    data_archive = np.load(trn_path)
    assert len({'p', 'u', 'v'} & set(data_archive.keys())) == 3

    pt = data_archive['p']
    vt = data_archive['v']
    ut = data_archive['u']
    dx = dy = h

    g = np.zeros_like(vt)
    g[:, 1:-1, 1:-1] = ((ut[:, 2:, 1:-1] - ut[:, :-2, 1:-1]) / (2 * dx)
                             + (vt[:, 1:-1, 2:] - vt[:, 1:-1, :-2]) / (2 * dy)) * rho / dt

    g2 = np.zeros_like(g)
    g2[:, 1:-1, 1:-1] = (
                                (ut[:, 2:, 1:-1] - ut[:, :-2, 1:-1]) ** 2 / (4 * dx ** 2)
                                + (vt[:, 1:-1, 2:] - vt[:, 1:-1, :-2]) ** 2 / (4 * dy ** 2)
                                + (ut[:, 1:-1, 2:] - ut[:, 1:-1, :-2]) * (vt[:, 2:, 1:-1] - vt[:, :-2, 1:-1]) / (
                                            2 * dx * dy)
                        ) * rho

    g -= g2

    p_max = abs(pt).max()
    g_max = abs(g).max()
    return p_max, g_max


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--serialize_dir", type=str, required=True, help="Folder to store the artifacts")
    arg_parser.add_argument("--force", action="store_true", help="Delete folder if exists")
    arg_parser.add_argument("--trn_path", type=str, required=True, help="Train dataset file")
    arg_parser.add_argument("--val_path", type=str, required=True, help="Validation dataset file")
    arg_parser.add_argument("--dt", type=float, default=0.001, help="Time derivative")

    # Trainer arguments
    arg_parser.add_argument("--max_epochs", type=int, default=100)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--patience", type=int, default=20)
    arg_parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"],
                            default="cpu")

    args = arg_parser.parse_args()
    serialize_dir = Path(args.serialize_dir)
    if serialize_dir.exists():
        if args.force:
            logging.warning(f"Force flag activated. Deleting {args.serialize_dir}...")
            shutil.rmtree(args.serialize_dir)
        else:
            logging.error(f"{args.serialize_dir} already exists! Choose another folder or use --force to overwrite")
            exit(-1)

    serialize_dir.mkdir(parents=True)
    wandb_logger = WandbLogger(name=serialize_dir.name, project="physics-ml")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(serialize_dir),
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    p_max, g_max = get_normalization_values(args.trn_path, dt=args.dt)
    system = PoissonSolver(
        trn_path=args.trn_path,
        val_path=args.val_path,
        dt=args.dt,
        batch_size=args.batch_size,
        p_max=p_max,
        g_max=g_max
    )

    patience_callback = EarlyStopping(
        min_delta=0.0,
        mode='min',
        monitor='val/loss',
        patience=args.patience
    )

    log_pressure_callback = LogPressureCallback(
        wandb_logger,
        indices=[10, 500, -10],  # draw frames near begin, end and center
        p_max=p_max,
        g_max=g_max
    )

    trainer = Trainer(accelerator=args.accelerator, logger=wandb_logger,
                      callbacks=[checkpoint_callback, patience_callback, log_pressure_callback], max_epochs=args.max_epochs)
    trainer.fit(model=system)