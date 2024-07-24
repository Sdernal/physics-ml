import logging
from argparse import ArgumentParser
from pathlib import Path
import shutil

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

from src.system import PoissonSolver
from src.callback import LogPressureCallback


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

    system = PoissonSolver(
        trn_path=args.trn_path,
        val_path=args.val_path,
        dt=args.dt,
        batch_size=args.batch_size
    )

    patience_callback = EarlyStopping(
        min_delta=0.0,
        mode='min',
        monitor='val/loss',
        patience=args.patience
    )

    log_pressure_callback = LogPressureCallback(
        wandb_logger
    )

    trainer = Trainer(accelerator=args.accelerator, logger=wandb_logger,
                      callbacks=[checkpoint_callback, patience_callback, log_pressure_callback], max_epochs=args.max_epochs)
    trainer.fit(model=system)