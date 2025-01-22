import yaml
import argparse
import pytorch_lightning as pl

from utils import Logger
from ..data import FireDataModule
from callbacks import EarlyStoppingHandler, ImagePredictionLogger
from model import SwinUnetFirePredictor

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    global_config = load_yaml_config(args.global_config)
    train_config = load_yaml_config(args.train_config)
    data_config = load_yaml_config(args.data_config)
    logger_config = train_config['logger']
    callbacks_config = train_config['callbacks']
    early_stopper_config = callbacks_config['early_stopper']
    image_prediction_logger_config = callbacks_config['image_prediction_logger']

    # Logger
    logger = Logger.get_tensorboard_logger(
        save_dir=logger_config['dir'],
        name=logger_config['name']
    )

    # Callbacks
    callbacks = []
    if early_stopper_config.get('enabled', False) is True:
        early_stopping_callback = EarlyStoppingHandler.get_early_stopping_callback(
            monitor=early_stopper_config['monitor'],
            patience=early_stopper_config['patience'],
            mode=early_stopper_config['mode'],
            min_delta=early_stopper_config['min_delta']
        )
        callbacks.append(early_stopping_callback)
    if image_prediction_logger_config.get('enabled', False) is True:
        image_prediction_logger_callback = ImagePredictionLogger()
        callbacks.append(image_prediction_logger_callback)

    # Datamodule
    datamodule = FireDataModule(
        data_dir=data_config['data_dir'],
        sequence_length=data_config['sequence_length'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        drop_last=data_config['drop_last'],
        pin_memory=data_config['pin_memory'],
        seed=global_config.get('seed', 42)
    )
    datamodule.setup()

    model = SwinUnetFirePredictor()

    trainer = pl.Trainer(
            max_epoch=train_config['max_epochs'],
            accelerator=train_config['accelerator'],
            devices=train_config['devices'],
            precision=train_config['precision'],
            logger=logger,
            callbacks=callbacks
    )
    trainer.fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_config", default="configs/global_config.yaml", help="Path to global config.")
    parser.add_argument("--train_config", default="configs/train_config.yaml", help="Path to training config.")
    parser.add_argument("--data_config", default="configs/data_config.yaml", help="Path to data config.")
    args = parser.parse_args()
    main(args)
