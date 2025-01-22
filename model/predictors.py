import torch
import torch.optim as optim
import torchmetrics
import torch.nn as nn
from einops import rearrange
import pytorch_lightning as pl

from .encoders import Encoder
from .decoder import Decoder
from .utils.loss_functions import WeightedFocalLoss, BCEWithWeights


class SwinUnetFirePredictor(pl.LightningModule):
    def __init__(self, in_channels, wind_dim, landscape_dim, hidden_dim, layers, heads, head_dim, window_size, dropout=0.0, loss_type='bce', learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Loss Function
        self.learning_rate = learning_rate
        if loss_type == "focal":
            alpha = 0.25
            gamma = 2.0
            self.loss_fn = WeightedFocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == "weighted_bce":
            pass
        else:
            self.loss_fn = BCEWithWeights()  # Default BCE

        # Metrics for training
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()

        # Metrics for training
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

        self.encoder = Encoder(in_channels + landscape_dim, hidden_dim, layers[0], 4, heads[0], head_dim, window_size, dropout)
        self.wind_fc = nn.Linear(wind_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, 1, layers[1], 4, heads[1], head_dim, window_size, dropout)

    def forward(self, fire_sequence, landscape_features, wind_sequence, valid_tokens):
        wind_latent = self.wind_fc(wind_sequence.mean(dim=1))
        wind_latent = rearrange(wind_latent, 'b d -> b d 1 1 1')

        # Align temporal dimensions of landscape_features and fire_sequence
        landscape_features = landscape_features[:, :, :fire_sequence.size(2), :, :]
        assert landscape_features.size(2) == fire_sequence.size(2), "Temporal dimension mismatch"

        fire_landscape_combined = torch.cat([fire_sequence, landscape_features], dim=1)

        encoded = self.encoder(fire_landscape_combined)
        encoded += wind_latent

        predicted_mask = self.decoder(encoded)
        return predicted_mask

    def training_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq, static_data, wind_inputs, valid_tokens)
        loss = self.loss_fn(pred, isochrone_mask)
        self.log("train_loss", loss)
        # Update metrics
        pred_binary = (torch.sigmoid(pred) > 0.5).float()  # Convert logits to binary predictions
        # Flatten
        pred_binary = pred_binary.flatten()
        isochrone_mask_flattened= isochrone_mask.flatten().int()
        self.train_accuracy(pred_binary, isochrone_mask_flattened)
        self.train_precision(pred_binary, isochrone_mask_flattened)
        self.train_recall(pred_binary, isochrone_mask_flattened)
        self.train_f1(pred_binary, isochrone_mask_flattened)
        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=False)
        self.log("train_precision", self.train_precision, on_step=True, on_epoch=False)
        self.log("train_recall", self.train_recall, on_step=True, on_epoch=False)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)
        return {"loss": loss, "predictions": pred, "targets": isochrone_mask}

    def validation_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq, static_data, wind_inputs, valid_tokens)
        loss = self.loss_fn(pred, isochrone_mask)
        self.log("val_loss", loss, prog_bar=True)

        # Update Metrics
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        # Flatten
        pred_binary = pred_binary.flatten()
        isochrone_mask_flattened = isochrone_mask.flatten().int()
        self.val_accuracy(pred_binary, isochrone_mask_flattened)
        self.val_precision(pred_binary, isochrone_mask_flattened)
        self.val_recall(pred_binary, isochrone_mask_flattened)
        self.val_f1(pred_binary, isochrone_mask_flattened)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        return {"loss": loss, "predictions": pred, "targets": isochrone_mask}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
