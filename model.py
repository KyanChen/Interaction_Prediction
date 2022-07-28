import json
import os
import torch
import numpy as np
from torch import optim, nn, utils, Tensor
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from einops import repeat
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar


class InteractionPredictor(pl.LightningModule):
    def __init__(self, lr, n_class=13, vocab_size=0, model_dim=256, num_encoder_layers=3,):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, model_dim))

        encoder_layer = TransformerEncoderLayer(
            d_model=model_dim, dim_feedforward=model_dim*2, dropout=0.1,
            activation='gelu', batch_first=True, nhead=8
        )
        encoder_norm = LayerNorm(model_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.cls_head = nn.Sequential(
            nn.Linear(model_dim, model_dim//2),
            nn.LeakyReLU(),
            nn.Linear(model_dim//2, n_class)
        )

    def forward(self, x):
        x = self.embedding(x)  # Bx2x512
        cls = repeat(self.cls_token, 'N C -> B N C', B=x.size(0))
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        cls_logit = self.cls_head(x[:, 0, ...])
        return cls_logit

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        acc = self._calculate_acc(batch, mode="val")
        return acc

    def test_step(self, batch, batch_idx):
        acc = self._calculate_acc(batch, mode="test")
        return acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        x, labels = batch['x'], batch['label']
        preds = self.forward(x)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def _calculate_acc(self, batch, mode="test"):
        x, labels = batch['x'], batch['label']
        preds = self.forward(x)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_acc" % mode, acc)
        return acc


def train_model(root_dir='EXP1', train_loader=None, val_loader=None, **model_kwargs):
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        default_root_dir=root_dir,
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(
                monitor='val_acc',
                dirpath=root_dir+'/checkpoints',
                filename='epoch_{epoch:02d}_valacc{val_acc:.2f}',
                auto_insert_metric_name=False,
                every_n_epochs=10,
                save_last=True,
                save_top_k=5,
                mode='max'
            ),
            LearningRateMonitor("epoch"),
            TQDMProgressBar()
        ],
    )
    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(root_dir+'/checkpoints', "last.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = InteractionPredictor.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = InteractionPredictor(**model_kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = InteractionPredictor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocab, interaction2id):
        super(MyDataset, self).__init__()
        self.data = np.loadtxt(data_path, dtype=str, delimiter='\t')
        self.vocab = json.load(open(vocab, 'r'))
        self.interaction2id = json.load(open(interaction2id, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1, x2, label = self.data[idx]
        x1 = self.vocab[x1]
        x2 = self.vocab[x2]
        label = self.interaction2id[label]
        x = torch.tensor([x1, x2])
        return {'x': x, 'label': label}


if __name__ == '__main__':
    vocab_file = 'data/gene_vocab.json'
    interaction2id_file = 'data/interaction2id.json'
    num_vocab = len(json.load(open(vocab_file, 'r')))
    num_class = len(json.load(open(interaction2id_file, 'r')))

    train_dataset = MyDataset(
        data_path='data/train.txt', vocab=vocab_file, interaction2id=interaction2id_file)
    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=1024, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_dataset = MyDataset(
        data_path='data/val.txt', vocab=vocab_file, interaction2id=interaction2id_file)
    val_loader = utils.data.DataLoader(
        val_dataset,
        batch_size=1024, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    model_kwargs = dict(lr=1e-4, n_class=num_class, vocab_size=num_vocab)
    model, result = train_model(
        root_dir='results/EXP1', train_loader=train_loader, val_loader=val_loader, **model_kwargs)
