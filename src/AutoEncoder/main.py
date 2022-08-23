import pytorch_lightning as pl

from AutoEncoder import LitAutoEncoder,MNISTDataModule,ImageSampler

dm = MNISTDataModule(batch_size=32, data_dir="/data/MNIST/")
model = LitAutoEncoder()

trainer = pl.Trainer(
    max_epochs=10, log_every_n_steps=1, gpus=1, callbacks=[ImageSampler()]
)
trainer.fit(model, dm)