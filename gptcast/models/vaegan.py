from typing import Any
import torch
import lightning as L
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from pathlib import Path

from gptcast.models.components.vaegan import Encoder, Decoder, VectorQuantizer
from gptcast.models.components.vaegan.losses import DummyLoss
from gptcast.utils.converters import dbz_to_rainfall, rainfall_to_dbz
from gptcast.utils.downloads import download_pretrained_model


# base class for a VAE with GAN loss
class VAEGAN(L.LightningModule):
    default_checkpoint_path = Path(__file__).parent.parent.parent.resolve() / "models"

    def __init__(self,
                 aeconfig,
                 loss,
                 embed_dim,
                 *args,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 base_learning_rate=1e-4,
                 freeze_weights=False,
                 clip_grads=False,
                 **kwargs,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['loss'])
        self.automatic_optimization = False
        self.encoder = Encoder(**aeconfig)
        self.decoder = Decoder(**aeconfig)
        self.loss = loss

        if aeconfig["double_z"]:
            self.z_to_emb = torch.nn.Conv2d(2*aeconfig["z_ch"], 2*embed_dim, 1)
        else:
            self.z_to_emb = torch.nn.Conv2d(aeconfig["z_ch"], embed_dim, 1)

        self.emb_to_z = torch.nn.Conv2d(embed_dim, aeconfig["z_ch"], 1)

        self.extra_ae_layers = self.extra_ae_layers_init()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

        self.patch_size = 2**sum([1 if m.__class__.__name__ == 'Downsample' else 0 for m in self.encoder.modules()])

    @staticmethod
    def load_from_pretrained(ckpt_path: str, device: str = "cpu"):
        raise NotImplementedError

    # if we need any extra layers in the autoencoder (e.g. for the variational step)
    # we can initialize them here and return them as a list so the optimizer can find them
    def extra_ae_layers_init(self) -> list:
        return []

    # variational step of the autoencoder (e.g. vector quantization, gaussian distribution)
    # this is called after the encoder and before the decoder
    # encoder -> pre_variational -> variational_step -> pre_decode -> decoder
    def variational_step(self, x):
        raise NotImplementedError
    
    # forward pass of the autoencoder
    # usually implemented as encode -> variational_step -> decode
    # we should return two things that are used in the loss function:
    # - the reconstruction (decoder output)
    # - an extra term from the variational step (e.g. the vector quantization loss, or the KL divergence)
    def forward(self, input, **kwargs) -> tuple:
        raise NotImplementedError

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.z_to_emb(h)
        var = self.variational_step(h)
        return var

    def decode(self, z):
        z = self.emb_to_z(z)
        dec = self.decoder(z)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def _get_tensorboard_logger(self):
        for l in self.loggers:
            if isinstance(l, L.loggers.TensorBoardLogger):
                return l
        return None
    
    def log_images(self, inputs, reconstructions, step, split="train"):
        # find tensorboard logger
        logger = self._get_tensorboard_logger()
        if logger is None:
            return
        
        # make a grid of images, top row is input, bottom row is reconstruction
        grid = torch.cat([inputs.detach()[:4], reconstructions.detach()[:4]], dim=0)
        logger.experiment.add_images(f"{split}/reconstructions", grid, step, dataformats="NCHW")

        
    def training_step(self, batch, batch_idx):
        ae_opt, d_opt = self.optimizers()
        inputs = self.get_input(batch, self.hparams.image_key)
        reconstructions, extra_term = self(inputs)

        self.log("train/global_step", float(self.global_step), prog_bar=True, logger=True, on_step=True, sync_dist=True)

        # train encoder+decoder+variational 
        optimizer_idx = 0
        self.toggle_optimizer(ae_opt)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, extra_term, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        if self.hparams.clip_grads:
            self.clip_gradients(ae_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        ae_opt.step()
        self.untoggle_optimizer(ae_opt)
 
        # discriminator
        optimizer_idx = 1
        self.toggle_optimizer(d_opt)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, extra_term, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        d_opt.zero_grad()
        self.manual_backward(discloss)
        if self.hparams.clip_grads:
            self.clip_gradients(ae_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        log_dict = {**log_dict_ae, **log_dict_disc}
        # ensure all tensors in log_dict are on the same device
        log_dict = {k: v.to(inputs.device.type) if isinstance(v, torch.Tensor) else v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.hparams.image_key)
        reconstructions, extra_term = self(inputs)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, extra_term, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)        

        discloss, log_dict_disc = self.loss(inputs, reconstructions, extra_term, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/discloss", discloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        # self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        log_dict = {**log_dict_ae, **log_dict_disc}
        # ensure all tensors in log_dict are on the same device
        log_dict = {k: v.to(inputs.device.type) if isinstance(v, torch.Tensor) else v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        bs = self.trainer.datamodule.hparams.batch_size
        agb = self.trainer.accumulate_grad_batches
        ngpu = self.trainer.num_devices
        # model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        # print(agb, ngpu, bs, self.base_learning_rate)
        self.learning_rate = agb * ngpu * bs * self.hparams.base_learning_rate
        lr = self.learning_rate
        print("lr", lr)
        # lr_d = self.learning_rate
        # lr_g = self.lr_g_factor*self.learning_rate
        # print("lr_d", lr_d)
        # print("lr_g", lr_g)
        ae_layers = [self.encoder, self.decoder, self.z_to_emb, self.emb_to_z]
        ae_layers += self.extra_ae_layers
        ae_params = []
        for l in ae_layers:
            ae_params += list(l.parameters())
        opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9), foreach=True)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9), foreach=True)

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight


# VAEGAN with vector quantization (discretization) as the variational step
# this VAE creates a discrete latent space
# (n_embed is the number of discrete tokens in the latent space, discretized from the continuous space of embed_dim)
class VAEGANVQ(VAEGAN):

    @classmethod
    def load_from_pretrained(cls, ckpt_path: str, device: str = "cpu") -> 'VAEGANVQ':
        return cls.load_from_checkpoint(ckpt_path, map_location=device, freeze_weights=True, ckpt_path=None,  loss=DummyLoss(), strict=False)
    
    @classmethod
    def load_from_zenodo(cls, model_flavour: str, path: Optional[str] = None, device: str = "cpu") -> 'VAEGANVQ':
        assert model_flavour in ["vae_mae", "vae_mwae"], f"Invalid model flavour, must be one of ['vae_mae', 'vae_mwae']"
        path = cls.default_checkpoint_path if path is None else Path(path)
        ae_path = download_pretrained_model(model_flavour, path, overwrite=False)
        return cls.load_from_pretrained(ae_path, device=device)

    def extra_ae_layers_init(self) -> list:
        self.quantize = VectorQuantizer(self.hparams.n_embed, self.hparams.embed_dim, beta=0.25)
        return [self.quantize]

    def variational_step(self, x):
        return self.quantize(x)

    def forward(self, input, return_pred_indices=False, auto_pad=False) -> tuple:
        if auto_pad:
            ps = self.patch_size
            pad = (ps - input.shape[-2] % ps, ps - input.shape[-1] % ps)
            inpt = F.pad(input, (0, pad[1], 0, pad[0]), mode='reflect') if pad != (0, 0) else input
        else:
            inpt = input

        quant, diff, ind = self.encode(inpt)
        dec = self.decode(quant)

        if auto_pad and pad != (0, 0):
            dec = dec[..., :-pad[0], :-pad[1]]

        if return_pred_indices:
            return dec, (diff, ind)
        return dec, diff

    def reconstruct(self, arr: Union[np.ndarray, np.ma.MaskedArray], units: str = "mm/h") -> Union[np.ndarray, np.ma.MaskedArray]:
        assert arr.ndim in [2, 3], "Input must be 2D or 3D (steps, height, width)"
        assert units in ["mm/h", 'dbz'], "Only 'mm/h' and 'dbz' units are supported"
        
        if isinstance(arr, np.ma.MaskedArray):
            x = arr.data.copy()
            mask = arr.mask.copy()
        else:
            x = arr.copy()
            mask = None

        # add batch and channel dimensions
        x = x[None, None, ...] if x.ndim == 2 else x[:, None, ...]

        # if input is in mm/h convert back to pseudo dbz
        if units == "mm/h":
            x = x.clip(0) # this should not be necessary, but just in case
            x = rainfall_to_dbz(x)
        
        x = x.clip(0, 60) # limit the range to 0-60 dbz
        x = (x / 30.) -1 # rescale to -1, 1

        x = torch.tensor(x, dtype=torch.float32).to(self.device, memory_format=torch.contiguous_format)
        with torch.no_grad():
            y, _ = self(x, auto_pad=True)
        y = y.cpu().numpy().squeeze().clip(-1, 1)
        y = (y + 1) * 30
        if units == "mm/h":
            y = dbz_to_rainfall(y)

        y = y.astype(arr.dtype)

        if mask is not None:
            y = np.ma.masked_array(y, mask=mask)
        
        return y
        

