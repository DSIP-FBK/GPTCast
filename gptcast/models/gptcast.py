import torch
import torch.nn.functional as F
import lightning as L
from typing import Optional, Union
from pathlib import Path
from gptcast.models import VAEGANVQ
from gptcast.models.components import GPT, GPTCastConfig
from gptcast.utils.converters import dbz_to_rainfall, rainfall_to_dbz
from gptcast.utils.downloads import download_pretrained_model
import numpy as np
import re
from collections import OrderedDict

import einops
from tqdm import tqdm
import math

    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class AbstractEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    

class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c


class GPTCast(L.LightningModule):
    default_checkpoint_path = Path(__file__).parent.parent.parent.resolve() / "models"

    @classmethod
    def load_from_pretrained(cls, gpt_chkpt: str, first_stage_chkpt: str, device: str = "cpu") -> 'GPTCast':
        first_stage = VAEGANVQ.load_from_pretrained(first_stage_chkpt, device=device).to(device).eval()
        
        ckpt = torch.load(gpt_chkpt, weights_only=False, map_location=device)

        vocab_size, n_embd = ckpt['state_dict']['transformer.tok_emb.weight'].shape
        block_size, n_embd2 = ckpt['state_dict']['transformer.pos_emb'].shape[-2:]
        assert n_embd == n_embd2, "Number of embeddings in token and position embeddings must match ({} != {})".format(n_embd, n_embd2)
        
        n_layer = 0
        cre = re.compile(r"transformer.blocks.(\d+)")
        for k in ckpt['state_dict'].keys():
            if match := cre.search(k):
                found = int(match.group(1))
                n_layer = max(n_layer, found)
        n_layer += 1
        if n_layer != GPTCastConfig.n_layer:
            print(f"Number of layers in checkpoint ({n_layer}) does not match the expected number of layers ({GPTCastConfig.n_layer}).")

        transformer = GPT(vocab_size, block_size, n_layer, GPTCastConfig.n_head, n_embd).to(device).eval()

        # remap state_dict keys and remove prefix
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith("transformer."):
                new_state_dict[k[12:]] = v

        transformer.load_state_dict(new_state_dict, strict=True)

        # gptcast = cls.load_from_checkpoint(gpt_chkpt, transformer=transformer, first_stage=first_stage, strict=False).to(device).eval()
        gptcast = cls(transformer=transformer, first_stage=first_stage).to(device).eval()

        return gptcast

    @classmethod
    def load_from_zenodo(cls, model_flavour: str, path: Optional[str] = None, device: str = "cpu") -> 'GPTCast':
        assert model_flavour in ["gptcast_8", "gptcast_16"], f"Invalid model flavour, must be one of ['gptcast_8', 'gptcast_16']"
        path = cls.default_checkpoint_path if path is None else Path(path)
        ae_path = download_pretrained_model("vae_mwae", path, overwrite=False)
        gpt_path = download_pretrained_model(model_flavour, path, overwrite=False)
        return cls.load_from_pretrained(gpt_path, ae_path, device=device)

    def __init__(self,
                 transformer: GPT,
                 first_stage: VAEGANVQ,
                #  permuter=None,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 first_stage_key: str = "image",
                 pkeep: float = 1.0,
                 sos_token: int = 0,
                 base_learning_rate: float = 1e-4,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['first_stage', 'transformer']) #, 'permuter'])
        self.base_learning_rate = base_learning_rate
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key

        self.first_stage_model = first_stage
        assert(self.first_stage_model.hparams.freeze_weights)
        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train

        # force unconditional training
        self.be_unconditional = True
        self.cond_stage_key = self.first_stage_key
        self.cond_stage_model = SOSProvider(self.sos_token)

        # self.permuter = Identity() if permuter is None else permuter

        self.transformer = transformer

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info.view(quant_z.shape[0], -1)
        # indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # if self.downsample_cond_size > -1:
        #     c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index: torch.Tensor, zshape: torch.Size) -> torch.Tensor:
        # index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def predict_next_index(self, context: torch.Tensor, temperature: float = 1., top_k: Optional[int] = None) -> torch.Tensor:
        assert context.ndim == 2
        logits, _ = self.transformer(context)
        # we just need the prediction for the last token
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        if top_k == 1:
            _, res = torch.topk(probs, k=1, dim=-1)
        else:
            res = torch.multinomial(probs, num_samples=1)

        assert(res.shape[1] == 1)
        return res

    @torch.no_grad()
    def predict_next_frame_indices(
        self,
        input_indices: torch.Tensor,
        c_indices: torch.Tensor,
        window_size: int,
        temperature: float = 1.,
        top_k: Optional[int] = None,
        show_progress: bool = True,
        pbar_position: int = 0
        ):
        idx_batch, idx_step, idx_h, idx_w = input_indices.shape  # b s h w

        predicted_indices = (torch.ones((idx_batch, idx_h, idx_w), dtype=input_indices.dtype, device=self.device) * -1)
        if show_progress:
            progressbar = tqdm(total=idx_h*idx_w, leave=True, desc="token", position=pbar_position)
        for i in range(0, idx_h):
            half_window = window_size // 2
            if i <= half_window:
                local_i = i
            elif idx_h - i < half_window:
                local_i = window_size - (idx_h - i)
            else:
                local_i = half_window
            for j in range(0, idx_w):
                if j <= half_window:
                    local_j = j
                elif idx_w - j < half_window:
                    local_j = window_size - (idx_w - j)
                else:
                    local_j = half_window

                i_start = i - local_i
                i_end = i_start + window_size
                j_start = j - local_j
                j_end = j_start + window_size
                # print(
                #     f"rows: {i_start}-{i_end}, cols: {j_start}-{j_end}, abs_target_pos: {i}-{j}, local_target_pos: {local_i}-{local_j}")

                conditioning = c_indices.reshape(c_indices.shape[0], -1)

                past_patches = input_indices[:, :, i_start:i_end, j_start:j_end]
                past_tokens = past_patches.reshape(past_patches.shape[0], -1)

                predicted_patch = predicted_indices[:, i_start:i_end, j_start:j_end]
                predicted_tokens = predicted_patch.reshape(predicted_patch.shape[0], -1)[:,
                                :local_i * predicted_patch.shape[1] + local_j]

                full_context = torch.cat((conditioning, past_tokens, predicted_tokens), dim=1)
                res = self.predict_next_index(full_context, temperature, top_k)
                predicted_indices[:, i, j] = res

                if show_progress:
                    progressbar.update()

        return predicted_indices
        # return {
        #     'predicted_indices': ,
        #     'predicted_quant_shape': torch.Size([idx_batch, quant_input_shape[1], idx_h, idx_w])
        # }

    @torch.no_grad()
    def predict_sequence(
        self,
        seq: torch.Tensor,
        steps: int = 1,
        window_size: Optional[int] = 16,
        padding_value: float =-1.,
        future: bool = True,
        temperature: float = 1.,
        top_k: Optional[int] = None,
        ae_precision: torch.dtype = torch.float32,
        gpt_precision: torch.dtype = torch.bfloat16,
        verbosity: int = 0,
        pbar_position: int = 0
        ) -> dict[str, torch.Tensor]:
        """
        Predict future frames for a given sequence. If future is True, the model
        will predict the next `steps` frames. If future is False, the model will
        predict the last `steps` frames of the sequence (i.e. the last `steps` frames
        are not used for prediction). The pretrained models support a maximum input sequence length of 7 steps.

        Args:
            seq (torch.Tensor): Input sequence of shape (h w s). The values should be in the range [-1, 1] where -1
                corresponds to 0 DBz and 1 corresponds to 60DBZ (or the maximum reflectivity value).
            steps (int): Number of steps to predict
            window_size (Optional[int]): GPT spatial window size (this is model dependent)
            padding_value (float): Value used for padding the input sequence
            future (bool): If True, predict the next `steps` frames, otherwise predict the last `steps` frames
            temperature (float): Temperature used for sampling the next token
            top_k (Optional[int]): If not None, sample from the top k tokens
            ae_precision (torch.dtype): Precision used for the autoencoder
            gpt_precision (torch.dtype): Precision used for the GPT model
            verbosity (int): Verbosity level
            pbar_position (int): Position of the progress bar
        """
        assert not self.transformer.training
        assert len(seq.shape) == 3
        # assert isinstance(self.first_stage_model, VQModel)
        seq = seq.to(device=self.device)
        seq = einops.rearrange(seq, 'h w s -> s h w')

        if future:
            assert steps >= 1
            input_sequence = seq
            target_sequence = None
        else:
            assert 1 <= steps < seq.shape[0]
            input_sequence = seq[:-steps]
            target_sequence = seq[-steps:]

        num_down = self.first_stage_model.encoder.num_resolutions-1
        patch_size = 2**num_down
        if window_size is None:
            # try to infer GPT spatial window size from trasnformer config
            # we assume that the transformer was trained with a temporal window size of 8
            # and derive the spatial window size from the block size
            window_size = int(math.sqrt(self.transformer.config.block_size // 8))
            print(f"Using window size of {window_size}x{window_size} tokens")

        window_size_pixel = window_size*patch_size
        # print(window_size_pixel)
        in_steps, h, w = input_sequence.shape
        assert (h >= window_size_pixel) and \
               (w >= window_size_pixel), f"Window size x patch size ({window_size}x{patch_size}={window_size_pixel})" \
                                         f" cannot be bigger than image height/width"
        # print(f"Patch size is {patch_size}x{patch_size} pixels")
        bottom_pad = (patch_size - h % patch_size) % patch_size
        right_pad = (patch_size - w % patch_size) % patch_size
        if bottom_pad + right_pad != 0:
            # print(f"Input tensor height/width is not a multiple of {patch_size}, padding tensor with {padding_value}")
            # print(f"Original size: {in_steps}steps x {h}h x {w}w,"
            #       f" padded size: {in_steps}steps x {h+bottom_pad}h x {w+right_pad}w")
            input_sequence = F.pad(input_sequence, (0, right_pad, 0, bottom_pad), value=padding_value)
            # in_steps, h, w = sequence.shape

        x = einops.rearrange(input_sequence, 's h w -> (s h) w')[None, None, ...]
        x = x.to(memory_format=torch.contiguous_format).float()
        c = x
        
        with torch.autocast(self.device.type, enabled=True if ae_precision!=torch.float32 else False, dtype=ae_precision):
            quant_input, input_indices = self.encode_to_z(x)
            x_rec = self.decode_to_img(input_indices, quant_input.shape).squeeze()
            x_rec = x_rec.reshape(in_steps, x_rec.shape[0]//in_steps, x_rec.shape[1])
        
        _, c_indices = self.encode_to_c(c)
        quant_shape = quant_input.shape
        indices = input_indices.reshape(quant_shape[0], in_steps, quant_shape[2] // in_steps, quant_shape[3])
        ind_b, ind_s, ind_h, ind_w = indices.shape
        # print(quant_input.shape, input_indices.shape, c_indices.shape, indices.shape)

        predicted_seq = list()
        disabe_step_progress = verbosity < 1
        show_token_progress = verbosity > 1
        for i in tqdm(range(steps), total=steps, desc="Timestep", disable=disabe_step_progress):
            with torch.autocast(self.device.type, enabled=True if gpt_precision!=torch.float32 else False, dtype=gpt_precision):
                predicted_indices = self.predict_next_frame_indices(indices[:, -in_steps:], c_indices, window_size,
                                                                    temperature=temperature, top_k=top_k,
                                                                    show_progress=show_token_progress,)    
            with torch.autocast(self.device.type, enabled=True if ae_precision!=torch.float32 else False, dtype=ae_precision):
                predicted_image = self.decode_to_img(
                    predicted_indices.reshape(predicted_indices.shape[0], -1),
                    torch.Size([ind_b, quant_shape[1], ind_h, ind_w])
                )
            predicted_seq.append(predicted_image)

            # append predicted_indices and remove first frame
            indices = torch.cat((indices, predicted_indices[:, None, ...]), dim=1)

        # pred_image = decoded.squeeze().numpy().clip(-1, 1)
        predicted_seq = torch.cat(predicted_seq, dim=1)
        result = {
            'input_indices': indices[:, :in_steps],
            'input_sequence_pad': input_sequence,
            'input_sequence_nopad': input_sequence[..., :h, :w],
            'input_reconstruction': x_rec[..., :h, :w],
            'pred_indices': indices[:, -in_steps:],
            'pred_sequence_pad': predicted_seq,
            'pred_sequence_nopad': predicted_seq[..., :h, :w],
        }
        if target_sequence is not None:
            result['target_sequence'] = target_sequence

        return result

    @torch.no_grad()
    def forecast(
        self,
        input_sequence: Union[np.ndarray, np.ma.MaskedArray],
        steps: int = 1,
        units: str = "mm/h",
        mask: Optional[np.ndarray] = None,
        verbosity: int = 1,
        ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        Forecast future frames for a given sequence. The model will predict the next `steps` frames.

        Args:
            x (np.ndarray or np.ma.MaskedArray): Input sequence of shape (s h w). Accepts both dbz and mm/h units.
                                                 Values are converted to dbz internally and clipped to 0-60 dbz.
                                                 That is the rage of the model. The model can leverage up to 7 input steps
                                                 of context, if the input sequence is longer, only the last 7 steps are used.
            steps (int): Number of steps to predict
            units (str): Units of the output. Can be either 'mm/h' or 'dbz'.
            mask (np.ndarray): 2D Mask to apply to all frames. Should have the same shape height x width as the input sequence.
                               If the input sequence is already a masked array, the mask will be added to the existing mask.
            verbosity (int): Verbosity level for the prediction process. 0: no output, 1: timestep output, 2: token output
        """
        assert len(input_sequence.shape) == 3
        assert units in ["mm/h", 'dbz'], "Only 'mm/h' and 'dbz' units are supported"
        assert steps >= 1
        if mask is not None:
            assert mask.shape == input_sequence.shape[1:], "Mask shape should match the input sequence shape"

        if input_sequence.shape[0] > 7:
            input_sequence = input_sequence[-7:]
            print("Input sequence is longer than 7 steps, only the last 7 steps are used.")

        # separate mask and data
        if isinstance(input_sequence, np.ma.MaskedArray):
            x_m = input_sequence.mask
            x = input_sequence.data
        else:
            x = input_sequence
            x_m = np.zeros_like(input_sequence, dtype=bool)

        # repeat mask for all input steps and add to input mask
        if mask is not None:
            mask = np.broadcast_to(mask, (input_sequence.shape[0], *mask.shape))
            x_m = np.logical_or(x_m, mask)

        # create output mask
        input_mask_sum = x_m.sum(axis=0).astype(bool)
        output_mask = np.broadcast_to(input_mask_sum, (steps, *input_mask_sum.shape))

        # if input is in mm/h convert back to pseudo dbz
        if units == "mm/h":
            x = x.clip(0) # this should not be necessary, but just in case
            x = rainfall_to_dbz(x)
        
        x = x.clip(0, 60) # limit the range to 0-60 dbz
        x = (x / 30.) -1 # rescale to -1, 1

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = einops.rearrange(x, 's h w -> h w s')

        with torch.no_grad():
            result = self.predict_sequence(x, steps=steps, future=True, window_size=None, verbosity=verbosity)
        y = result['pred_sequence_nopad'].cpu().numpy().squeeze().clip(-1,1)
        # rescale to 0-60 dbz
        y = (y + 1) * 30

        # convert back to mm/h if necessary
        if units == "mm/h":
            y = dbz_to_rainfall(y)

        # set dtype to input dtype
        y = y.astype(input_sequence.dtype)

        # apply output mask if it is not all false or if the input is a masked array
        if output_mask.any() or isinstance(input_sequence, np.ma.MaskedArray):
            y = np.ma.masked_array(y, mask=output_mask)
        
        return y

    def get_input(self, key: str, batch: dict) -> torch.Tensor:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch: dict, N: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch: dict) -> torch.Tensor:
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        bs = self.trainer.datamodule.hparams.batch_size
        agb = self.trainer.accumulate_grad_batches
        ngpu = self.trainer.num_devices
        self.learning_rate = agb * ngpu * bs * self.base_learning_rate
        return self.transformer.configure_optimizers(self.learning_rate, fused=True)
