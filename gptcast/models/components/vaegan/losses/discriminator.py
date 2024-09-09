import torch
import torch.nn as nn
import torch.nn.functional as F

from gptcast.models.components.vaegan.losses.lpips import LPIPS


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)


# Magnitude-Weighted Absolute Error (MWAE)
# the idea is to give more weight to the pixels with higher values
def mwae(x, y):
    sx = torch.sigmoid(x)
    sy = torch.sigmoid(y)
    return torch.abs(sx - sy)*sx


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = False
        norm_layer = nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


# A base class for VAEGAN losses.
# Features:
#   - the convolutional discriminator network (NLayerDiscriminator)
#   - the discriminator loss (hinge or vanilla)
#   - the generator (VAE) loss composed of:
#       - the reconstruction loss (L1 or L2 + LPIPS perceptual loss if enabled)
#       - the ganerator loss (to fool the discriminator)
#       - the variational loss (KL or VQ) **to be implemented in the subclasses**
# The loss components of the generator can be weighted with different factors.
class VAEGANBaseLoss(nn.Module):
    def __init__(self, disc_start, disc_num_layers=3, disc_in_channels=3,
                 disc_factor=1.0, disc_weight=1.0, disc_ndf=64, disc_loss="hinge",
                 pixelloss_weight=1.0, pixel_loss="l1", perceptual_weight=1.0) -> None:
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2", "mwae"]
        assert perceptual_weight >= 0
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0:
            print(f"{self.__class__.__name__}: Running with LPIPS with weight {self.perceptual_weight}.")
            self.perceptual_loss = LPIPS().eval()

        if pixel_loss == "l1":
            self.pixel_loss = l1
        elif pixel_loss == "l2":
            self.pixel_loss = l2
        elif pixel_loss == "mwae":
            self.pixel_loss = mwae
        else:
            raise ValueError(f"Unknown pixel loss '{pixel_loss}'.")
        print(f"{self.__class__.__name__} running with {pixel_loss} reconstruction loss.")
    
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                            n_layers=disc_num_layers,
                                            ndf=disc_ndf
                                            ).apply(weights_init)

        self.discriminator_iter_start = disc_start * 2 
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"{self.__class__.__name__} running with {disc_loss} discriminator loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    # calculate the adaptive weight for the discriminator loss based on the gradients of the generator and discriminator
    # the weight is calculated as the ratio of the norm of the gradients of the NLL loss and the GAN loss
    # the weight is then clamped to a certain range and multiplied by the discriminator weight
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def perceptual_reconstruction_loss(self, inputs, reconstructions, split="train"):
        rec_loss = self.pixelloss_weight * self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        log = {
            "{}/rec_loss".format(split): rec_loss.detach().mean()
        }

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_weight * self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            log["{}/p_loss".format(split)] = p_loss.detach().mean()
            prec_loss = rec_loss + p_loss
        else:
            prec_loss = rec_loss
        log["{}/prec_loss".format(split)] = prec_loss.detach().mean()

        return prec_loss, log

    def discriminator_loss(self, inputs, reconstructions, disc_factor, split="train"):
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())

        # disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
                }
        return d_loss, log
    
    def generator_loss(self, nll_loss, reconstructions, disc_factor, last_layer=None, split="train"):
        logits_fake = self.discriminator(reconstructions.contiguous())
        g_loss = -torch.mean(logits_fake)

        try:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        # disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        loss = d_weight * disc_factor * g_loss

        log = {"{}/total_g_loss".format(split): loss.clone().detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/g_loss".format(split): g_loss.detach().mean(),
                }
        return loss, log
    
    def compute_disc_factor(self, global_step):
        return adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

    def forward(self, inputs, reconstructions, variational_term, optimizer_idx,
                global_step, last_layer=None, split="train"):
        global_log = {}

        disc_factor = self.compute_disc_factor(global_step)
        global_log["{}/disc_factor".format(split)] = torch.tensor(disc_factor)

        # now the GAN part
        if optimizer_idx == 0:
            rec_loss, log = self.perceptual_reconstruction_loss(inputs, reconstructions, split=split)
            global_log.update(log)

            nll_loss, log = self.nll_loss(rec_loss, split=split)
            global_log.update(log)

            gen_loss, log = self.generator_loss(nll_loss, reconstructions, disc_factor, last_layer=last_layer, split=split)
            global_log.update(log)

            variational_loss, log = self.variational_loss(variational_term, split=split)
            global_log.update(log)

            loss = gen_loss + nll_loss + variational_loss
            global_log["{}/total_loss".format(split)] = loss.clone().detach().mean()

            return loss, global_log

        if optimizer_idx == 1:
            # second pass for discriminator update
            d_loss, log = self.discriminator_loss(inputs, reconstructions, disc_factor, split=split)
            global_log.update(log)
            return d_loss, global_log
    
    # the nll_loss is just the reconstruction loss for VQ-VAE and the NLL loss for VAE
    def nll_loss(self, rec_loss, split="train"):
        raise NotImplementedError()
    
    # the variational_loss is the codebook loss for VQ-VAE and the KL loss for VAE
    def variational_loss(self, variational_term, split="train"):
        raise NotImplementedError()


class AdversarialVQLoss(VAEGANBaseLoss):
    def __init__(self, *args, codebook_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_weight = codebook_weight

    def nll_loss(self, rec_loss, split="train"):
        nll_loss = torch.mean(rec_loss)
        log = {"{}/nll_loss".format(split): nll_loss.detach().mean()}
        return nll_loss, log
    
    def variational_loss(self, codebook_loss, split="train"):
        codebook_loss = self.codebook_weight * codebook_loss.mean()
        log = {"{}/quant_loss".format(split): codebook_loss.detach().mean()}
        return codebook_loss, log
