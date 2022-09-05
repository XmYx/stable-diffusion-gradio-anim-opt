import PIL
import gradio as gr
import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm.notebook import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import accelerate
import mimetypes
import gc
from basicsr.utils import imwrite
import cv2
from gfpgan import GFPGANer
from io import BytesIO
import random

#Implementing InPaint interface
import os
import sys
import torch
import inspect
from typing import Any, Callable, Dict, List, Type, Optional, Union, Tuple
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from abc import ABC, abstractmethod
import paddlehub as hub

masking_model = hub.Module(name='U2Net')
os.makedirs("/content/waifu-diffusion/diffusers-cache", exist_ok=True)
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir="diffusers-cache",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
)


mimetypes.init()
mimetypes.add_type('application/javascript', '.js')


import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def preprocess_init_image(image: Image, width: int, height: int):
    image = image.resize((width, height), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def preprocess_mask(mask: Image, width: int, height: int):
    mask = mask.convert("L")
    mask = mask.resize((width // 8, height // 8), resample=Image.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = torch.from_numpy(mask)
    return mask

def infer(img, masking_option, prompt, width, height, prompt_strength, num_outputs, num_inference_steps, guidance_scale, seed, GFPGAN: bool, bg_upsampling: bool, upscale: int, outdir: str):

    print(f'prompt: {prompt}')
    if masking_option == "automatic (U2net)":
        result = masking_model.Segmentation(
            images=[cv2.cvtColor(np.array(img["image"]), cv2.COLOR_RGB2BGR)],
            paths=None,
            batch_size=1,
            input_size=320,
            output_dir='output',
            visualization=True)
        init_image = img["image"]
        mask = Image.fromarray(result[0]['mask']) #Black pixels are inpainted and white pixels are preserved. Experimental feature, tends to work better with prompt strength of 0.5-0.7
    else:
        init_image = img["image"]
        mask = img['mask']

    images_list = predictor.predict(prompt, init_image, mask, outdir, width=width, height=height, prompt_strength=prompt_strength, num_outputs=num_outputs, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    f = []
    message = ''
    rand = random.randint(1,9999999)
    for i in range(len(images_list)):
        aaa = seed
#        message+= f'Request "{images_list[i][1]}" was saved to folder: {aaa}/ \n'

#            cfg=cfg_scales
        pt = f'{outpath}/{aaa}_{rand}_{i}.jpg'

        if GFPGAN:
            (Image.fromarray(FACE_RESTORATION(images_list[i], bg_upsampling, upscale).astype(np.uint8))).save(pt, format = 'JPEG', optimize = True)
        else:
            images_list[i].save(pt, format = 'JPEG', optimize = True)
        f.append(pt)
        with Image.open(f[i]) as img:
            print(img.size)

    print(images_list)
    return images_list, mask

def FACE_RESTORATION(image, bg_upsampling, upscale):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True)
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    model_path = os.path.join('/content/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth')
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None
        )

    image=np.array(image)
    input_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img, has_aligned=False, only_center_face=False, paste_back=True)

    image = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    image = bg_upsampler.enhance(image, outscale=upscale)[0]
    return image

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img_pil(img_pil):
    image = img_pil.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    print(f"cropped image to size ({w}, {h})")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_img(path):
    return load_img_pil(Image.open(path))


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

#from https://github.com/lstein/stable-diffusion/blob/main/ldm/simplet2i.py
def split_weighted_subprompts(text):
        """
        grabs all text up to the first occurrence of ':'
        uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
        if ':' has no value defined, defaults to 1.0
        repeats until no text remaining
        """
        remaining = len(text)
        prompts = []
        weights = []
        while remaining > 0:
            if ":" in text:
                idx = text.index(":") # first occurrence from start
                # grab up to index as sub-prompt
                prompt = text[:idx]
                remaining -= idx
                # remove from main text
                text = text[idx+1:]
                # find value for weight
                if " " in text:
                    idx = text.index(" ") # first occurence
                else: # no space, read to end
                    idx = len(text)
                if idx != 0:
                    try:
                        weight = float(text[:idx])
                    except: # couldn't treat as float
                        print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                        weight = 1.0
                else: # no value found
                    weight = 1.0
                # remove from main text
                remaining -= idx
                text = text[idx+1:]
                # append the sub-prompt and its weight
                prompts.append(prompt)
                weights.append(weight)
            else: # no : found
                if len(text) > 0: # there is still text though
                    # take remainder as weight 1
                    prompts.append(text)
                    weights.append(1.0)
                remaining = 0
        return prompts, weights


config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "/gdrive/My Drive/model.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.half().to(device)

class StableDiffusionImg2ImgPipeline(DiffusionPipeline):
    """
    From https://github.com/huggingface/diffusers/pull/241
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: Optional[torch.FloatTensor],
        mask: Optional[torch.FloatTensor],
        width: int,
        height: int,
        prompt_strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Image:
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if prompt_strength < 0 or prompt_strength > 1:
            raise ValueError(
                f"The value of prompt_strength should in [0.0, 1.0] but is {prompt_strength}"
            )

        if mask is not None and init_image is None:
            raise ValueError(
                "If mask is defined, then init_image also needs to be defined"
            )

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must both be divisible by 8")

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        if init_image is not None:
            init_latents_orig, latents, init_timestep = self.latents_from_init_image(
                init_image,
                prompt_strength,
                offset,
                num_inference_steps,
                batch_size,
                generator,
            )
        else:
            latents = torch.randn(
                (batch_size, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=self.device,
            )
            init_timestep = num_inference_steps

        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self.embed_text(
            prompt, do_classifier_free_guidance, batch_size
        )

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        mask_noise = torch.randn(latents.shape, generator=generator, device=self.device)

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
                "prev_sample"
            ]

            # replace the unmasked part with original latents, with added noise
            if mask is not None:
                timesteps = self.scheduler.timesteps[t_start + i]
                timesteps = torch.tensor(
                    [timesteps] * batch_size, dtype=torch.long, device=self.device
                )
                noisy_init_latents = self.scheduler.add_noise(init_latents_orig, mask_noise, timesteps)
                latents = noisy_init_latents * mask + latents * (1 - mask)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
#        safety_cheker_input = self.feature_extractor(
#            self.numpy_to_pil(image), return_tensors="pt"
#        ).to(self.device)
#        image, has_nsfw_concept = self.safety_checker(
#            images=image, clip_input=safety_cheker_input.pixel_values
#        )

        image = self.numpy_to_pil(image)

#        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}
        return {"sample": image}
#        return image:["sample"]

    def latents_from_init_image(
        self,
        init_image: torch.FloatTensor,
        prompt_strength: float,
        offset: int,
        num_inference_steps: int,
        batch_size: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
        # encode the init image into latents and scale the latents
        init_latents = self.vae.encode(init_image.to(self.device)).sample()
        init_latents = 0.18215 * init_latents
        init_latents_orig = init_latents

        # prepare init_latents noise to latents
        init_latents = torch.cat([init_latents] * batch_size)

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * prompt_strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size, dtype=torch.long, device=self.device
        )

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        return init_latents_orig, init_latents, init_timestep

    def embed_text(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool,
        batch_size: int,
    ) -> torch.FloatTensor:
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

MODEL_CACHE = "diffusers-cache"

class BasePredictor(ABC):
    def setup(self) -> None:
        """
        An optional method to prepare the model so multiple predictions run efficiently.
        """

    @abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """
        Run a single prediction on the model
        """

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

#        self.pipe.safety_checker = dummy

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self, prompt, init_image, mask, outdir, width=512, height=512, prompt_strength=0.8, num_outputs=1, num_inference_steps=50, guidance_scale=7.5, seed=None):
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if init_image:
            init_image = init_image.convert("RGB")
            init_image = preprocess_init_image(init_image, width, height).to("cuda")
        if mask:
            mask = mask.convert("RGB")
            mask = ImageOps.invert(mask)
            mask = preprocess_mask(mask, width, height).to("cuda")

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            init_image=init_image,
            mask=mask,
            width=width,
            height=height,
            prompt_strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )
#        if output["nsfw_content_detected"]:
#            raise Exception("NSFW content detected, please try a different prompt")


        print(output["sample"])
        return output["sample"] #output_paths

predictor = Predictor()
predictor.setup()

def dream(prompt: str, init_img, ddim_steps: int, plms: bool, fixed_code: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scales: str, denoising_strength: float, seed: int, height: int, width: int, same_seed: bool, GFPGAN: bool, bg_upsampling: bool, upscale: int, outdir: str):
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()



    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=height,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=width,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/gdrive/My Drive/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    rng_seed = seed_everything(seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    if plms and init_img == None:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    seedit = 0

    if fixed_code and init_img == None:
        start_code = torch.randn([n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if init_img != None:
        image = init_img.convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        w, h = map(lambda x: x - x % 32, (opt.W, opt.H))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        print(f"cropped image to size ({w}, {h})")
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    if len(cfg_scales) > 1: cfg_scales = list(map(float, cfg_scales.split(' ')))
    output_images = []
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        with precision_scope("cuda"):
            if init_img != None:
                init_image = 2.*image - 1.
                init_image = init_image.to(device)
                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                x0 = init_latent

                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

                assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
                t_enc = int(denoising_strength * ddim_steps)
                print(f"target t_enc is {t_enc} steps")
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                        output_images.append([])

                        aaa = f'{rng_seed + seedit}{random.randint(8, 10000)}'
                        output_images[-1].append(aaa)
                        output_images[-1].append(prompts[0])

#                        os.makedirs(f'{outpath}/{aaa}', exist_ok=True)
                        for n in trange(n_iter, desc="Sampling", disable=not accelerator.is_main_process):


                            with open(f'{outpath}/{aaa}_prompt.txt', 'w') as f:
                                f.write(prompts[0])

                            if n_iter > 1: seedit += 1
                            for cfg in tqdm(cfg_scales, desc="cfg_scales", disable=not accelerator.is_main_process):
                                cfg_scale = cfg
                                uc = None
                                if cfg_scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)


                                #from https://github.com/lstein/stable-diffusion/blob/main/ldm/simplet2i.py
                                subprompts,weights = split_weighted_subprompts(prompts[0])

                                if len(subprompts) > 1:
                                    # i dont know if this is correct.. but it works
                                    c = torch.zeros_like(uc)
                                    # get total weight for normalizing
                                    totalWeight = sum(weights)
                                    # normalize each "sub prompt" and add it
                                    for i in range(0,len(subprompts)):
                                        weight = weights[i]
                                        weight = weight / totalWeight
                                        c = torch.add(c,model.get_learned_conditioning(subprompts[i]), alpha=weight)


                                else: # just standard 1 prompt
                                    c = model.get_learned_conditioning(prompts)

                                torch.manual_seed(rng_seed + seedit) # changes manual seeding procedure
                                sigmas = model_wrap.get_sigmas(ddim_steps)
                                if init_img == None:
                                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                    x = torch.randn([n_samples, *shape], device=device) * sigmas[0] # for GPU draw
                                else:
                                    noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1] # for GPU draw
                                    x = x0 + noise
                                    sigmas = sigmas[ddim_steps - t_enc - 1:]

                                model_wrap_cfg = CFGDenoiser(model_wrap)
                                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}
                                samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = accelerator.gather(x_samples_ddim)

                                if accelerator.is_main_process and not opt.skip_save:
                                    for x_sample in x_samples_ddim:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}-{rng_seed + seedit}-{cfg_scale}_{prompt.replace(' ', '_')[:128]}.png"))
                                        output_images[-1].append(Image.fromarray(x_sample.astype(np.uint8)))
                                        print(prompt, cfg_scale, rng_seed + seedit)
                                        #output_images[-1].show() #not working
                                        #display(output_images[-1])#not working
                                        ########
                                        base_count += 1
                                        if not same_seed: seedit += 1



                toc = time.time()
                gc.collect()
                torch.cuda.empty_cache()
    del sampler



    rand1 = random.randint(0,9999999)
    f = []
    message = ''
    for i in range(len(output_images)):
        aaa = output_images[i][0]
        message+= f'Request "{output_images[i][1]}" was saved to folder: {outpath}/ \n'
        for k in range(2, len(output_images[i])):
            cfg=cfg_scales
            pt = f'{outpath}/{aaa}_{rand1}_{k-2}.jpg'
            if GFPGAN:
                (Image.fromarray(FACE_RESTORATION(output_images[i][k], bg_upsampling, upscale).astype(np.uint8))).save(pt, format = 'JPEG', optimize = True)
            else:
                output_images[i][k].save(pt, format = 'JPEG', optimize = True)
            f.append(pt)
            with Image.open(f[i]) as img:
                print(img.size)

    #files.download(f'/content/waifu-diffusion/outputs/img2img-samples/samples/0.jpg') #not working
    return f, rng_seed, message

def dev_dream(prompt: str, init_img,use_img: bool, ddim_steps: int, plms: bool, fixed_code: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scales: str, denoising_strength: float, seed: int, height: int, width: int, same_seed: bool, GFPGAN: bool, bg_upsampling: bool, upscale: int):
    prompts = list(map(str, prompt.split('|')))
    if not use_img:
        init_img=None
    f, rng_seed, message = [], [], []
    for prompt in prompts:
        ff, rng_seedf, messagef = dream(prompt, init_img, ddim_steps, plms, fixed_code, ddim_eta, n_iter, n_samples, cfg_scales, denoising_strength, seed, height, width, same_seed, GFPGAN, bg_upsampling, upscale)
        f+=ff
        rng_seed = rng_seedf
        message+=messagef
    return f, rng_seed, message

#InPaint Interface Starts here

#InPaint Interface Ends here


dream_interface = gr.Interface(
    dream,
    inputs=[
        gr.Textbox(label='Prompt',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Variable(value=None, visible=False),
        gr.Slider(minimum=1, maximum=200, step=1, label="Steps", value=50),
        gr.Checkbox(label='PLMS ', value=True, visible=False),
        gr.Checkbox(label='Sampling from one point', value=False, visible=False),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=50, step=1, label='Iterations (How many times to run)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Samples', value=1),
        gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scale, Floating number, e.g. 7.0', lines=1, value=7.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Image strength', value=0.75, visible=False),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512),
        gr.Checkbox(label='Use Same Seed', value=False),
        gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False),
        gr.Checkbox(label='BG Enhancement', value=False),
        gr.Slider(minimum=1, maximum=8, step=1, label="Upscaler, 1 to turn off", value=1),
        gr.Textbox(label='Save Dir:', value = "/gdrive/My Drive/GradIO_out/")

    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='Seed'),
        gr.Textbox(label='Saved to Drive')
    ],
    title="Stable Diffusion 1.4 text to image",
    description="             ",
)

# prompt, init_img, ddim_steps, plms, ddim_eta, n_iter, n_samples, cfg_scale, denoising_strength, seed

img2img_interface = gr.Interface(
    dream,
    inputs=[
        gr.Textbox(label='Prompt',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Image(value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=1, maximum=200, step=1, label="Steps", value=100),
        gr.Checkbox(label='PLMS ', value=True, visible=True),
        gr.Checkbox(label='Sample from one point', value=False, vivible=False),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=50, step=1, label='Iterations', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Samples', value=1),
        gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scale, Floating number, e.g. 7.0', lines=1, value=9.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Strength of AI', value=0.75),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Width", value=512),
        gr.Checkbox(label='Use Same Seed', value=False),
        gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False),
        gr.Checkbox(label='BG Enhancement', value=False),
        gr.Slider(minimum=1, maximum=8, step=1, label="Upscaler, 1 to turn off", value=1),
        gr.Textbox(label='Save Dir:', value = "/gdrive/My Drive/GradIO_out/")

    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='Seed'),
        gr.Textbox(label='Saved.')
    ],
    title="Stable Diffusion Image-to-Image",
    description="",
)

ctrbbl_interface = gr.Interface(
    dev_dream,
    inputs=[
        gr.Textbox(label='Prompt',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Image(value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", source="upload", interactive=True, type="pil"),
	gr.Checkbox(label='Use img2img, off is txt2img ', value=False, visible=True),
        gr.Slider(minimum=1, maximum=200, step=1, label="Steps", value=100),
        gr.Checkbox(label='PLMS ', value=True, visible=True),
        gr.Checkbox(label='Сэмплинг с одной точки', value=False, vivible=False),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=50, step=1, label='Iterations', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Samples', value=1),
        gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scale, Floating number, e.g. 7.0', lines=1, value=9.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Strength of AI', value=0.75),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Width", value=512),
        gr.Checkbox(label='Use Same Seed', value=False),
        gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False),
        gr.Checkbox(label='BG Enhancement', value=False),
        gr.Slider(minimum=1, maximum=8, step=1, label="Upscaler, 1 to turn off", value=1)

    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='Seed'),
        gr.Textbox(label='Image saved to Drive')
    ],
    title="Stable Diffusion multiprompt",
    description="",
)

inpaint_interface = gr.Interface(
    infer,
    inputs=[gr.Image(tool='sketch', label='Input', type='pil'),
          gr.inputs.Radio(choices=['automatic (U2net)', 'manual'], type='value', default='manual', label='Masking option'),
          gr.Textbox(label="Enter your prompt", max_lines=1),
          gr.Radio(label="Output image width", choices=[128, 256, 512, 768, 1024], value=512),
          gr.Radio(label="Output image height", choices=[128, 256, 512, 768, 1024], value=512),
          gr.Slider(label="Prompt strength", maximum = 1, value=0.6),
          gr.Radio(label="Number of images to output", choices=[1, 2, 3, 4, 5], value=1),
          gr.Slider(label="Number of denoising steps", minimum=1, maximum = 500, value=50),
          gr.Slider(label="Scale for classifier-free guidance", minimum=7.0, maximum = 20.0, value=7.5, step=0.5),
          gr.Number(label="Seed", value=int(int.from_bytes(os.urandom(2), "big")), precision=0),
          gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False),
          gr.Checkbox(label='BG Enhancement', value=False),
          gr.Slider(minimum=1, maximum=8, step=1, label="Upscaler, 1 to turn off", value=1),
          gr.Textbox(label='Save Dir:', value = "/gdrive/My Drive/GradIO_out/")
          ],
    outputs=[gr.Gallery(height='auto', label='Inpainted images'),
           gr.outputs.Image(type='pil', label='Mask')],
    title='SD Inpaint',
    description="",

)


demo = gr.TabbedInterface(interface_list=[dream_interface, img2img_interface, inpaint_interface, ctrbbl_interface], tab_names=["Dream", "Image Translation", "Inpaint Interface", "Dev inference"])

demo.launch(share=True)
