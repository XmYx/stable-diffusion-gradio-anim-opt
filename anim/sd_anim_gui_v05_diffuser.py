import json
from pickle import FALSE
import threading, asyncio, argparse, math, os, pathlib, shutil, subprocess, sys, time, string
import cv2
import numpy as np
from numpy.typing import _16Bit
import pandas as pd
import random
from pandas.core.indexing import Sequence
import requests
import torch, torchvision
from torchvision import transforms
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
import PIL
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torchvision.utils import make_grid as mkgrid
from tqdm import tqdm, trange
from types import SimpleNamespace
from torch import autocast
import gradio as gr
from gfpgan import GFPGANer
from io import BytesIO
import fire
import gc

#Prompt-to-Promtp image editing
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from IPython.display import Markdown

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  embeds.to(dtype)

  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  if num_added_tokens == 0:
    raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))

  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
def dummy(images, **kwargs):
    return images, False


def generate_diff(prompt, num_samples, num_rows, steps, scale):
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        use_auth_token=True,
    ).to("cuda")

    pipe.safety_checker = dummy
    #prompt = "a grafitti in a favela wall with a <cat-toy> on it" #@param {type:"string"}

    #num_samples = 2 #@param {type:"number"}
    #num_rows = 2 #@param {type:"number"}

    all_images = []
    for _ in range(num_rows):
        with autocast("cuda"):
            images = pipe([prompt] * num_samples, num_inference_steps=steps, guidance_scale=scale)["sample"]
            print(images)
            all_images.append(images)
            print(all_images)
    return all_images


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help="choose which GPU to use if you have multiple", default=0)
parser.add_argument("--cli", type=str, help="don't launch web server, take Python function kwargs from this file.", default=None)
parser.add_argument("--ckpt", type=str, help="Model Path", default=None)
parser.add_argument("--no_var", type=str, help="Variations Model Path", default=None)
parser.add_argument("--var_ckpt", type=str, help="Variations Model Path", default=None)
parser.add_argument("--cfg_path", type=str, help="Config Snapshots Path", default=None)
parser.add_argument("--outdir", type=str, help="Config Snapshots Path", default=None)
parser.add_argument("--token", type=str, help="Config Snapshots Path", default=None)
parser.add_argument("--load_p2p", type=bool, help="Config Snapshots Path", default=None)
parser.add_argument("--embeds", type=bool, help="Config Snapshots Path", default=None)

opt = parser.parse_args()

sys.path.extend([
    '/content/src/taming-transformers',
    '/content/src/clip',
    '/content/stable-diffusion-gradio-anim-opt/',
    '/content/k-diffusion',
    '/content/pytorch3d-lite',
    '/content/AdaBins',
    '/content/MiDaS',
    '/content/soup',
    '/content/Real-ESRGAN'
])

import py3d_tools as p3d
from helpers import save_samples, sampler_fn
from infer import InferenceHelper
from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from difflib import SequenceMatcher
from diffusers import LMSDiscreteScheduler
from realesrgan import RealESRGANer


import nsp_pantry
from nsp_pantry import nspterminology, nsp_parse

def load_RealESRGAN(model_name: str, checking = False):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }
    RealESRGAN_dir = '/content/Real-ESRGAN'
    model_path = os.path.join(RealESRGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(model_name+".pth not found at path "+model_path)
    if checking == True:
        return True
    sys.path.append(os.path.abspath(RealESRGAN_dir))
    from realesrgan import RealESRGANer

    #if opt.esrgan_cpu or opt.extra_models_cpu:
    #    instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=False) # cpu does not support half
    #    instance.device = torch.device('cpu')
    #    instance.model.to('cpu')
    #elif opt.extra_models_gpu:
    #    instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=not opt.no_half, gpu_id=opt.esrgan_gpu)
    #else:
    instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=True)
    instance.model.name = model_name
    print("ESRGAN Loaded...")
    return instance

RealESRGAN = load_RealESRGAN('RealESRGAN_x4plus')

def convert_pil_img(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def go_big(image, prompts, passes, overlap, detail_steps, detail_scale, strength, outdir, W, H):

    passes = int(passes)
    W = int(W)
    H = int(H)
    overlap = int(overlap)
    detail_steps = int(detail_steps)

    if opt.gobigsampler == 'plms':
      ddim_eta = 0
    if opt.gobigsampler == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = CompVisDenoiser(model)


    mask = None

    dynamic_threshold = None
    static_threshold = None





  #result = processRealESRGAN(image,)
    outdir = f'{opt.outdir}/_GoBig'
    print(f'Running GoBig, Passes: {passes}')
    for _ in range(passes):
              #realesrgan2x(opt.realesrgan, os.path.join(sample_path, f"{base_filename}.png"), os.path.join(sample_path, f"{base_filename}u.png"))



              batch_size = 1
              precision_scope = autocast
              img = processRealESRGAN(img,)
              img.save(f'{outdir}/gobigsource.png')
              #base_filename = f"{base_filename}u"

              source_image = Image.open(f'{outdir}/gobigsource.png')
              og_size = (H, W) #H W
              slices, _ = grid_slice(source_image, overlap, og_size, False)

              betterslices = []
              for _, chunk_w_coords in tqdm(enumerate(slices), "Slices"):
                  chunk, coord_x, coord_y = chunk_w_coords

                  #init_image = load_img(chunk, shape=(W, H)).to(device)
                  init_image = convert_pil_img(chunk).to(device)
                  #init_image = sample_from_cv2(chunk)
                  #init_image = init_image.half().to(device)
                  init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                  init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                  if opt.gobigsampler in ['plms','ddim']:
                      sampler.make_schedule(ddim_num_steps=detail_steps, ddim_eta=0, verbose=False)

                  assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
                  t_enc = int(strength * detail_steps)

                  k_sigmas = model_wrap.get_sigmas(detail_steps)
                  k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]


                  callback = make_callback(sampler_name=opt.gobigsampler,
                        dynamic_threshold=dynamic_threshold,
                        static_threshold=static_threshold,
                        mask=mask,
                        init_latent=init_latent,
                        sigmas=k_sigmas,
                        sampler=sampler)




                  with torch.inference_mode():
                      with precision_scope("cuda"):
                          with model.ema_scope():
                              for prompts in prompts:
                                    uc = None
                                    if detail_scale != 1.0:
                                      uc = model.get_learned_conditioning(batch_size * [""])
                                    if isinstance(prompts, tuple):
                                      prompts = list(prompts)
                                    c = model.get_learned_conditioning(prompts)

                                    #
                                    if opt.gobigsampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                                        shape = [4, H // 8, W // 8]
                                        samples = sampler_fn(
                                            c=c,
                                            uc=uc,
                                            shape=shape,
                                            steps=detail_steps,
                                            use_init=True,
                                            n_samples=1,
                                            samplern=opt.gobigsampler,
                                            scale=detail_scale,
                                            model_wrap=model_wrap,
                                            init_latent=init_latent,
                                            t_enc=t_enc,
                                            device=device,
                                            cb=callback)
                                    else:
                                        # samplern == 'plms' or samplern == 'ddim':
                                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))

                                        if opt.gobigsampler == 'ddim':
                                            samples = sampler.decode(z_enc,
                                                                     c,
                                                                     t_enc,
                                                                     unconditional_guidance_scale=detail_scale,
                                                                     unconditional_conditioning=uc,
                                                                     img_callback=callback)
                                        elif opt.gobigsampler == 'plms': # no "decode" function in plms, so use "sample"
                                            shape = [4, H // 8, W // 8]
                                            samples, _ = sampler.sample(S=steps,
                                                                            conditioning=c,
                                                                            batch_size=1,
                                                                            shape=shape,
                                                                            verbose=False,
                                                                            unconditional_guidance_scale=detail_scale,
                                                                            unconditional_conditioning=uc,
                                                                            eta=ddim_eta,
                                                                            x_T=z_enc,
                                                                            img_callback=callback)


                                        # encode (scaled latent)
                                        # decode it
                                        #samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=detail_scale,
                                        #              unconditional_conditioning=uc,)

                                        x_samples = model.decode_first_stage(samples)
                                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                        for x_sample in x_samples:
                                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                            resultslice = Image.fromarray(x_sample.astype(np.uint8)).convert('RGBA')
                                            betterslices.append((resultslice.copy(), coord_x, coord_y))

              alpha = Image.new('L', og_size, color=0xFF)
              alpha_gradient = ImageDraw.Draw(alpha)
              a = 0
              i = 0
              #overlap = opt.gobig_overlap
              shape = (og_size, (0,0))
              while i < overlap:
                  alpha_gradient.rectangle(shape, fill = a)
                  a += 4
                  i += 1
                  shape = ((og_size[0] - i, og_size[1]- i), (i,i))
              mask = Image.new('RGBA', og_size, color=0)
              mask.putalpha(alpha)
              finished_slices = []
              for betterslice, x, y in betterslices:
                  finished_slice = addalpha(betterslice, mask)
                  finished_slices.append((finished_slice, x, y))
              sp = sanitize(prompts)
              # # Once we have all our images, use grid_merge back onto the source, then save
              final_output = grid_merge(source_image.convert("RGBA"), finished_slices).convert("RGB")
              final_output.save(os.path.join(outdir, f"{sp}_GoBig_Test_{random.randint(10000, 99999)}_001d.png"))
              #base_filename = f"{base_filename}d"

              torch.cuda.empty_cache()
              gc.collect()

              #put_watermark(final_output, wm_encoder)
              #final_output.save(os.path.join(outdir, f"{base_filename}.png"))
              return final_output


if opt.load_p2p == True:
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
    clip = clip_model.text_model
    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=opt.token, revision="fp16", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=opt.token, revision="fp16", torch_dtype=torch.float16)
    print('P2P models loaded')

    clip.to('cuda')
    unet.to('cuda')
    vae.to('cuda')

#load_p2p_model(opt)



import numpy as np
import random
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher


if opt.embeds:
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4" #@param {type:"string"}
    learned_embeds_path = "/content/downloaded_embedding/learned_embeds.bin"
    with open('/content/downloaded_embedding/token_identifier.txt', 'r') as file:
        placeholder_token_string = file.read()
    tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                use_auth_token=opt.token)

    text_encoder = CLIPTextModel.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    use_auth_token=opt.token)


    load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)

    print("inception")

#GoBig functions
def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
    return(im)

# Alternative method composites a grid of images at the positions provided
def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices: # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source

def grid_coords(target, original, overlap):
    #generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    #target should be the size for the gobig result, original is the size of each chunk being rendered
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
    x = center_x - int(original_x / 2)
    y = center_y - int(original_y / 2)
    center.append((x,y)) #center chunk
    uy = y #up
    uy_list = []
    dy = y #down
    dy_list = []
    lx = x #left
    lx_list = []
    rx = x #right
    rx_list = []
    while uy > 0: #center row vertical up
        uy = uy - original_y + overlap
        uy_list.append((lx, uy))
    while (dy + original_y) <= target_y: #center row vertical down
        dy = dy + original_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - original_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((lx, dy))
    while (rx + original_x) <= target_x:
        rx = rx + original_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
    # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
    render_edgex = last_coordx + original_x # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (new_edgex, new_edgey)


def processRealESRGAN(image):

        imgproc_realesrgan_model_name = 'RealESRGAN_x4plus'

        if 'x2' in imgproc_realesrgan_model_name:
            # downscale to 1/2 size
            modelMode = imgproc_realesrgan_model_name.replace('x2','x4')
        else:
            modelMode = imgproc_realesrgan_model_name
        image = image.convert("RGB")
        RealESRGAN = load_RealESRGAN(modelMode)
        result, res = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
        result = Image.fromarray(result)
        if 'x2' in imgproc_realesrgan_model_name:
            # downscale to 1/2 size
            result = result.resize((result.width//2, result.height//2), LANCZOS)

        return result
def get_resampling_mode():
    try:
        from PIL import __version__, Image
        major_ver = int(__version__.split('.')[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.
get_resampling_mode()

# Chop our source into a grid of images that each equal the size of the original render
def grid_slice(source, overlap, og_size, maximize=False):
    width, height = og_size # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap)
    if maximize == True:
        source = source.resize(new_size, get_resampling_mode()) # minor concern that we're resizing twice
        coordinates, new_size = grid_coords(source.size, og_size, overlap) # re-do the coordinates with the new canvas size
    # loc_width and loc_height are the center point of the goal size, and we'll start there and work our way out
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x+width, y+height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, new_size



def init_attention_weights(weight_tuples):
    tokens_length = clip_tokenizer.model_max_length
    weights = torch.ones(tokens_length)

    for i, w in weight_tuples:
        if i < tokens_length and i >= 0:
            weights[i] = w


    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_weights = weights.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_weights = None


def init_attention_edit(tokens, tokens_edit):
    tokens_length = clip_tokenizer.model_max_length
    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long)
    indices = torch.zeros(tokens_length, dtype=torch.long)

    tokens = tokens.input_ids.numpy()[0]
    tokens_edit = tokens_edit.input_ids.numpy()[0]

    for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
        if b0 < tokens_length:
            if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_mask = mask.to(device)
            module.last_attn_slice_indices = indices.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_mask = None
            module.last_attn_slice_indices = None


def init_attention_func():
    def new_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
            attn_slice = attn_slice.softmax(dim=-1)

            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice

                self.use_last_attn_slice = False

            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False

            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False

            attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module._attention = new_attention.__get__(module, type(module))

def use_last_tokens_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_slice = use

def use_last_tokens_attention_weights(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use

def use_last_self_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.use_last_attn_slice = use

def save_last_tokens_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.save_last_attn_slice = save

def save_last_self_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.save_last_attn_slice = save

@torch.no_grad()
def stablediffusion(prompt="", prompt_edit=None, prompt_edit_token_weights=[], prompt_edit_tokens_start=0.0, prompt_edit_tokens_end=1.0, prompt_edit_spatial_start=0.0, prompt_edit_spatial_end=1.0, guidance_scale=7.5, steps=50, seed=None, width=512, height=512, init_image=None, init_image_strength=0.5):
    #Change size to multiple of 64 to prevent size mismatches inside model
    #width = width - width % 64
    #height = height - height % 64

    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)

    #Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    device='cuda'

    #Preprocess image if it exists (img2img)
    if init_image is not None:
        #Resize and transpose for numpy b h w c -> torch b c h w
        width = int(width)
        height = int(height)
        shape=(width, height)

        init_image = init_image.resize(shape, Image.LANCZOS)
        init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
        init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

        #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

        #Move image to GPU
        init_image = init_image.to(device)

        #Encode image
        with autocast(device):
            init_latent = vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215

        t_start = steps - int(steps * init_image_strength)

    else:
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_start = 0

    #Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    latent = scheduler.add_noise(init_latent, noise, t_start).to(device)

    #Process clip
    with autocast(device):
        tokens_unconditional = clip_tokenizer("", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

        #Process prompt editing
        if prompt_edit is not None:
            tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state

            init_attention_edit(tokens_conditional, tokens_conditional_edit)

        init_attention_func()
        init_attention_weights(prompt_edit_token_weights)

        timesteps = scheduler.timesteps[t_start:]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i

            sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = (latent_model_input / ((sigma**2 + 1) ** 0.5)).to(unet.dtype)

            #Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample

            #Prepare the Cross-Attention layers
            if prompt_edit is not None:
                save_last_tokens_attention()
                save_last_self_attention()
            else:
                #Use weights on non-edited prompt when edit is None
                use_last_tokens_attention_weights()

            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample

            #Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = t / scheduler.num_train_timesteps
                if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                    use_last_tokens_attention()
                if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                    use_last_self_attention()

                #Use weights on edited prompt
                use_last_tokens_attention_weights()

                #Predict the edited conditional noise residual using the cross-attention masks
                noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample

            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        #scale and decode the image latents with vae
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)

def prompt_token(prompt, index):
    tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
    return clip_tokenizer.decode(tokens[index:index+1])

def run_p2p(prompt, prompt_edit, prompt_edit_token_weights,
            prompt_edit_tokens_start, prompt_edit_tokens_end,
            prompt_edit_spatial_start, prompt_edit_spatial_end,
            guidance_scale, steps, seed, width, height,
            init_image, init_image_strength, e_outdir):
                clip.to('cuda')
                unet.to('cuda')
                vae.to('cuda')

                if seed == '':
                    seed = None

                prompt_token(prompt, 7)

                """
                prompt=""
                prompt_edit=None
                prompt_edit_token_weights=[]
                prompt_edit_tokens_start=0.0
                prompt_edit_tokens_start=1.0
                prompt_edit_spatial_start=0.0
                prompt_edit_spatial_end=1.0
                guidance_scale=7.5
                steps=50
                seed=None
                width=512
                height=512
                init_image=None
                init_image_strength=0.5
                """
                init_image = Image.fromarray(init_image.astype(np.uint8))
                output = stablediffusion(prompt, prompt_edit, prompt_edit_token_weights,
                                      prompt_edit_tokens_start, prompt_edit_tokens_end,
                                      prompt_edit_spatial_start, prompt_edit_spatial_end,
                                      guidance_scale, steps, seed, width, height,
                                      init_image, init_image_strength)
                if prompt == None:
                    prompt = "Prompt was none"
                p_sanitized = sanitize(prompt)
                os.makedirs(e_outdir, exist_ok=True)
                output.save(os.path.join(e_outdir, f"{prompt[:128]}_{seed}_{random.randint(10000, 99999)}.png"))
                torch_gc()
                return output




class log:
    f = lambda color: lambda string: print(color + string + "\33[0m")
    black = f("\33[30m")
    red = f("\33[31m")
    green = f("\33[32m")
    yellow = f("\33[33m")
    blue = f("\33[34m")
    megenta = f("\33[35m")
    cyan = f("\33[36m")
    white = f("\33[37m")


def process_noodle_soup(text_prompts):

  new_prom = list(text_prompts.split("\n"))
  nan = "nan"
  prompt_series = pd.Series([np.nan for a in range(len(new_prom))])
  for i, nan in prompt_series.items():

    prompt_series[i] = new_prom[i]
  text_prompts = new_prom

  terms = []
  for term in terminology_database:
    if term not in terms:
      terms.append(term)

  processed_prompt_list = {}
  processed_prompts = []

  print("")
  print("text_prompts = {")
  #for pstep, pvalue in text_prompts.items():
    #print("    "+str(pstep)+": [")
  for prompt in text_prompts:
    new_prompt = prompt
    for term in terms:
      tk = '_'+term+'_'
      tc = prompt.count(tk)
      for i in range(tc):
        new_prompt = new_prompt.replace(tk, random.choice(terminology_database[term]), 1)
    processed_prompts.append(new_prompt)
    #processed_prompt_list[pstep] = processed_prompts
    for npr in processed_prompts:
      log.yellow("        \""+npr+"\",")
    print("        ],")
    processed_prompts = []
  print("}")
  return npr

class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = -1

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        try:
            pynvml.nvmlInit()
        except:
            print(f"[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n")
            return
        print(f"[{self.name}] Recording max memory usage...\n")
        handle = pynvml.nvmlDeviceGetHandleByIndex(opt.gpu)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f"[{self.name}] Stopped recording.\n")
        pynvml.nvmlShutdown()

    def read(self):
        return self.max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))

    #whitelist = set(string.ascii_lowercase + string.digits)
    #name = ''.join(c for c in tmp if c in whitelist)
    return '_'.join(tmp.split(" "))

def download_depth_models():
    def wget(url, outputdir):
        print(subprocess.run(['wget', url, '-P', outputdir], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    if not os.path.exists(os.path.join(models_path, 'dpt_large-midas-2f21e586.pt')):
        print("Downloading dpt_large-midas-2f21e586.pt...")
        wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", models_path)
    if not os.path.exists('pretrained/AdaBins_nyu.pt'):
        print("Downloading AdaBins_nyu.pt...")
        os.makedirs('pretrained', exist_ok=True)
        wget("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", 'pretrained')

def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def crash(e, s):
    global model
    global device

    print(s, '\n', e)

    del model
    del device
    del model_var
    del clip
    del unet
    del vae

    print('exiting...calling os._exit(0)')
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda")
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
    model = model.half()
    model.cuda()
    model.eval()
    return model

def load_var_model_from_config(config_var, ckpt_var, device, verbose=False, half_precision=True):
    #model.to("cpu")
    torch_gc()
    print(f"Loading model from {ckpt_var}")
    pl_sd = torch.load(ckpt_var, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model_var = instantiate_from_config(config_var.model)
    m, u = model_var.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    #model.to("cpu")
    torch_gc()
    model_var.half().to(device)
    model_var.eval()
    return model_var

terminology_database = nspterminology
models_path = opt.ckpt #@param {type:"string"}
#output_path = "/content/output" #@param {type:"string"}
mount_google_drive = False #@param {type:"boolean"}Will Remove
force_remount = False #Will Remove
model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
model_checkpoint =  "sd-v1-4.ckpt" #@param ["custom","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt"]
custom_config_path = "" #@param {type:"string"}
custom_checkpoint_path = "" #@param {type:"string"}
check_sha256 = False #@param {type:"boolean"}
load_on_run_all = True #@param {type: 'boolean'}
half_precision = True # needs to be fixed
model_map = {
    "sd-v1-4-full-ema.ckpt": {'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a'},
    "sd-v1-4.ckpt": {'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'},
    "sd-v1-3-full-ema.ckpt": {'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca'},
    "sd-v1-3.ckpt": {'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f'},
    "sd-v1-2-full-ema.ckpt": {'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a'},
    "sd-v1-2.ckpt": {'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d'},
    "sd-v1-1-full-ema.ckpt": {'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829'},
    "sd-v1-1.ckpt": {'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea'}
}

# config path
ckpt_config_path = custom_config_path if model_config == "custom" else os.path.join(models_path, model_config)
if os.path.exists(ckpt_config_path):
    print(f"{ckpt_config_path} exists")
else:
    ckpt_config_path = "/content/stable-diffusion-gradio-anim-opt/configs/stable-diffusion/v1-inference.yaml"
print(f"Using config: {ckpt_config_path}")

# checkpoint path or download
ckpt_path = custom_checkpoint_path if model_checkpoint == "custom" else os.path.join(models_path, model_checkpoint)
ckpt_valid = True
if os.path.exists(ckpt_path):
    print(f"{ckpt_path} exists")
else:
    print(f"Please download model checkpoint and place in {os.path.join(models_path, model_checkpoint)}")
    ckpt_valid = False

if check_sha256 and model_checkpoint != "custom" and ckpt_valid:
    import hashlib
    print("\n...checking sha256")
    with open(ckpt_path, "rb") as f:
        bytes = f.read()
        hash = hashlib.sha256(bytes).hexdigest()
        del bytes
    if model_map[model_checkpoint]["sha256"] == hash:
        print("hash is correct\n")
    else:
        print("hash in not correct\n")
        ckpt_valid = False

if ckpt_valid:
    print(f"Using ckpt: {ckpt_path}")

local_config = OmegaConf.load(f"{ckpt_config_path}")
model = load_model_from_config(local_config, f"{ckpt_path}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to("cpu")


ckpt_var="/gdrive/MyDrive/sd-clip-vit-l14-img-embed_ema_only.ckpt"
config_var="stable-diffusion-gradio-anim-opt/configs/stable-diffusion/sd-image-condition-finetune.yaml"
config_var = OmegaConf.load(config_var)

if not opt.no_var:
    model_var = load_var_model_from_config(config_var, opt.var_ckpt, 'cpu')

if terminology_database:
	log.green("Loaded terminology database from the pantry.")
	print("\x1B[3mMmm. Noodle Soup.\x1B[0m")
else:
	log.red("Unable to load terminology database")


#Definitions
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
    model_path = '/content/models/GFPGANv1.3.pth'
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

def add_noise(sample: torch.Tensor, noise_amt: float):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path,time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path

def load_img(path, shape):
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(path).convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_mask_img(path, shape):
    # path (str): Path to the mask image
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    mask_w_h = (shape[-1], shape[-2])
    if path.startswith('http://') or path.startswith('https://'):
        mask_image = Image.open(requests.get(path, stream=True).raw).convert('RGBA')
    else:
        mask_image = Image.open(path).convert('RGBA')
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def prepare_mask(mask_file, mask_shape, invert_mask, mask_brightness_adjust, mask_contrast_adjust):
    # path (str): Path to the mask image
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge,
    #     0 is black, 1 is no adjustment, >1 is brighter
    # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image,
    #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast

    mask = load_mask_img(mask_file, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask,(4,1,1))
    mask = np.expand_dims(mask,axis=0)
    mask = torch.from_numpy(mask)

    if invert_mask:
        mask = ( (mask - 0.5) * -1) + 0.5

    mask = np.clip(mask,0,1)
    return mask

def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

def make_callback(sampler_name, dynamic_threshold=None, static_threshold=None, mask=None, init_latent=None, sigmas=None, sampler=None, masked_noise_modifier=1.0):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image at each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(args_dict):
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if mask is not None:
            init_noise = init_latent + noise * args_dict['sigma']
            is_masked = torch.logical_and(mask >= mask_schedule[args_dict['i']], mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + args_dict['x'] * torch.where(is_masked,0,1)
            args_dict['x'].copy_(new_img)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1*static_threshold, static_threshold)
        if mask is not None:
            i_inv = len(sigmas) - i - 1
            init_noise = sampler.stochastic_encode(init_latent, torch.tensor([i_inv]*batch_size).to(device), noise=noise)
            is_masked = torch.logical_and(mask >= mask_schedule[i], mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + img * torch.where(is_masked,0,1)
            img.copy_(new_img)


    if init_latent is not None:
        noise = torch.randn_like(init_latent, device=device) * masked_noise_modifier
    if sigmas is not None and len(sigmas) > 0:
        mask_schedule, _ = torch.sort(sigmas/torch.max(sigmas))
    elif len(sigmas) == 0:
        mask = None # no mask needed if no steps (usually happens because strength==1.0)
    if sampler_name in ["plms","ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        if mask is not None:
            assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
            batch_size = init_latent.shape[0]

        callback = img_callback_
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback_

    return callback

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return sample_int8

def makevideo(outdir, mp4_p, batch_name, seed, timestring, max_frames):
    skip_video_for_run_all = False #@param {type: 'boolean'}
    fps = 12#@param {type:"number"}

    if skip_video_for_run_all == True:
        print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
    else:
        print('Saving video')
        image_path = os.path.join(outdir, f"{timestring}_%05d.png")
        mp4_path = os.path.join(mp4_p, f"{batch_name}_{seed}_{timestring}.mp4")

        print(f"{image_path} -> {mp4_path}")

        # make video

        #cmd = f'ffmpeg -y -vcodec png -r {str(fps)} -start_number {str(0)} -i {image_path} -frames:v {str(args.max_frames)} -c:v -vf fps={fps} -pix_fmt yuv420p -crf 17 -preset very_fast {mp4_path}'
        cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', image_path,
        '-frames:v', str(max_frames),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        mp4_path
        ]
        with open(os.devnull, 'wb') as devnull:
            process = subprocess.call(cmd, stdout=devnull)

    return mp4_path
        #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #stdout, stderr = process.communicate()
        #if process.returncode != 0:
        #    print(stderr)
        #    raise RuntimeError(stderr)

        #mp4 = open(mp4_path,'rb').read()
        #data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        #display.display( display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )

def DeformAnimKeys(angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, noise_schedule, strength_schedule, contrast_schedule, max_frames):
    angle_series = get_inbetweens(parse_key_frames(angle), max_frames)
    zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames)
    translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames)
    translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames)
    translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames)
    rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames)
    rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames)
    rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames)
    noise_schedule_series = get_inbetweens(parse_key_frames(noise_schedule), max_frames)
    strength_schedule_series = get_inbetweens(parse_key_frames(strength_schedule), max_frames)
    contrast_schedule_series = get_inbetweens(parse_key_frames(contrast_schedule), max_frames)
    return angle_series, zoom_series, translation_x_series, translation_y_series, translation_z_series, rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series, noise_schedule_series, strength_schedule_series, contrast_schedule_series

def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
      interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
      interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(),limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def transform_image_3d(prev_img_cv2, adabins_helper, midas_model, midas_transform, rot_mat, translate, midas_weight, near_plane, far_plane, fov, sampling_mode, padding_mode):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion

    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    # predict depth with AdaBins
    use_adabins = midas_weight < 1.0 and adabins_helper is not None
    if use_adabins:
        print(f"Estimating depth of {w}x{h} image with AdaBins...")
        MAX_ADABINS_AREA = 500000
        MIN_ADABINS_AREA = 448*448

        # resize image if too large or too small
        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2, cv2.COLOR_RGB2BGR))
        image_pil_area = w*h
        resized = True
        if image_pil_area > MAX_ADABINS_AREA:
            scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS) # LANCZOS is good for downsampling
            print(f"  resized to {depth_input.width}x{depth_input.height}")
        elif image_pil_area < MIN_ADABINS_AREA:
            scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            print(f"  resized to {depth_input.width}x{depth_input.height}")
        else:
            depth_input = img_pil
            resized = False

        # predict depth and resize back to original dimensions
        try:
            _, adabins_depth = adabins_helper.predict_pil(depth_input)
            if resized:
                adabins_depth = torchvision.transforms.functional.resize(
                    torch.from_numpy(adabins_depth),
                    torch.Size([h, w]),
                    interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC
                )
            adabins_depth = adabins_depth.squeeze()
        except:
            print(f"  exception encountered, falling back to pure MiDaS")
            use_adabins = False
        torch.cuda.empty_cache()

    if midas_model is not None:
        device = 'cuda'
        # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
        img_midas = prev_img_cv2.astype(np.float32) / 255.0
        img_midas_input = midas_transform({"image": img_midas})["image"]

        # MiDaS depth estimation implementation
        print(f"Estimating depth of {w}x{h} image with MiDaS...")
        sample = torch.from_numpy(img_midas_input).float().to(device).unsqueeze(0)
        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #if device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

        midas_depth = midas_model.forward(sample)
        midas_depth = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        midas_depth = midas_depth.cpu().detach().numpy()
        torch.cuda.empty_cache()

        # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
        midas_depth = np.subtract(50.0, midas_depth)
        midas_depth = midas_depth / 19.0

        # blend between MiDaS and AdaBins predictions
        if use_adabins:
            depth_map = midas_depth*midas_weight + adabins_depth*(1.0-midas_weight)
        else:
            depth_map = midas_depth

        depth_map = np.expand_dims(depth_map, axis=0)
        depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)
    else:
        depth_tensor = torch.ones((h, w), device=device)

    pixel_aspect = 1.0 # aspect of an individual pixel (so usually 1.0)
    near, far, fov_deg = near_plane, far_plane, fov
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    image_tensor = torchvision.transforms.functional.to_tensor(Image.fromarray(prev_img_cv2)).to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0),
        offset_coords_2d,
        mode=sampling_mode,
        padding_mode=padding_mode,
        align_corners=False
    )

    # convert back to cv2 style numpy array 0->255 uint8
    result = rearrange(
        new_image.squeeze().clamp(0,1) * 255.0,
        'c h w -> h w c'
    ).cpu().numpy().astype(np.uint8)
    return result

def next_seed(seed, seed_behavior):
    if seed_behavior == 'iter':
        if seed == -1:
            seed = random.randint(0, 2**32)
        else:
            seed += 1
    elif seed_behavior == 'fixed':
        if seed == -1:
            seed = random.randint(0, 2**32)
        else:
            pass # always keep seed the same
    else:
        seed = random.randint(0, 2**32)
    return seed

def parse_key_frames(string, prompt_parser=None):
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


#Image generator

def generate(prompt, name, outdir, GFPGAN, bg_upsampling, upscale, W, H, steps, scale, seed, samplern, n_batch, n_samples, ddim_eta, use_init, init_image, init_sample, strength, use_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, invert_mask, dynamic_threshold, static_threshold, C, f, init_c, return_latent=False, return_sample=False, return_c=False):
    opt.H = H
    opt.W = W

    precision = "autocast"
    seed_everything(seed)
    os.makedirs(outdir, exist_ok=True)
    if samplern == 'plms':
      ddim_eta = 0
    if samplern == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = CompVisDenoiser(model)
    batch_size = n_samples
    #gprompt = prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    init_latent = None
    if init_latent is not None:
        init_latent = init_latent
    elif init_sample is not None:
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_sample))
    elif use_init and init_image != None and init_image != '':
        init_image = load_img(init_image, shape=(W, H)).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    if not use_init and strength > 0:
        print("\nNo init image, but strength > 0. This may give you some strange results.\n")

    # Mask functions
    mask = None
    if use_mask:
        assert mask_file is not None, "use_mask==True: An mask image is required for a mask"
        assert use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"
        print(f'Using Mask {mask_file}')
        mask = prepare_mask(mask_file,
                            init_latent.shape,
                            mask_contrast_adjust,
                            mask_brightness_adjust,
                            invert_mask)

        mask = mask.to(device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)

    t_enc = int((1.0-strength) * steps)

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(steps)
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if samplern in ['plms','ddim']:
        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, ddim_discretize='fill', verbose=False)

    callback = make_callback(sampler_name=samplern,
                            dynamic_threshold=dynamic_threshold,
                            static_threshold=static_threshold,
                            mask=mask,
                            init_latent=init_latent,
                            sigmas=k_sigmas,
                            sampler=sampler)


    results = []
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in data:
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)


                    c = model.get_learned_conditioning(prompts)

                    if init_c != None:
                        c = init_c

                    if samplern in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                        shape = [4, H // 8, W // 8]
                        samples = sampler_fn(
                            c=c,
                            uc=uc,
                            shape=shape,
                            steps=steps,
                            use_init=use_init,
                            n_samples=n_samples,
                            samplern=samplern,
                            scale=scale,
                            model_wrap=model_wrap,
                            init_latent=init_latent,
                            t_enc=t_enc,
                            device=device,
                            cb=callback)
                    else:
                        # samplern == 'plms' or samplern == 'ddim':
                        if init_latent is not None and strength > 0:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        else:
                            if type(H) == 'int':
                              H = int(H)
                              W = int(W)
                            z_enc = torch.randn([n_samples, 4, H // 8, W // 8], device=device)
                        if samplern == 'ddim':
                            samples = sampler.decode(z_enc,
                                                     c,
                                                     t_enc,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     img_callback=callback)
                        elif samplern == 'plms': # no "decode" function in plms, so use "sample"
                            shape = [C, H // f, W // f]
                            samples, _ = sampler.sample(S=steps,
                                                            conditioning=c,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=callback)
                        else:
                            raise Exception(f"Sampler {samplern} not recognised.")

                    if return_latent:
                        results.append(samples.clone())

                    x_samples = model.decode_first_stage(samples)
                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        if opt.pre_gobig_gfpgan:
                            image = FACE_RESTORATION(image, bg_upsampling, upscale).astype(np.uint8)
                            image = Image.fromarray(image)

                            opt.W = opt.W * upscale
                            opt.H = opt.H * upscale
                        if opt.gobig:

                            image = go_big(image, prompts, opt.g_passes, opt.overlap, opt.gobigsteps, opt.gobigscale, opt.gobigstrength, outdir, opt.W, opt.H)

                        if GFPGAN:
                            image = FACE_RESTORATION(image, bg_upsampling, upscale).astype(np.uint8)
                            image = Image.fromarray(image)
                        else:
                            image = image

                        results.append(image)


    print(f'image: {image}')

    print(f'results: {results}')
    yield results

#Variations by Justinpinkey

def variations(input_im, outdir, var_samples, var_plms, v_cfg_scale, v_steps, v_W, v_H, v_ddim_eta, v_GFPGAN, v_bg_upsampling, v_upscale):
    #im_path="data/example_conditioning/superresolution/sample_0.jpg",
    ckpt_var="/gdrive/MyDrive/sd-clip-vit-l14-img-embed_ema_only.ckpt"
    config_var="stable-diffusion-gradio-anim-opt/configs/stable-diffusion/sd-image-condition-finetune.yaml"
    outpath=outdir
    scale=v_cfg_scale
    h=v_H
    w=v_W
    n_samples=var_samples
    precision="autocast"
    if var_plms == True:
        plms=True
    ddim_steps=v_steps
    ddim_eta=v_ddim_eta
    device_idx=0


    device = 'cuda'

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im*2-1
    #input_im = load_im(im_path).to(device)



    if plms:
        sampler = PLMSSampler(model_var)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model_var)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    paths = list()
    x_samples_ddim = sample_model(input_im, model_var, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta)
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        if v_GFPGAN:
          img = FACE_RESTORATION(img, v_bg_upsampling, v_upscale).astype(np.uint8)
          img = Image.fromarray(img)
        else:
          img = img
        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
        paths.append(f"{sample_path}/{base_count:05}.png")
        base_count += 1
    del x_samples_ddim
    del sampler
    torch_gc()
    return paths

def sample_model(input_im, model_var, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta):
    model_var.to("cuda")
    precision_scope = autocast if precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope('cuda'):
            with model_var.ema_scope():
                c = model_var.get_learned_conditioning(input_im).tile(n_samples,1,1)

                if scale != 1.0:
                    uc = torch.zeros_like(c)
                else:
                    uc = None

                shape = [4, h // 8, w // 8]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 x_T=None)

                x_samples_ddim = model_var.decode_first_stage(samples_ddim)
                img = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                del x_samples_ddim
                del samples_ddim
                del c
                mem = torch.cuda.memory_allocated()/1e6
                model_var.to('cpu')
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)
                return img


#Batch Prompts by Deforum

def batch_dict(b_prompts, b_name, b_outdir, b_GFPGAN, b_bg_upsampling,
                b_upscale, b_W, b_H, b_steps, b_scale, b_seed_behavior,
                b_seed, b_sampler, b_save_grid, b_save_settings,
                b_save_samples, b_n_batch, b_n_samples, b_ddim_eta,
                b_use_init, b_init_image, b_strength, b_make_grid):
                return locals()

def run_batch(b_prompts, b_name, b_outdir, b_GFPGAN, b_bg_upsampling,
              b_upscale, b_W, b_H, b_steps, b_scale, b_seed_behavior,
              b_seed, b_sampler, b_save_grid, b_save_settings, b_save_samples,
              b_n_batch, b_n_samples, b_ddim_eta, b_use_init, b_init_image,
              b_strength, b_make_grid, b_init_img_array, b_use_mask,
              b_mask_file, b_invert_mask, b_mask_brightness_adjust, b_mask_contrast_adjust,
              b_gobig, b_passes, b_overlap, b_detail_steps, b_detail_scale, b_g_strength, b_gobigsampler, b_pregobig):
        timestring = time.strftime('%Y%m%d%H%M%S')


        model.to('cuda')

        #b_prompts = prompts

        b_prompts = list(b_prompts.split("\n"))
        g_outdir = f'{b_outdir}/_grid_images'
        # create output folder for the batch
        os.makedirs(b_outdir, exist_ok=True)

        index = 0
        all_images = []
        b_outputs = []

        # function for init image batching
        init_array = []
        #Defaults needed by def generate()
        b_init_latent = None
        b_init_sample = None
        b_init_c = None
        dynamic_threshold = None
        static_threshold = None
        precision = 'autocast'
        #fixed_code = True
        C = 4
        f = 8
        b_seed_list = []
        b_seed_list.append(b_seed)
        if b_gobig:
            if b_pregobig:
                opt.pre_gobig_gfpgan = True
            else:
                opt.pre_gobig_gfpgan = False
            opt.gobig = True
            opt.g_passes = b_passes
            opt.overlap = b_overlap
            opt.gobigsteps = b_detail_steps
            opt.gobigscale = b_detail_scale
            opt.gobigstrength = b_g_strength
            opt.gobigsampler = b_gobigsampler
        else:
            opt.gobig = False
            opt.pre_gobig_gfpgan = False
        if b_init_img_array != None:
            initdir = f'{b_outdir}/init'
            os.makedirs(initdir, exist_ok=True)
            r = random.randint(10000, 99999)
            b_init_image = f'{b_outdir}/init/init_{b_seed}_{timestring}.png'
            b_mask_file = f'{b_outdir}/init/mask_{b_seed}_{timestring}.png'
            b_init_img_array['image'].save(os.path.join(b_outdir, b_init_image))
            b_init_img_array['mask'].save(os.path.join(b_outdir, b_mask_file))
            b_use_mask = True
            b_use_init = True
        else:
            b_mask_file = ""
            b_mask_contrast_adjust = 1.0
            b_mask_brightness_adjust = 1.0
            b_invert_mask = False

        if b_use_init:
            if b_init_image == "":
                raise FileNotFoundError("No path was given for init_image")
            if b_init_image.startswith('http://') or b_init_image.startswith('https://'):
                init_array.append(b_init_image)
            elif not os.path.isfile(b_init_image):
                if b_init_image[-1] != "/": # avoids path error by adding / to end if not there
                    b_init_image += "/"
                for image in sorted(os.listdir(b_init_image)): # iterates dir and appends images to init_array
                    if image.split(".")[-1] in ("png", "jpg", "jpeg"):
                        init_array.append(b_init_image + image)
            else:
                init_array.append(b_init_image)
        else:
            init_array = [""]

        # when doing large batches don't flood browser with images
        #clear_between_batches = b_n_batch >= 32

        for iprompt, prompt in enumerate(b_prompts):
            b_prompt = b_prompts[iprompt]
            b_sanitized = sanitize(b_prompt)
            b_sanitized = f'{b_sanitized[:128]}_{b_seed}_{timestring}'


            for batch_index in range(b_n_batch):
                #if clear_between_batches:
                #    display.clear_output(wait=True)
                #print(f"Batch {batch_index+1} of {b_n_batch}")

                for image in init_array: # iterates the init images
                    u_seed = seed_everything(b_seed)
                    b_init_image = image
                    print(f'Seed:{u_seed}')
                    if b_sampler != "diffusers":

                        results = generate(b_prompts[iprompt], b_name, b_outdir,
                                           b_GFPGAN, b_bg_upsampling, b_upscale,
                                           b_W, b_H, b_steps, b_scale, u_seed,
                                           b_sampler, b_n_batch, b_n_samples, b_ddim_eta,
                                           b_use_init, b_init_image,
                                           b_init_sample, b_strength,
                                           b_use_mask, b_mask_file,
                                           b_mask_contrast_adjust,
                                           b_mask_brightness_adjust, b_invert_mask,
                                           dynamic_threshold=None, static_threshold=None, C=4, f=8, init_c=None)
                    else:
                        images = generate_diff(b_prompts[iprompt], b_n_samples, b_n_batch, b_steps, b_scale)
                        results = images[0]


                    for image in results:
                        #all_images.append(results[image])
                        if b_make_grid:
                            all_images.append(T.functional.pil_to_tensor(image))
                        if b_save_samples:

                            b_filename = (f"{b_sanitized}_{u_seed}_{index:05}.png")
                            b_fpath = (f"{b_outdir}/{b_filename}")
                            if type(image) == list:
                              image = image[0]
                            print(f'image before saving: {image}')
                            print(f'image type before saving: {type(image)}')

                            image.save(os.path.join(b_outdir, b_filename))
                            b_outputs.append(b_fpath)
                            yield gr.update(value=b_outputs), gr.update(visible=False)

                    if b_seed_behavior != 'fixed':
                        b_seed = next_seed(b_seed, b_seed_behavior)
                        b_seed_list.append(b_seed)

                        #if b_display_samples:
                        #    display.display(image)
                        index += 1

        print(f"Filepath List: {b_outputs}")

        if b_make_grid:
            b_grid_rows = 2
            grid = mkgrid(all_images, nrow=int(len(all_images)/b_grid_rows))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid_filename = f"{b_sanitized}_{b_name}_{iprompt:05d}_grid_{random.randint(10000, 99999)}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            grid_image.save(os.path.join(b_outdir, grid_filename))
            grid_path = f"{g_outdir}/{grid_filename}"
            b_outputs.append(grid_path)

        batch_path_list=os.listdir(f'{opt.outdir}/_batch_images')
        batchargs = SimpleNamespace(**batch_dict(b_prompts,
                                                b_name,
                                                b_outdir,
                                                b_GFPGAN,
                                                b_bg_upsampling,
                                                b_upscale,
                                                b_W,
                                                b_H,
                                                b_steps,
                                                b_scale,
                                                b_seed_behavior,
                                                b_seed_list,
                                                b_sampler,
                                                b_save_grid,
                                                b_save_settings,
                                                b_save_samples,
                                                b_n_batch,
                                                b_n_samples,
                                                b_ddim_eta,
                                                b_use_init,
                                                b_init_image,
                                                b_strength,
                                                b_make_grid
        ))




        if b_save_settings or b_save_samples:
            print(f"Saving to {opt.cfg_path}/batch_configs/_*")

        #save settings for the batch
        if b_save_settings:
            os.makedirs(f'{opt.cfg_path}/batch_configs', exist_ok=True)
            filename = os.path.join(opt.cfg_path, f"batch_configs/{b_name}_{b_seed}_{timestring}_settings.txt")
            with open(filename, "w+", encoding="utf-8") as f:
                json.dump(dict(batchargs.__dict__), f, ensure_ascii=False, indent=4)



        torch_gc()
        log = (f'Seeds Used:\n{b_seed_list}')
        opt.gobig = False
        print(f'image before saving: {image}')
        print(f'image type before saving: {type(image)}')

        yield gr.update(value=b_outputs), gr.Dropdown.update(visible=True, choices=batch_path_list), gr.update(value=log)

#Animation by Deforum

def anim_dict(new_k_prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, translation_z, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring, make_grid, GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode):
    return locals()

def run_anim_seq(seqlist, seqname, outdir):
    seq_list = list(seqlist.split("\n"))
    for seq in seq_list:
        path = f'{opt.cfg_path}/{seq}'
        cfgfile = open(path)
        scfg = json.load(cfgfile)
        scfg = SimpleNamespace(**scfg)
        cfgfile.close()

        #Defaults
        scfg.outdir = f'{outdir}/{seqname}_{random.randint(10000, 99999)}'
        scfg.save_grid = False
        scfg.save_settings = False
        scfg.save_samples = True
        scfg.display_samples = False
        scfg.n_batch = 1
        scfg.n_samples = 1
        scfg.use_init = False
        scfg.init_image = ""
        scfg.init_strength = 0
        scfg.timestring = ""
        scfg.resume_from_timestring = False
        scfg.resume_timestring = ""
        scfg.make_grid = False
        scfg.inPaint = None
        scfg.b_use_mask = False
        scfg.b_mask_file = ""
        scfg.invert_mask = False
        scfg.mask_brightness_adjust = 1.0
        scfg.mask_contrast_adjust = 1.0
        mp4_path = ''
        scfg.prompts = ""
        mp4_pathlist = []
        scfg.keyframes = True

        anim(scfg.animation_mode, scfg.new_k_prompts, scfg.key_frames,
                scfg.prompts, scfg.batch_name, scfg.outdir, scfg.max_frames, scfg.GFPGAN,
                scfg.bg_upsampling, scfg.upscale, scfg.W, scfg.H, scfg.steps, scfg.scale,
                scfg.angle, scfg.zoom, scfg.translation_x, scfg.translation_y, scfg.translation_z,
                scfg.rotation_3d_x, scfg.rotation_3d_y, scfg.rotation_3d_z, scfg.use_depth_warping,
                scfg.midas_weight, scfg.near_plane, scfg.far_plane, scfg.fov, scfg.padding_mode,
                scfg.sampling_mode, scfg.seed_behavior, scfg.seed, scfg.interp_spline, scfg.noise_schedule,
                scfg.strength_schedule, scfg.contrast_schedule, scfg.sampler, scfg.extract_nth_frame,
                scfg.interpolate_x_frames, scfg.border, scfg.color_coherence, scfg.previous_frame_noise,
                scfg.previous_frame_strength, scfg.video_init_path, scfg.save_grid, scfg.save_settings,
                scfg.save_samples, scfg.display_samples, scfg.n_batch, scfg.n_samples, scfg.ddim_eta,
                scfg.use_init, scfg.init_image, scfg.init_strength, scfg.timestring,
                scfg.resume_from_timestring, scfg.resume_timestring, scfg.make_grid, scfg.inPaint, scfg.b_use_mask,
                scfg.b_mask_file, scfg.invert_mask, scfg.mask_brightness_adjust, scfg.mask_contrast_adjust)
    print(img)
    print(mp4_path)
    yield gr.update(value=Image.img), gr.update(value=mp4_path)




def anim(animation_mode, animation_prompts, key_frames,
        prompts, batch_name, outdir, max_frames, GFPGAN,
        bg_upsampling, upscale, W, H, steps, scale,
        angle, zoom, translation_x, translation_y, translation_z,
        rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping,
        midas_weight, near_plane, far_plane, fov, padding_mode,
        sampling_mode, seed_behavior, seed, interp_spline, noise_schedule,
        strength_schedule, contrast_schedule, sampler, extract_nth_frame,
        interpolate_x_frames, border, color_coherence, previous_frame_noise,
        previous_frame_strength, video_init_path, save_grid, save_settings,
        save_samples, display_samples, n_batch, n_samples, ddim_eta,
        use_init, init_image, init_strength, timestring,
        resume_from_timestring, resume_timestring, make_grid, inPaint, b_use_mask,
        b_mask_file, invert_mask, mask_brightness_adjust, mask_contrast_adjust,
        use_seq, seqlist, seqname):
            opt.pre_gobig_gfpgan = False
            img = np.random.random((600, 600, 3))
            opt.outdir = outdir
            model.to('cuda')
            strength = init_strength
            #Load Default Values
            index = 0
            # function for init image batching
            init_array = []
            #Defaults needed by def generate()
            init_latent = None
            init_sample = None
            init_c = None
            mask_contrast_adjust = 1.0
            mask_brightness_adjust = 1.0
            invert_mask = False
            use_mask = False
            fixed_code = True
            precision = 'autocast'
            fixed_code = True
            C = 4
            f = 8
            dynamic_threshold = None
            static_threshold = None

            images = []
            results = []

            def load_depth_model(optimize=True):
                midas_model = DPTDepthModel(
                    path=f"/content/models/dpt_large-midas-2f21e586.pt",
                    backbone="vitl16_384",
                    non_negative=True,
                )
                normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

                midas_transform = T.Compose([
                    Resize(
                        384, 384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    normalization,
                    PrepareForNet()
                ])

                midas_model.eval()
                if optimize:
                    if device == torch.device("cuda"):
                        midas_model = midas_model.to(memory_format=torch.channels_last)
                        midas_model = midas_model.half()
                midas_model = midas_model.half()
                midas_model.to(device)
                mtransform = midas_transform

                return midas_model, midas_transform, mtransform

            def anim_frame_warp_2d(prev_img_cv2, angle_series, zoom_series, translation_x_series, translation_y_series, frame_idx):
                angle = angle_series[frame_idx]
                zoom = zoom_series[frame_idx]
                translation_x = translation_x_series[frame_idx]
                translation_y = translation_y_series[frame_idx]

                center = (W // 2, H // 2)
                trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
                rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
                trans_mat = np.vstack([trans_mat, [0,0,1]])
                rot_mat = np.vstack([rot_mat, [0,0,1]])
                xform = np.matmul(rot_mat, trans_mat)

                return cv2.warpPerspective(
                    prev_img_cv2,
                    xform,
                    (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
                    borderMode=cv2.BORDER_WRAP if border == 'wrap' else cv2.BORDER_REPLICATE
                )

            def anim_frame_warp_3d(prev_img_cv2, translation_x_series, translation_y_series, translation_z_series, rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series, frame_idx, adabins_helper, midas_model, mtransform):
                TRANSLATION_SCALE = 1.0/200.0 # matches Disco
                translate_xyz = [
                    -translation_x_series[frame_idx] * TRANSLATION_SCALE,
                    translation_y_series[frame_idx] * TRANSLATION_SCALE,
                    -translation_z_series[frame_idx] * TRANSLATION_SCALE
                ]
                rotate_xyz = [
                    math.radians(rotation_3d_x_series[frame_idx]),
                    math.radians(rotation_3d_y_series[frame_idx]),
                    math.radians(rotation_3d_z_series[frame_idx])
                ]
                rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
                result = transform_image_3d(prev_img_cv2, adabins_helper, midas_model, mtransform, rot_mat, translate_xyz, midas_weight, near_plane, far_plane, fov, sampling_mode, padding_mode)
                torch.cuda.empty_cache()
                return result

                #prom = animation_prompts
                #key = prompts



            def make_xform_2d(width, height, translation_x, translation_y, angle, scale):
                center = (width // 2, height // 2)
                trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
                rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
                trans_mat = np.vstack([trans_mat, [0,0,1]])
                rot_mat = np.vstack([rot_mat, [0,0,1]])
                return np.matmul(rot_mat, trans_mat)

            def render_input_video(animation_prompts, prompts, outdir, resume_from_timestring, resume_timestring, angle, zoom, translation_x, translation_y, translation_z, noise_schedule, contrast_schedule, extract_nth_frame, video_init_path, max_frames, use_init):
                # create a folder for the video input frames to live in
                video_in_frame_path = os.path.join(outdir, 'inputframes')
                os.makedirs(os.path.join(outdir, video_in_frame_path), exist_ok=True)

                # save the video frames from input video
                print(f"Exporting Video Frames (1 every {extract_nth_frame}) frames to {video_in_frame_path}...")
                try:
                    for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
                        f.unlink()
                except:
                    pass
                vf = r'select=not(mod(n\,'+str(extract_nth_frame)+'))'
                subprocess.run([
                    'ffmpeg', '-i', f'{video_init_path}',
                    '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2',
                    '-loglevel', 'error', '-stats',
                    os.path.join(video_in_frame_path, '%04d.jpg')
                ], stdout=subprocess.PIPE).stdout.decode('utf-8')

                # determine max frames from length of input frames
                max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])

                use_init = True
                print(f"Loading {max_frames} input frames from {video_in_frame_path} and saving video frames to {outdir}")
                render_animation(animation_prompts, prompts, seed, outdir, resume_from_timestring, resume_timestring, timestring, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, noise_schedule, contrast_schedule)

            def render_interpolation(prompt, name, outdir, GFPGAN,
                                    bg_upsampling, upscale, W, H,
                                    steps, scale, seed, samplern,
                                    n_batch, n_samples, ddim_eta,
                                    use_init, init_image, init_sample,
                                    strength, use_mask, mask_file,
                                    mask_contrast_adjust, mask_brightness_adjust,
                                    invert_mask, timestring):
                # animations use key framed prompts
                prompts = animation_prompts
                # create output folder for the batch
                os.makedirs(outdir, exist_ok=True)
                print(f"Saving animation frames to {outdir}")
                # save settings for the batch
                settings_filename = os.path.join(outdir, f"{timestring}_settings.txt")
                with open(settings_filename, "w+", encoding="utf-8") as f:
                    s = {**dict(__dict__), **dict(__dict__)}
                    json.dump(dict(anim_args.__dict__), f, ensure_ascii=False, indent=4)
                # Interpolation Settings
                n_samples = 1
                seed_behavior = 'fixed' # force fix seed at the moment bc only 1 seed is available
                prompts_c_s = [] # cache all the text embeddings

                print(f"Preparing for interpolation of the following...")

                for i, prompt in animation_prompts.items():
                  prompt = prompt

                  # sample the diffusion model
                  results = generate(prompt, name, outdir, GFPGAN,
                                    bg_upsampling, upscale, W, H,
                                    steps, scale, seed, samplern,
                                    n_batch, n_samples, ddim_eta,
                                    use_init, init_image, init_sample,
                                    strength, use_mask, mask_file,
                                    mask_contrast_adjust, mask_brightness_adjust,
                                    invert_mask, dynamic_threshold, static_threshold,
                                    C, f, init_c, return_c=True)
                  c, image = results[0], results[1]
                  prompts_c_s.append(c)

                  # display.clear_output(wait=True)
                  #display.display(image)

                  seed = next_seed(seed, seed_behavior)

                display.clear_output(wait=True)
                print(f"Interpolation start...")

                frame_idx = 0

                if interpolate_key_frames:
                  for i in range(len(prompts_c_s)-1):
                    dist_frames = list(animation_prompts.items())[i+1][0] - list(animation_prompts.items())[i][0]
                    if dist_frames <= 0:
                      print("key frames duplicated or reversed. interpolation skipped.")
                      return
                    else:
                      for j in range(dist_frames):
                        # interpolate the text embedding
                        prompt1_c = prompts_c_s[i]
                        prompt2_c = prompts_c_s[i+1]
                        init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1/dist_frames))

                        # sample the diffusion model
                        results = generate(prompt, name, outdir, GFPGAN,
                                          bg_upsampling, upscale, W, H,
                                          steps, scale, seed, samplern,
                                          n_batch, n_samples, ddim_eta,
                                          use_init, init_image, init_sample,
                                          strength, use_mask, mask_file,
                                          mask_contrast_adjust, mask_brightness_adjust,
                                          invert_mask, dynamic_threshold, static_threshold,
                                          C, f, init_c)
                        image = results[0]

                        filename = f"{timestring}_{frame_idx:05}.png"
                        image.save(os.path.join(outdir, filename))
                        frame_idx += 1

                        #display.clear_output(wait=True)
                        #display.display(image)

                        seed = next_seed(seed, seed_behavior)

                else:
                  for i in range(len(prompts_c_s)-1):
                    for j in range(interpolate_x_frames+1):
                      # interpolate the text embedding
                      prompt1_c = prompts_c_s[i]
                      prompt2_c = prompts_c_s[i+1]
                      init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1/(interpolate_x_frames+1)))

                      # sample the diffusion model
                      results = generate(prompt, name, outdir, GFPGAN,
                                        bg_upsampling, upscale, W, H,
                                        steps, scale, seed, samplern,
                                        n_batch, n_samples, ddim_eta,
                                        use_init, init_image, init_sample,
                                        strength, use_mask, mask_file,
                                        mask_contrast_adjust, mask_brightness_adjust,
                                        invert_mask, dynamic_threshold, static_threshold,
                                        C, f, init_c)
                      image = results[0]

                      filename = f"{timestring}_{frame_idx:05}.png"
                      image.save(os.path.join(outdir, filename))
                      frame_idx += 1

                      #display.clear_output(wait=True)
                      #display.display(image)

                      seed = next_seed(seed, seed_behavior)

                # generate the last prompt
                init_c = prompts_c_s[-1]

                results = generate(prompt, name, outdir, GFPGAN,
                                  bg_upsampling, upscale, W, H,
                                  steps, scale, seed, samplern,
                                  n_batch, n_samples, ddim_eta,
                                  use_init, init_image, init_sample,
                                  strength, use_mask, mask_file,
                                  mask_contrast_adjust, mask_brightness_adjust,
                                  invert_mask, dynamic_threshold, static_threshold,
                                  C, f, init_c)
                image = results[0]
                filename = f"{timestring}_{frame_idx:05}.png"
                image.save(os.path.join(outdir, filename))

                #display.clear_output(wait=True)
                #display.display(image)
                seed = next_seed(seed, seed_behavior)

                #clear init_c
                init_c = None

            #animation_prompts = dict(zip(new_key, new_prom))
            if animation_mode == '2D' or animation_mode == '3D':
                timestring = time.strftime('%Y%m%d%H%M%S')

                #animation_mode = animation_mode
                anim_dict(animation_prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, translation_z, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring, make_grid, GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode)

                anim_args = SimpleNamespace(**anim_dict(animation_prompts, animation_mode,
                                                      strength, max_frames, border, key_frames,
                                                      interp_spline, angle, zoom, translation_x,
                                                      translation_y, translation_z, color_coherence,
                                                      previous_frame_noise, previous_frame_strength,
                                                      video_init_path, extract_nth_frame, interpolate_x_frames,
                                                      batch_name, outdir, save_grid, save_settings, save_samples,
                                                      display_samples, n_samples, W, H, init_image, seed, sampler,
                                                      steps, scale, ddim_eta, seed_behavior, n_batch, use_init,
                                                      timestring, noise_schedule, strength_schedule, contrast_schedule,
                                                      resume_from_timestring, resume_timestring, make_grid,
                                                      GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y,
                                                      rotation_3d_z, use_depth_warping, midas_weight, near_plane,
                                                      far_plane, fov, padding_mode, sampling_mode))
                #Moving Sequence Animation to main anim loop to make sense

                if use_seq == True:
                    seq_list = list(seqlist.split("\n"))
                else:
                    seq_list = ['one line']

                base_outdir = outdir
                mp4_p = f'{outdir}/_mp4s'
                total_frames = 0
                for seq in seq_list:
                    if use_seq == True:
                        path = f'{opt.cfg_path}/{seq}'
                        cfgfile = open(path)
                        scfg = json.load(cfgfile)
                        scfg = SimpleNamespace(**scfg)
                        cfgfile.close()
                        prev_total = total_frames
                        total_frames = total_frames + scfg.max_frames
                        print(f'Rendering animation sequence from frame: {prev_total} to {total_frames}')

                        animation_mode = scfg.animation_mode
                        animation_prompts = scfg.new_k_prompts
                        key_frames = scfg.key_frames
                        prompts = ""
                        batch_name = scfg.batch_name
                        outdir = f'{base_outdir}/_sequences/{seqname}_{timestring}'
                        max_frames = scfg.max_frames
                        #GFPGAN = scfg.GFPGAN
                        #bg_upsampling = scfg.bg_upsampling
                        #upscale = scfg.upscale
                        #W = scfg.W
                        #H = scfg.H
                        steps = scfg.steps
                        scale = scfg.scale
                        angle = scfg.angle
                        zoom = scfg.zoom
                        translation_x = scfg.translation_x
                        translation_y = scfg.translation_y
                        translation_z = scfg.translation_z
                        rotation_3d_x = scfg.rotation_3d_x
                        rotation_3d_y = scfg.rotation_3d_y
                        rotation_3d_z = scfg.rotation_3d_z
                        use_depth_warping = scfg.use_depth_warping
                        midas_weight = scfg.midas_weight
                        near_plane = scfg.near_plane
                        far_plane = scfg.far_plane
                        fov = scfg.fov
                        padding_mode = scfg.padding_mode
                        sampling_mode = scfg.sampling_mode
                        seed_behavior = scfg.seed_behavior
                        seed = scfg.seed
                        interp_spline = scfg.interp_spline
                        noise_schedule = scfg.noise_schedule
                        strength_schedule = scfg.strength_schedule
                        contrast_schedule = scfg.contrast_schedule
                        sampler = scfg.sampler
                        extract_nth_frame = scfg.extract_nth_frame
                        interpolate_x_frames = scfg.interpolate_x_frames
                        border = scfg.border
                        color_coherence = scfg.color_coherence
                        previous_frame_noise = scfg.previous_frame_noise
                        previous_frame_strength = scfg.previous_frame_strength
                        video_init_path = scfg.video_init_path
                    outputs = []
                    if animation_mode == 'None':
                      max_frames = 1
                    strength = max(0.0, min(1.0, strength))
                    returns = {}
                    mask_file = ""
                    if seed == -1:
                        seed = random.randint(0, 2**32)
                        anim_args.seed = seed
                    os.makedirs(mp4_p, exist_ok=True)
                    if use_seq == True:
                        outdir =f'{outdir}/_anim_stills/{seqname}_{timestring}'
                    else:
                        outdir = f'{outdir}/_anim_stills/{batch_name}_{seed}_{timestring}'
                    if animation_mode == 'Video Input':
                        use_init = True
                    if not use_init:
                        init_image = None
                        strength = 0
                    if sampler == 'plms' and (use_init or animation_mode != 'None'):
                        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
                        sampler = 'klms'
                    if sampler != 'ddim':
                        ddim_eta = 0
                    if animation_mode == '2D' or animation_mode == '3D':
                        new_key = []
                        new_prom = []
                        #new_prom = list(prom.split("\n"))
                        #new_key = list(key.split("\n"))
                        for data in animation_prompts:
                          k, p = data
                          if type(k) != 'int' and k != '':
                            k = int(k)
                          if k != '':
                            new_key.append(k)
                          if p != '':
                            new_prom.append(p)
                        prompts = dict(zip(new_key, new_prom))
                        #prompts = animation_prompts
                        # animations use key framed prompts
                        #prompts = animation_prompts
                        angle_series, zoom_series, translation_x_series, translation_y_series, translation_z_series, rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series, noise_schedule_series, strength_schedule_series, contrast_schedule_series = DeformAnimKeys(angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, noise_schedule, strength_schedule, contrast_schedule, max_frames)
                        #print(f'Keys: {keys}')
                        # resume animation
                        start_frame = 0
                        if resume_from_timestring:
                            for tmp in os.listdir(outdir):
                                if tmp.split("_")[0] == resume_timestring:
                                    start_frame += 1
                            start_frame = start_frame - 1

                        # create output folder for the batch
                        os.makedirs(outdir, exist_ok=True)
                        print(f"Saving animation frames to {outdir}")

                        #save settings for the batch
                        if use_seq != True:
                            settings_filename = os.path.join(opt.cfg_path, f"{batch_name}_{timestring}_settings.txt")
                            with open(settings_filename, "w+", encoding="utf-8") as f:
                                  json.dump(dict(anim_args.__dict__), f, ensure_ascii=False, indent=4)

                        # resume from timestring
                        if resume_from_timestring:
                            timestring = resume_timestring


                        #promptList = list(animation_prompts.split("\n"))
                        #promptList = animation_prompts
                        prompt_series = pd.Series([np.nan for a in range(max_frames)])
                        for i in prompts:
                            prompt_series[i] = prompts[i]

                        prompt_series = prompt_series.ffill().bfill()

                        # check for video inits
                        using_vid_init = animation_mode == 'Video Input'

                        # load depth model for 3D
                        if animation_mode == '3D' and use_depth_warping:
                            download_depth_models()
                            adabins_helper = InferenceHelper(dataset='nyu', device=device)
                            midas_model, midas_transform, mtransform = load_depth_model()
                            #load_depth_model()
                        else:
                            adabins_helper, midas_model, midas_transform = None, None, None
                        opt.should_stop = False
                        n_samples = 1
                        prev_sample = None
                        color_match_sample = None

                        for frame_idx in range(start_frame,max_frames):
                            print(f"Rendering animation frame {frame_idx} of {max_frames}")
                            noise = noise_schedule_series[frame_idx]
                            strength = strength_schedule_series[frame_idx]
                            contrast = contrast_schedule_series[frame_idx]
                            if frame_idx == 0:
                                strength = 0
                            # resume animation
                            if resume_from_timestring:
                                path = os.path.join(outdir,f"{timestring}_{frame_idx-1:05}.png")
                                img = cv2.imread(path)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                prev_sample = sample_from_cv2(img)

                            # apply transforms to previous frame
                            if prev_sample is not None:

                                if animation_mode == '2D':
                                    prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), angle_series, zoom_series, translation_x_series, translation_y_series, frame_idx)
                                else: # '3D'
                                    prev_img = anim_frame_warp_3d(sample_to_cv2(prev_sample), translation_x_series, translation_y_series, translation_z_series, rotation_3d_x_series, rotation_3d_y_series, rotation_3d_z_series, frame_idx, adabins_helper, midas_model, mtransform)

                                # apply color matching
                                if color_coherence != 'None':
                                    if color_match_sample is None:
                                        color_match_sample = prev_img.copy()
                                    else:
                                        prev_img = maintain_colors(prev_img, color_match_sample, color_coherence)

                                # apply scaling
                                contrast_sample = prev_img * contrast
                                # apply frame noising
                                noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

                                # use transformed previous frame as init for current
                                use_init = True
                                #init_sample = noised_sample.half().to(device)
                                if half_precision:
                                    init_sample = noised_sample.half().to(device)
                                else:
                                    init_sample = noised_sample.to(device)
                                strength = max(0.0, min(1.0, strength))

                            # grab prompt for current frame
                            prompt = prompt_series[frame_idx]
                            print(f"{prompt} {seed}")

                            # grab init image for current frame
                            if using_vid_init:
                                init_frame = os.path.join(outdir, 'inputframes', f"{frame_idx+1:04}.jpg")
                                print(f"Using video init frame {init_frame}")
                                init_image = init_frame
                            # sample the diffusion model
                            results = generate(prompt, batch_name, outdir, GFPGAN, bg_upsampling, upscale, W, H, steps, scale, seed, sampler, n_batch, n_samples, ddim_eta, use_init, init_image, init_sample, strength, use_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, invert_mask, dynamic_threshold, static_threshold, C, f, init_c, return_latent=False, return_sample=True)





                            sample, image = results[0], results[1]

                            if use_seq == True:
                                filename = f"{timestring}_{prev_total + frame_idx:05}.png"
                            else:
                                filename = f"{timestring}_{frame_idx:05}.png"
                            image.save(os.path.join(outdir, filename))
                            if not using_vid_init:
                                prev_sample = sample

                            seed = next_seed(seed, seed_behavior)


                            img = image
                            mp4_path = ''
                            mp4_pathlist = []
                            yield gr.update(value=img, visible=True), gr.update(visible=False)




                if use_seq == True:
                  max_frames = total_frames
                  batch_name = seqname
                  seed = "sequence"


                mp4_path = makevideo(outdir, mp4_p, batch_name, seed, timestring, max_frames)
                mp4_pathlist=os.listdir(f'{opt.outdir}/_mp4s')


                torch_gc()
                if mp4_path == "":
                    yield gr.update(value=img), gr.update(visible=False)
                else:
                    yield gr.update(visible=False), gr.update(value=mp4_path, visible=True)


                #return mp4_path, gr.Dropdown.update(choices=mp4_pathlist)
            elif animation_mode == 'Video Input':
                render_input_video(animation_prompts, prompts, outdir,
                                resume_from_timestring, resume_timestring,
                                angle, zoom, translation_x, translation_y,
                                translation_z, noise_schedule, contrast_schedule,
                                outdir, extract_nth_frame, video_init_path,
                                max_frames, use_init)
                mp4_path = makevideo(outdir, mp4_p, batch_name, seed, timestring, max_frames)
                torch_gc()
                return mp4_path
            elif animation_mode == 'Interpolation':
                render_interpolation(prompt, name, outdir, GFPGAN,
                                  bg_upsampling, upscale, W, H,
                                  steps, scale, seed, samplern,
                                  n_batch, n_samples, ddim_eta,
                                  use_init, init_image, init_sample,
                                  strength, use_mask, mask_file,
                                  mask_contrast_adjust, mask_brightness_adjust,
                                  invert_mask, timestring)
                mp4_path = makevideo(outdir, mp4_p, batch_name, seed, timestring, max_frames)

                torch_gc()

                return mp4_path
            else:
                print('No Animation Mode Selected')
                torch_gc()
                return outputs

#UI

torch_gc()
inPaint=None
cfg_snapshots = []
demo = gr.Blocks('SD 1.4 - Anim 0.5')



soup_help1 = """
  ##                     Adjective Types\n
  * _adj-architecture_ - A list of architectural adjectives and styles
  * _adj-beauty_ - A list of beauty adjectives for people (maybe things?)
  * _adj-general_ - A list of general adjectives for people/things.
  * _adj-horror_ - A list of horror adjectives
  ##                        Art Types
  * _artist_ - A comprehensive list of artists by MisterRuffian (Discord Misterruffian#2891)
  * _color_ - A comprehensive list of colors
  * _portrait-type_ - A list of common portrait types/poses
  * _style_ - A list of art styles and mediums
  ##              Computer Graphics Types
  * _3d-terms_ - A list of 3D graphics terminology
  * _color-palette_ - A list of computer and video game console color palettes
  * _hd_ - A list of high definition resolution terms
  ##            Miscellaneous Types
  * _details_ - A list of detail descriptors
  * _site_ - A list of websites to query
  * _gen-modififer_ - A list of general modifiers adopted from Weird Wonderful AI Art
  * _neg-weight_ - A lsit of negative weight ideas
  * _punk_ - A list of punk modifier (eg. cyberpunk)
  * _pop-culture_ - A list of popular culture movies, shows, etc
  * _pop-location_ - A list of popular tourist locations
  * _fantasy-setting_ - A list of fantasy location settings
  * _fantasy-creature_ - A list of fantasy creatures"""

soup_help2 ="""
  ##                Noun Types
  * _noun-beauty_ - A list of beauty related nouns
  * _noun-emote_ - A list of emotions and expressions
  * _noun-fantasy_ - A list of fantasy nouns
  * _noun-general_ - A list of general nouns
  * _noun-horror_ - A list of horror nouns
  ##People Types
  * _bodyshape_ - A list of body shapes
  * _celeb_ - A list of celebrities
  * _eyecolor_ - A list of eye colors
  * _hair_ - A list of hair types
  * _nationality_ - A list of nationalities
  * _occputation_ A list of occupation types
  * _skin-color_ - A list of skin tones
  * _identity-young_ A list of young identifiers
  * _identity-adult_ A list of adult identifiers
  * _identity_ A list of general identifiers
  ##Photography / Image / Film Types
  * _aspect-ratio_ - A list of common aspect ratios
  * _cameras_ - A list of camera models (including manufactuerer)
  * _camera-manu_ - A list of camera manufacturers
  * _f-stop_ - A list of camera aperture f-stop
  * _focal-length_ - A list of focal length ranges
  * _photo-term_ - A list of photography terms relating to photos
  """

prompt_placeholder = "First Prompt\nSecond Prompt\nThird Prompt\n\nMake sure your prompts are divided by having them in separate lines."
keyframe_placeholder = "0\n25\n50\n\nMake sure you only have numbers here, and they are all in new lines, without empty lines."
list1 = []
os.makedirs(f'{opt.outdir}/_mp4s', exist_ok=True)
mp4_pathlist=os.listdir(f'{opt.outdir}/_mp4s')

os.makedirs(f'{opt.outdir}/_batch_images', exist_ok=True)
batch_pathlist=os.listdir(f'{opt.outdir}/_batch_images')

def view_video(mp4_path_to_view):
  mp4_pathlist=os.listdir(f'{opt.outdir}/_mp4s')
  return gr.Video.update(value=f'{opt.outdir}/_mp4s/{mp4_path_to_view}'), gr.Dropdown.update(choices=mp4_pathlist)

def view_batch_file(inputfile):
  print(inputfile)
  print(type(inputfile))
  print(f'{opt.outdir}/_batch_images/{inputfile}')
  return gr.update(value=[f'{opt.outdir}/_batch_images/{inputfile}'])
def view_editor_file(inputfile):
  print(inputfile)
  print(type(inputfile))
  path = f'{opt.outdir}/_batch_images/{inputfile}'
  image = Image.open(path).convert('RGB')
  return gr.update(value=image)




def refresh_video_list():
  mp4_pathlist=os.listdir(f'{opt.outdir}/_mp4s')
  return gr.Dropdown.update(choices=mp4_pathlist)

def refresh(choice):

  return gr.update(value=choice)


if opt.cfg_path == "" or opt.cfg_path == None:
  opt.cfg_path = "/gdrive/MyDrive/sd_anim_configs"
os.makedirs(opt.cfg_path, exist_ok=True)
for files in os.listdir(opt.cfg_path):
    if files.endswith(".txt"):
        list1.append(files)
list2 = list1

def stop():
    opt.should_stop = True
    print('Generation should stop now..')
    return opt.should_stop

def test_update(scale):
  for _ in range(10):
        time.sleep(1)
        image = np.random.random((600, 600, 3))
        print(type(image))
        yield image
  yield image

print(f'I-------------------------------------------I')
print(f'I                                           I')
print(f'I        Stable Diffusion 1.4 Loaded        I')
print(f'I              Anim GUI by mix              I')
print(f'I                                           I')
print(f'I              features loaded:             I')
print(f'I                                           I')
print(f'I      -  2D/3D Animation Sequencer         I')
print(f'I      -  GFPGAN Upscaler                   I')
print(f'I      -  Batch Prompts                     I')
print(f'I      -  InPaint                           I')
if opt.load_p2p:
    print(f'I      -  Prompt to Prompt Image Editor     I')
if not opt.no_var:
    print(f'I      -  Variations model                  I')
if not opt.embeds:
    print(f'I      -  Concept Embeddings                I')

print(f'I                                           I')
print(f'I-------------------------------------------I')


with demo:
    with gr.Tabs():
        with gr.TabItem('Animation'):

            with gr.Row():
                with gr.Column(scale=3):
                    img = gr.Image(visible=False)
                    mp4_path_to_view = gr.Dropdown(label='videos', choices=mp4_pathlist)
                    mp4_paths = gr.Video(label='Generated Video')
                    new_k_prompts = gr.Dataframe(headers=["keyframe", "prompt"], datatype=("number", "str"), col_count=(2, "fixed"), type='array')

                    animation_prompts = gr.Textbox(label='Prompts - divided by enter',
                                                    placeholder=prompt_placeholder,
                                                    lines=5, interactive=True, visible=False)#animation_prompts
                    key_frames = gr.Checkbox(label='KeyFrames',
                                            value=True,
                                            visible=False, interactive=True)#key_frames
                    prompts = gr.Textbox(label='Keyframes - numbers divided by enter',
                                        placeholder=keyframe_placeholder,
                                        lines=5,
                                        interactive=True, visible=False)#prompts
                    with gr.Accordion(label = 'Render Settings', open=False):
                        batch_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim', interactive=True)#batch_name
                        outdir = gr.Textbox(label='Output Dir',  placeholder='/content', lines=1, value=opt.outdir, interactive=True)#outdir
                        sampler = gr.Radio(label='Sampler',
                                          choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                          value='klms', interactive=True)#sampler
                        max_frames = gr.Slider(minimum=1, maximum=2500, step=1, label='Frames to render', value=20)#max_frames
                        steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=20, interactive=True)#steps
                        scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11, interactive=True)#scale
                        ddim_eta = gr.Slider(minimum=0, maximum=1.0, step=0.1, label='DDIM ETA', value=0.0)#ddim_eta
                        with gr.Row():
                            W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                            H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height
                        with gr.Row():

                            GFPGAN = gr.Checkbox(label='GFPGAN, Upscaler', value=False)
                            bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                            upscale = gr.Slider(minimum=1, maximum=8, step=1, label='Upscaler, 1 to turn off', value=1, interactive=True)


                        n_batch = gr.Slider(minimum=1, maximum=50, step=1, label='Number of Batches', value=1, visible=False)#n_batch
                        n_samples = gr.Slider(minimum=1, maximum=4, step=1, label='Samples (keep on 1)', value=1, visible=False)#n_samples
                        resume_timestring = gr.Textbox(label='Resume from:',  placeholder='20220829210106', lines=1, value='', interactive = True)
                        timestring = gr.Textbox(label='Timestring',  placeholder='timestring', lines=1, value='')#timestring

                    anim_btn = gr.Button('Generate')
                    with gr.Row():
                      with gr.Column():
                        with gr.Row():
                            save_cfg_btn = gr.Button('save config snapshot')
                            load_cfg_btn = gr.Button('load config snapshot')
                      cfg_snapshots = gr.Dropdown(label = 'config snapshots (loading is WIP)', choices = list1, interactive=True)

                        #output = gr.Text()
                with gr.Column(scale=4):
                    with gr.TabItem('Animation'):
                        with gr.Accordion(label = 'Animation Settings', open=False):
                            with gr.Row():
                                border = gr.Dropdown(label='Border', choices=['wrap', 'replicate'], value='wrap')#border
                                animation_mode = gr.Dropdown(label='Animation Mode',
                                                                choices=['None', '2D', '3D', 'Video Input', 'Interpolation'],
                                                                value='3D')#animation_mode
                            with gr.Row():
                                seed_behavior = gr.Dropdown(label='Seed Behavior', choices=['iter', 'fixed', 'random'], value='iter')#seed_behavior
                                seed = gr.Number(label='Seed',  placeholder='SEED HERE', value='-1')#seed
                            with gr.Row():
                                interp_spline = gr.Dropdown(label='Spline Interpolation', choices=['Linear', 'Quadratic', 'Cubic'], value='Linear')#interp_spline
                                color_coherence = gr.Dropdown(label='Color Coherence', choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'], value='Match Frame 0 RGB')#color_coherence
                            strength_schedule = gr.Textbox(label='Strength_Schedule',  placeholder='0:(0)', lines=1, value='0:(0.65)')#strength_schedule
                            with gr.Row():
                                contrast_schedule = gr.Textbox(label='Contrast Schedule',  placeholder='0:(0)', lines=1, value='0:(1.0)')#contrast_schedule
                                noise_schedule = gr.Textbox(label='Noise Schedule',  placeholder='0:(0)', lines=1, value='0:(0.02)')#noise_schedule

                        with gr.Accordion(label = 'Movements', open=False) as movement_settings:
                            gr.Markdown('Keyframe Builder:')
                            with gr.Row():
                                kb_frame = gr.Textbox(label = 'Frame', interactive = True)
                                kb_value = gr.Textbox(label = 'Value', interactive = True)
                            kb_btn = gr.Button('Add to Keyframe sequence below')
                            kb_string = gr.Textbox(label = 'Keyframe sequence', interactive = True)

                            with gr.Row():

                                with gr.Column():
                                    angle = gr.Textbox(label='2D Angles (rotation)',  placeholder='0:(0)', lines=1, value='0:(0)')#angle
                                    zoom = gr.Textbox(label='2D Zoom',  placeholder='0: (1.04)', lines=1, value='0:(1.0)')#zoom
                                    translation_x = gr.Textbox(label='2D/3D Translation X (+ is Camera Left, large values [1 - 50])',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_x
                                    translation_y = gr.Textbox(label='2D/3D Translation Y + = Up',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_y
                                    translation_z = gr.Textbox(label='3D Translation Z + = Forward',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#translation_y
                                with gr.Column():
                                    rotation_3d_x = gr.Textbox(label='3D Rotation X : Tilt (+ is Up)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_x
                                    rotation_3d_y = gr.Textbox(label='3D Rotation Y : Pan (+ is Right)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_y
                                    rotation_3d_z = gr.Textbox(label='3D Rotation Z : Push In(+ is Clockwise)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_z
                                    midas_weight = gr.Slider(minimum=0, maximum=5, step=0.1, label='Midas Weight', value=0.3, visible=True)#midas_weight
                        with gr.Accordion('3D Settings', open=False):
                            use_depth_warping = gr.Checkbox(label='Depth Warping', value=True, visible=True)#use_depth_warping
                            near_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Near Plane', value=200, visible=True)#near_plane
                            far_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Far Plane', value=1000, visible=True)#far_plane
                            fov = gr.Slider(minimum=0, maximum=360, step=1, label='FOV', value=40, visible=True)#fov
                            with gr.Row():
                                padding_mode = gr.Dropdown(label='Padding Mode', choices=['border', 'reflection', 'zeros'], value='border', visible=True)#padding_mode
                                sampling_mode = gr.Dropdown(label='Sampling Mode', choices=['bicubic', 'bilinear', 'nearest'], value='bicubic', visible=True)#sampling_mode

                        with gr.Accordion(label = 'Other Settings', open=False):
                            with gr.Row():
                                save_grid = gr.Checkbox(label='Save Grid', value=False, visible=True)#save_grid
                                make_grid = gr.Checkbox(label='Make Grid', value=False, visible=False)#make_grid
                            with gr.Row():
                                save_samples = gr.Checkbox(label='Save Samples', value=True, visible=False)#save_samples
                                display_samples = gr.Checkbox(label='Display Samples', value=False, visible=False)#display_samples
                            with gr.Row():
                                save_settings = gr.Checkbox(label='Save Settings', value=True, visible=True)#save_settings
                                resume_from_timestring = gr.Checkbox(label='Resume from Timestring', value=False, visible=True)#resume_from_timestring
                        with gr.Accordion(label = 'Animation Sequncer', open=False) as sequencer_settings:
                            with gr.Row():
                                seqname = gr.Textbox(label='Sequence Name', interactive=True)
                                use_seq = gr.Checkbox(label='Use Sequence', interactive=True)
                            cfg_seq_snapshots = gr.Dropdown(label = 'select snapshot to add', choices = list2, interactive=True)
                            add_cfg_btn = gr.Button('add config snapshot to sequence')
                            sequence = gr.Textbox(label='sequence', lines = 10, interactive=True)



                    with gr.TabItem('Video / Init Video / Interpolation settings'):
                      with gr.Row():
                          extract_nth_frame = gr.Slider(minimum=1, maximum=100, step=1, label='Extract n-th frame', value=1)#extract_nth_frame
                          interpolate_x_frames = gr.Slider(minimum=1, maximum=25, step=1, label='Interpolate n frames', value=4)#interpolate_x_frames
                      with gr.Row():
                          previous_frame_noise = gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Noise', value=0.02)#previous_frame_noise
                          previous_frame_strength = gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Strength', value=0.0)#previous_frame_strength
                      use_init = gr.Checkbox(label='Use Init', value=False, visible=True, interactive=True)#use_init
                      init_image = gr.Textbox(label='Init Image link',  placeholder='https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg', lines=1, interactive=True)#init_image
                      video_init_path = gr.Textbox(label='Video init path',  placeholder='/content/video_in.mp4', lines=1)#video_init_path
                      strength = gr.Slider(minimum=0, maximum=1, step=0.1, label='Init Image Strength', value=0.0)#strength





        with gr.TabItem('Batch Prompts'):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        stop_batch_btn = gr.Button('stop')
                        batch_path_to_view = gr.Dropdown(label='Images', choices=batch_pathlist)
                    b_init_img_array = gr.Image(visible=False)

                    b_sampler = gr.Radio(label='Sampler',
                                        choices=['diffusers','klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                        value='klms',
                                        interactive=True)#sampler
                    b_prompts = gr.Textbox(label='Prompts',
                                                    placeholder='a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation',
                                                    lines=5)#animation_prompts
                    b_seed_behavior = gr.Dropdown(label='Seed Behavior', choices=['iter', 'fixed', 'random'], value='iter', interactive=True)#seed_behavior
                    b_seed = gr.Number(label='Seed',  placeholder='SEED HERE', value='-1', interactive=True)#seed
                    b_save_settings = gr.Checkbox(label='Save Settings', value=True, visible=True)#save_settings
                    b_save_samples = gr.Checkbox(label='Save Samples', value=True, visible=True)#save_samples
                    b_n_batch = gr.Slider(minimum=1, maximum=100, step=1, label='Number of Batches', value=1, visible=True)#n_batch
                    b_n_samples = gr.Slider(minimum=1, maximum=4, step=1, label='Samples (keep on 1)', value=1)#n_samples
                    b_ddim_eta = gr.Slider(minimum=0, maximum=1.0, step=0.1, label='DDIM ETA', value=0.0)#ddim_eta
                    b_use_init = gr.Checkbox(label='Use Init', value=False, visible=True)#use_init
                    b_init_image = gr.Textbox(label='Init Image link',  placeholder='https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg', lines=1)#init_image
                    b_strength = gr.Slider(minimum=0, maximum=1, step=0.1, label='Init Image Strength', value=0.0, interactive=True)#strength
                    b_make_grid = gr.Checkbox(label='Make Grid', value=False, visible=True)#make_grid
                    b_use_mask = gr.Checkbox(label='Use Mask', value=False, visible=True)
                    b_save_grid = gr.Checkbox(label='Save Grid', value=False, visible=True)
                    b_mask_file = gr.Textbox(label='Mask File', value='', visible=True) #
                with gr.Column():
                    with gr.Accordion('Upscalers'):
                        with gr.Row():
                            b_pregobig = gr.Checkbox(label = 'Pre-GoBig GFPGAN', value=False)
                            b_gobig = gr.Checkbox(label = 'GoBig', value=False)
                            b_GFPGAN = gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False)
                            b_bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                        b_gobigsampler = gr.Radio(label='Sampler',
                                            choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                            value='klms',
                                            interactive=True)#sampler

                        b_passes = gr.Slider(minimum=1, maximum=100, step=1, label='GoBig Passes', value=1, visible=True)
                        b_overlap = gr.Slider(minimum=1, maximum=512, step=1, label='GoBig Overlap', value=128, visible=True)
                        b_detail_steps = gr.Slider(minimum=1, maximum=250, step=1, label='GoBig Steps', value=50, visible=True)
                        b_detail_scale = gr.Slider(minimum=0, maximum=100, step=1, label='GoBig Detail Scale', value=10, visible=True)
                        b_g_strength = gr.Slider(minimum=0, maximum=1, step=0.1, label='GoBig Strength', value=0.3, visible=True)
                        b_upscale = gr.Slider(minimum=1, maximum=8, step=1, label='GFPGAN Scale (1x = Off)', value=1, interactive=True)

                    batch_outputs = gr.Gallery()
                    b_log = gr.Textbox(lines=4)
                    b_W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                    b_H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height
                    b_steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100, interactive=True)#steps
                    b_scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11, interactive=True)#scale
                    b_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim', interactive=True)#batch_name
                    b_outdir = gr.Textbox(label='Output Dir',  placeholder='/content/', lines=1, value=f'{opt.outdir}/_batch_images', interactive=True)#outdir
                    batch_btn = gr.Button('Generate')
                    with gr.Row():
                      b_mask_brightness_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Brightness', value=1.0, interactive=True)
                      b_mask_contrast_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Contrast', value=1.0, interactive=True)
                    b_invert_mask = gr.Checkbox(label='Invert Mask', value=True, interactive=True) #@param {type:"boolean"}
                    b_console = gr.Interface(lambda cmd:subprocess.run([cmd], capture_output=True, shell=True).stdout.decode('utf-8').strip(), "text", "text")
        with gr.TabItem('InPainting'):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        i_path_to_view = gr.Dropdown(label='Images', choices=batch_pathlist)
                    refresh_btn = gr.Button('Refresh')
                    i_init_img_array = gr.Image(value=inPaint, source="upload", interactive=True,
                                                                      type="pil", tool="sketch", visible=True,
                                                                      elem_id="mask")
                    i_prompts = gr.Textbox(label='Prompts',
                                placeholder='a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation',
                                lines=1)#animation_prompts
                    inPaint_btn = gr.Button('Generate')
                    i_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='Init Image Strength', value=0.01, interactive=True)#strength
                    i_batch_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim', interactive=True)#batch_name
                    i_outdir = gr.Textbox(label='Output Dir',  placeholder='/content/', lines=1, value=f'{opt.outdir}/_inPaint', interactive=True)#outdir
                    i_use_mask = gr.Checkbox(label='Use Mask Path', value=True, visible=False) #@param {type:"boolean"}
                    i_mask_file = gr.Textbox(label='Mask File', placeholder='https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg', interactive=True) #@param {type:"string"}
                    with gr.Row():
                        i_use_init = gr.Checkbox(label='use_init', value=True, visible=False)
                        i_init_image = gr.Textbox(label='Init Image link',  placeholder='https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg', lines=1)#init_image
                    i_seed_behavior = gr.Dropdown(label='Seed Behavior', choices=['iter', 'fixed', 'random'], value='iter', interactive=True)#seed_behavior
                    i_seed = gr.Number(label='Seed',  placeholder='SEED HERE', value='-1', interactive=True)#seed
                    i_save_settings = gr.Checkbox(label='Save Settings', value=True, visible=True)#save_settings

                with gr.Column():
                    inPainted = gr.Gallery()
                    i_log = gr.Textbox()
                    i_sampler = gr.Radio(label='Sampler',
                                     choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral', 'ddim'],
                                     value='klms', interactive=True)#sampler
                    with gr.Accordion('Upscalers'):
                        with gr.Row():
                            i_pregobig = gr.Checkbox(label = 'Pre-GoBig GFPGAN', value=False)
                            i_gobig = gr.Checkbox(label = 'GoBig', value=False)
                            i_GFPGAN = gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False)
                            i_bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                        i_gobigsampler = gr.Radio(label='Sampler',
                                            choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                            value='klms',
                                            interactive=True)#sampler

                        i_passes = gr.Slider(minimum=1, maximum=100, step=1, label='GoBig Passes', value=1, visible=True)
                        i_overlap = gr.Slider(minimum=1, maximum=512, step=1, label='GoBig Overlap', value=128, visible=True)
                        i_detail_steps = gr.Slider(minimum=1, maximum=250, step=1, label='GoBig Steps', value=50, visible=True)
                        i_detail_scale = gr.Slider(minimum=0, maximum=100, step=1, label='GoBig Detail Scale', value=10, visible=True)
                        i_g_strength = gr.Slider(minimum=0, maximum=1, step=0.1, label='GoBig Strength', value=0.3, visible=True)
                        i_upscale = gr.Slider(minimum=1, maximum=8, step=1, label='GFPGAN Scale (1x = Off)', value=1, interactive=True)




                    with gr.Row():
                        i_W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                        i_H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height
                    i_steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100, interactive=True)#steps
                    i_scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11, interactive=True)#scale
                    i_invert_mask = gr.Checkbox(label='Invert Mask', value=True, interactive=True) #@param {type:"boolean"}
                    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                    with gr.Row():

                        i_mask_brightness_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Brightness', value=1.0, interactive=True)
                        i_mask_contrast_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Contrast', value=1.0, interactive=True)
                    #
                    i_animation_mode = gr.Dropdown(label='Animation Mode',
                                                      choices=['None', '2D', '3D', 'Video Input', 'Interpolation'],
                                                      value='None',
                                                      visible=False)#animation_mode
                    i_max_frames = gr.Slider(minimum=1, maximum=1, step=1, label='Steps', value=1, visible=False)#inpaint_frames=0
                    i_ddim_eta = gr.Slider(minimum=0, maximum=1, step=0.1, label='DDIM ETA', value=1, visible=True)#

                    with gr.Row():
                        i_save_grid = gr.Checkbox(label='Save Grid', value=False)
                        i_make_grid = gr.Checkbox(label='Make Grid', value=False)
                        i_save_samples = gr.Checkbox(label='Save Samples', value=True)
                        i_n_samples = gr.Slider(minimum=1, maximum=4, step=1, label='Samples', value=1, visible=True)
                        i_n_batch = gr.Slider(minimum=1, maximum=100, step=1, label='Batches', value=1, visible=True)



        if not opt.no_var:
            with gr.TabItem('Variations'):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                input_var = gr.Image()
                                var_samples = gr.Slider(minimum=1, maximum=8, step=1, label='Samples (V100 = 3 x 512x512)', value=1)#n_samples
                                var_plms = gr.Checkbox(label='PLMS (Off is DDIM)', value=True, visible=True, interactive=True)
                                with gr.Row():
                                    v_GFPGAN = gr.Checkbox(label='GFPGAN, Upscaler', value=False)
                                    v_bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                                    v_upscale = gr.Slider(minimum=1, maximum=8, step=1, label='Upscaler, 1 to turn off', value=1, interactive=True)
                            output_var = gr.Gallery()
                        var_outdir = gr.Textbox(label='Output Folder',  value=f'{opt.outdir}/_variations', lines=1)
                        v_ddim_eta = gr.Slider(minimum=0, maximum=1, step=0.01, label='DDIM ETA', value=1.0, interactive=True)#scale
                        with gr.Row():

                            v_cfg_scale = gr.Slider(minimum=0, maximum=25, step=0.1, label='Cfg Scale', value=3.0, interactive=True)#scale
                            v_steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100, interactive=True)#steps

                        with gr.Row():
                            v_W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                            v_H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height

                        var_btn = gr.Button('Variations')
        if opt.load_p2p:
            with gr.TabItem('Text Image Editing'):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            p2p_path_to_view = gr.Dropdown(label='Images', choices=batch_pathlist)
                        e_init_image = gr.Image()
                        e_prompt = gr.Textbox(label='Input Image Prompt',  value='', interactive=True, lines=2)
                        prompt_edit = gr.Textbox(label='Input Image Prompt',  value='', interactive=True, lines=2)
                        with gr.Row():
                            e_width = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)
                            e_height = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)
                        with gr.Row():
                            e_steps = gr.Slider(minimum=10, maximum=250, step=1, label='Steps', value=50, interactive=True)
                            e_init_image_strength = gr.Slider(minimum=0, maximum=2, step=0.1, label='Init Image Strength', value=0.5, interactive=True)
                        e_guidance_scale = gr.Slider(minimum=0, maximum=25, step=0.1, label='Guidance Scale', value=7.5, interactive=True)
                        e_seed = gr.Textbox(label='Seed',  value='', interactive=True, lines=1)
                        e_outdir = gr.Textbox(label='Output Dir',  placeholder='/content/', lines=1, value=f'{opt.outdir}/_editor', interactive=True)#outdir
                        with gr.Accordion(label = 'Advanced Prompt Settings'):
                            with gr.Row():
                                prompt_edit_token_weights = gr.Textbox(label='Input Image Prompt',  placeholder='[(2, 2.5), (6, -5.0)]', interactive=True, lines=1)
                                prompt_edit_tokens_start = gr.Slider(minimum=0, maximum=2, step=0.1, label='Token Start', value=0.0, interactive=True)
                            with gr.Row():
                                prompt_edit_tokens_end = gr.Slider(minimum=0, maximum=2, step=0.1, label='Token End', value=1.0, interactive=True)
                                prompt_edit_spatial_start = gr.Slider(minimum=0, maximum=2, step=0.1, label='Token Spatial Start', value=0.0, interactive=True)
                            prompt_edit_spatial_end = gr.Slider(minimum=0, maximum=2, step=0.1, label='Token Spatial End', value=1.0, interactive=True)
                        edit_btn = gr.Button('edit')

                    with gr.Column():
                        edit_output = gr.Image()
        with gr.TabItem('NoodleSoup'):
            with gr.Column():
                input_prompt = gr.Textbox(label='IN',  placeholder='Portrait of a _adj-beauty_ _noun-emote_ _nationality_ woman from _pop-culture_ in _pop-location_ with pearlescent skin and white hair by _artist_, _site_', lines=2)
                output_prompt = gr.Textbox(label='OUT',  placeholder='Your Soup', lines=2)
                soup_btn = gr.Button('Cook')
                with gr.Row():
                  with gr.Column():
                      gr.Markdown(value=soup_help1)
                  with gr.Column():
                      gr.Markdown(value=soup_help2)
        for event in [stop_batch_btn.click]:
          event(stop, [], [])
    def saveSnapshot(new_k_prompts, animation_mode,
                        strength, max_frames, border, key_frames,
                        interp_spline, angle, zoom, translation_x,
                        translation_y, translation_z, color_coherence,
                        previous_frame_noise, previous_frame_strength,
                        video_init_path, extract_nth_frame, interpolate_x_frames,
                        batch_name, outdir, save_grid, save_settings, save_samples,
                        display_samples, n_samples, W, H, init_image, seed, sampler,
                        steps, scale, ddim_eta, seed_behavior, n_batch, use_init,
                        timestring, noise_schedule, strength_schedule, contrast_schedule,
                        resume_from_timestring, resume_timestring, make_grid,
                        GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y,
                        rotation_3d_z, use_depth_warping, midas_weight, near_plane,
                        far_plane, fov, padding_mode, sampling_mode):
                            anim_args = SimpleNamespace(**anim_dict(new_k_prompts, animation_mode,strength, max_frames, border, key_frames,interp_spline, angle, zoom, translation_x,translation_y, translation_z, color_coherence,previous_frame_noise, previous_frame_strength,video_init_path, extract_nth_frame, interpolate_x_frames,batch_name, outdir, save_grid, save_settings, save_samples,display_samples, n_samples, W, H, init_image, seed, sampler,steps, scale, ddim_eta, seed_behavior, n_batch, use_init,timestring, noise_schedule, strength_schedule, contrast_schedule,resume_from_timestring, resume_timestring, make_grid,GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y,rotation_3d_z, use_depth_warping, midas_weight, near_plane,far_plane, fov, padding_mode, sampling_mode))
                            os.makedirs(opt.cfg_path, exist_ok=True)
                            #filename = "/content/configs/test.txt"
                            #pseudoFilename = "test"
                            filename = f'{opt.cfg_path}/{batch_name}_{random.randint(10000, 99999)}_settings_snapshot.txt'
                            with open(filename, "w+", encoding="utf-8") as f:
                                json.dump(dict(anim_args.__dict__), f, ensure_ascii=False, indent=4)
                            list1 = []
                            for files in os.listdir(opt.cfg_path):
                                if files.endswith(".txt"):
                                    list1.append(files)
                            list2 = list1
                            return gr.Dropdown.update(choices=list1), gr.Dropdown.update(choices=list2)

    def loadSnapshot(snapshotFile):
        path = f'{opt.cfg_path}/{snapshotFile}'
        cfgfile = open(path)
        cfg = json.load(cfgfile)
        cfg = SimpleNamespace(**cfg)
        cfgfile.close()

        return cfg.new_k_prompts, cfg.animation_mode,cfg.strength, cfg.max_frames, cfg.border, cfg.key_frames,cfg.interp_spline, cfg.angle, cfg.zoom, cfg.translation_x,cfg.translation_y, cfg.translation_z, cfg.color_coherence,cfg.previous_frame_noise, cfg.previous_frame_strength,cfg.video_init_path, cfg.extract_nth_frame, cfg.interpolate_x_frames,cfg.batch_name, cfg.outdir, cfg.save_grid, cfg.save_settings, cfg.save_samples,cfg.display_samples, cfg.n_samples, cfg.W, cfg.H, cfg.init_image, cfg.seed, cfg.sampler,cfg.steps, cfg.scale, cfg.ddim_eta, cfg.seed_behavior, cfg.n_batch, cfg.use_init,cfg.timestring, cfg.noise_schedule, cfg.strength_schedule, cfg.contrast_schedule,cfg.resume_from_timestring, cfg.resume_timestring, cfg.make_grid,cfg.GFPGAN, cfg.bg_upsampling, cfg.upscale, cfg.rotation_3d_x, cfg.rotation_3d_y,cfg.rotation_3d_z, cfg.use_depth_warping, cfg.midas_weight, cfg.near_plane,cfg.far_plane, cfg.fov, cfg.padding_mode, cfg.sampling_mode

    def add_cfg_to_seq(cfg, seq):
        if seq == "":
            seq = f'{cfg}'
        else:
            seq = f'{seq}\n{cfg}'
        return seq, gr.update(open=True)

    def kb_build(string, frame, value):
      if frame != "" and value != "":
        if string == "":
            kf = f'{frame}:({value})'
        else:
            kf = f'{string},{frame}:({value})'
      else:
        kf = string
      return kf, gr.update(open=True)

    anim_func = anim
    anim_inputs = [animation_mode, new_k_prompts, key_frames,
                    prompts, batch_name, outdir, max_frames, GFPGAN,
                    bg_upsampling, upscale, W, H, steps, scale,
                    angle, zoom, translation_x, translation_y, translation_z,
                    rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping,
                    midas_weight, near_plane, far_plane, fov, padding_mode,
                    sampling_mode, seed_behavior, seed, interp_spline, noise_schedule,
                    strength_schedule, contrast_schedule, sampler, extract_nth_frame,
                    interpolate_x_frames, border, color_coherence, previous_frame_noise,
                    previous_frame_strength, video_init_path, save_grid, save_settings,
                    save_samples, display_samples, n_batch, n_samples, ddim_eta,
                    use_init, init_image, strength, timestring,
                    resume_from_timestring, resume_timestring, make_grid, b_init_img_array, b_use_mask,
                    b_mask_file, b_invert_mask, b_mask_brightness_adjust, b_mask_contrast_adjust,
                    use_seq, sequence, seqname]

    batch_inputs = [b_prompts, b_name, b_outdir, b_GFPGAN, b_bg_upsampling,
                    b_upscale, b_W, b_H, b_steps, b_scale, b_seed_behavior,
                    b_seed, b_sampler, b_save_grid, b_save_settings, b_save_samples,
                    b_n_batch, b_n_samples, b_ddim_eta, b_use_init, b_init_image,
                    b_strength, b_make_grid, b_init_img_array, b_use_mask,
                    b_mask_file, b_invert_mask, b_mask_brightness_adjust, b_mask_contrast_adjust,
                    b_gobig, b_passes, b_overlap, b_detail_steps, b_detail_scale, b_g_strength, b_gobigsampler, b_pregobig]




    mask_inputs = [i_prompts, i_batch_name, i_outdir, i_GFPGAN, i_bg_upsampling,
                    i_upscale, i_W, i_H, i_steps, i_scale, i_seed_behavior,
                    i_seed, i_sampler, i_save_grid, i_save_settings, i_save_samples,
                    i_n_batch, i_n_samples, i_ddim_eta, i_use_init, i_init_image,
                    i_strength, i_make_grid, i_init_img_array, i_use_mask,
                    i_mask_file, i_invert_mask, i_mask_brightness_adjust, i_mask_contrast_adjust,
                    i_gobig, i_passes, i_overlap, i_detail_steps, i_detail_scale, i_g_strength, i_gobigsampler, i_pregobig]

    kb_inputs = [kb_string, kb_frame, kb_value]
    kb_outputs = [kb_string, movement_settings]

    anim_outputs = [img, mp4_paths]

    anim_cfg_inputs = [new_k_prompts, animation_mode,
                        strength, max_frames, border, key_frames,
                        interp_spline, angle, zoom, translation_x,
                        translation_y, translation_z, color_coherence,
                        previous_frame_noise, previous_frame_strength,
                        video_init_path, extract_nth_frame, interpolate_x_frames,
                        batch_name, outdir, save_grid, save_settings, save_samples,
                        display_samples, n_samples, W, H, init_image, seed, sampler,
                        steps, scale, ddim_eta, seed_behavior, n_batch, use_init,
                        timestring, noise_schedule, strength_schedule, contrast_schedule,
                        resume_from_timestring, resume_timestring, make_grid,
                        GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y,
                        rotation_3d_z, use_depth_warping, midas_weight, near_plane,
                        far_plane, fov, padding_mode, sampling_mode]



    soup_inputs = [input_prompt]
    soup_outputs = [output_prompt]




    batch_outs = [batch_outputs, batch_path_to_view, b_log]
    inPaint_outputs = [inPainted, i_path_to_view, i_log]

    add_cfg_inputs = [cfg_seq_snapshots, sequence]
    add_cfg_outputs = [sequence, sequencer_settings]

    view_inputs=[mp4_path_to_view]
    view_outputs=[mp4_paths, mp4_path_to_view]

    anim_cfg_outputs = [cfg_snapshots, cfg_seq_snapshots]
    load_anim_cfg_inputs = [cfg_snapshots]
    if opt.load_p2p:
        editor_inputs=[e_prompt, prompt_edit, prompt_edit_token_weights,
                        prompt_edit_tokens_start, prompt_edit_tokens_end,
                        prompt_edit_spatial_start, prompt_edit_spatial_end,
                        e_guidance_scale, e_steps, e_seed, e_width, e_height,
                        e_init_image, e_init_image_strength, e_outdir]
        editor_outputs=[edit_output]
        edit_btn.click(fn=run_p2p, inputs=editor_inputs, outputs=editor_outputs)
        p2p_path_to_view.change(fn=view_editor_file, inputs=[p2p_path_to_view], outputs=[e_init_image])

    if not opt.no_var:
        var_inputs = [input_var, var_outdir, var_samples, var_plms, v_cfg_scale, v_steps, v_W, v_H, v_ddim_eta, v_GFPGAN, v_bg_upsampling, v_upscale]
        var_outputs = [output_var]
        var_btn.click(variations, inputs=var_inputs, outputs=var_outputs)

    soup_btn.click(fn=process_noodle_soup, inputs=soup_inputs, outputs=soup_outputs)

    refresh_btn.click(fn=refresh, inputs=i_init_img_array, outputs=i_init_img_array)
    inPaint_btn.click(fn=run_batch, inputs=mask_inputs, outputs=inPaint_outputs)

    anim_btn.click(fn=anim, inputs=anim_inputs, outputs=anim_outputs)
    kb_btn.click(fn=kb_build, inputs=kb_inputs, outputs=kb_outputs)


    add_cfg_btn.click(fn=add_cfg_to_seq, inputs=add_cfg_inputs, outputs=add_cfg_outputs)
    save_cfg_btn.click(fn=saveSnapshot, inputs=anim_cfg_inputs, outputs=anim_cfg_outputs)

    load_cfg_btn.click(fn=loadSnapshot, inputs=load_anim_cfg_inputs, outputs=anim_cfg_inputs)

    mp4_path_to_view.change(fn=view_video, inputs=view_inputs, outputs=view_outputs)

    batch_btn.click(fn=run_batch, inputs=batch_inputs, outputs=batch_outs)
    batch_path_to_view.change(fn=view_batch_file, inputs=[batch_path_to_view], outputs=[batch_outputs])


    i_path_to_view.change(fn=view_editor_file, inputs=[i_path_to_view], outputs=[i_init_img_array])


class ServerLauncher(threading.Thread):
    def __init__(self, demo):
        threading.Thread.__init__(self)
        self.name = 'Gradio Server Thread'
        self.demo = demo

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gradio_params = {
            'inbrowser': True,
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': True,
            'show_error': True,
            'debug': False
        }
        #if not opt.share:
        demo.queue(concurrency_count=5)
        #if opt.share and opt.share_password:
        #    gradio_params['auth'] = ('webui', opt.share_password)

        # Check to see if Port 7860 is open
        port_status = 1
        while port_status != 0:
            try:
                self.demo.launch(**gradio_params)
            except (OSError) as e:
                print (f'Error: Port: 7860 is not open yet. Please wait, this may take upwards of 60 seconds...')
                time.sleep(10)
            else:
                port_status = 0

    def stop(self):
        self.demo.close() # this tends to hang

def launch_server():
    server_thread = ServerLauncher(demo)
    server_thread.start()

    try:
        while server_thread.is_alive():
            time.sleep(60)
    except (KeyboardInterrupt, OSError) as e:
        crash(e, 'Shutting down...')
"""
def run_headless():
    with open(opt.cli, 'r', encoding='utf8') as f:
        kwargs = yaml.safe_load(f)
    target = kwargs.pop('target')
    if target == 'txt2img':
        target_func = txt2img
    elif target == 'img2img':
        target_func = img2img
        raise NotImplementedError()
    else:
        raise ValueError(f'Unknown target: {target}')
    prompts = kwargs.pop("prompt")
    prompts = prompts if type(prompts) is list else [prompts]
    for i, prompt_i in enumerate(prompts):
        print(f"===== Prompt {i+1}/{len(prompts)}: {prompt_i} =====")
        output_images, seed, info, stats = target_func(prompt=prompt_i, **kwargs)
        print(f'Seed: {seed}')
        print(info)
        print(stats)
        print()
"""

if __name__ == '__main__':
    #if opt.cli is None:
        launch_server()
    #else:
    #    run_headless()
