import json
import threading, asyncio, argparse, math, os, pathlib, shutil, subprocess, sys, time
import cv2
import numpy as np
import pandas as pd
import random
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
from PIL import Image
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

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help="choose which GPU to use if you have multiple", default=0)
parser.add_argument("--cli", type=str, help="don't launch web server, take Python function kwargs from this file.", default=None)
parser.add_argument("--ckpt", type=str, help="Model Path", default=None)
parser.add_argument("--no_var", type=str, help="Variations Model Path", default=None)
parser.add_argument("--var_ckpt", type=str, help="Variations Model Path", default=None)
opt = parser.parse_args()

sys.path.extend([
    '/content/src/taming-transformers',
    '/content/src/clip',
    '/content/stable-diffusion-gradio-anim-opt/',
    '/content/k-diffusion',
    '/content/pytorch3d-lite',
    '/content/AdaBins',
    '/content/MiDaS',
    '/content/soup'
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

import nsp_pantry
from nsp_pantry import nspterminology, nsp_parse
models_path = opt.ckpt #@param {type:"string"}
output_path = "/content/output" #@param {type:"string"}

mount_google_drive = False #@param {type:"boolean"}Will Remove
force_remount = False #Will Remove

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



terminology_database = nspterminology

if terminology_database:
	log.green("Loaded terminology database from the pantry.")
	print("\x1B[3mMmm. Noodle Soup.\x1B[0m")
else:
	log.red("Unable to load terminology database")


def process_noodle_soup(text_prompts):

  new_prom = list(text_prompts.split("\n"))
  nan = "nan"
  prompt_series = pd.Series([np.nan for a in range(len(new_prom))])
  for i, nan in prompt_series.items():
    print(i)
    print(nan)
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
    return '_'.join(prompt.split(" "))

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

    print('exiting...calling os._exit(0)')
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()

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
    model = model.half()
    model.cuda()
    model.eval()
    return model


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

if load_on_run_all and ckpt_valid:
    local_config = OmegaConf.load(f"{ckpt_config_path}")
    model = load_model_from_config(local_config, f"{ckpt_path}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to("cpu")

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

ckpt_var="/gdrive/MyDrive/sd-clip-vit-l14-img-embed_ema_only.ckpt"
config_var="stable-diffusion-gradio-anim-opt/configs/stable-diffusion/sd-image-condition-finetune.yaml"
config_var = OmegaConf.load(config_var)

if not opt.no_var:
    model_var = load_var_model_from_config(config_var, opt.var_ckpt, 'cpu')

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

def makevideo(args):
    skip_video_for_run_all = False #@param {type: 'boolean'}
    fps = 12#@param {type:"number"}

    if skip_video_for_run_all == True:
        print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
    else:
        print('Saving video')
        image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
        mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")

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
        '-frames:v', str(args.max_frames),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        mp4_path
        ]
        subprocess.call(cmd)
    args.mp4_path = mp4_path
    return mp4_path
        #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #stdout, stderr = process.communicate()
        #if process.returncode != 0:
        #    print(stderr)
        #    raise RuntimeError(stderr)

        #mp4 = open(mp4_path,'rb').read()
        #data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        #display.display( display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )

def get_inbetweens(key_frames, integer=False, interp_method='Linear'):
    key_frame_series = pd.Series([np.nan for a in range(args.max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
      interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
      interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[args.max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(),limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def transform_image_3d(prev_img_cv2, adabins_helper, midas_model, midas_transform, rot_mat, translate, args):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion

    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    # predict depth with AdaBins
    use_adabins = args.midas_weight < 1.0 and adabins_helper is not None
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
            depth_map = midas_depth*args.midas_weight + adabins_depth*(1.0-args.midas_weight)
        else:
            depth_map = midas_depth

        depth_map = np.expand_dims(depth_map, axis=0)
        depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)
    else:
        depth_tensor = torch.ones((h, w), device=device)

    pixel_aspect = 1.0 # aspect of an individual pixel (so usually 1.0)
    near, far, fov_deg = args.near_plane, args.far_plane, args.fov
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
        mode=args.sampling_mode,
        padding_mode=args.padding_mode,
        align_corners=False
    )

    # convert back to cv2 style numpy array 0->255 uint8
    result = rearrange(
        new_image.squeeze().clamp(0,1) * 255.0,
        'c h w -> h w c'
    ).cpu().numpy().astype(np.uint8)
    return result

def generate(prompt, name, outdir, GFPGAN, bg_upsampling, upscale, W, H, steps, scale, seed, samplern, n_batch, n_samples, ddim_eta, use_init, init_image, init_sample, strength, use_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, invert_mask, dynamic_threshold, static_threshold, C, f, init_c, return_latent=False, return_sample=False, return_c=False):
    print(f'Prompt when Generating: {prompt}')
    precision = "autocast"
    seed_everything(seed)
    os.makedirs(outdir, exist_ok=True)

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

                    print(f'Prompts when Generating: {prompts}')

                    c = model.get_learned_conditioning(prompts)

                    if init_c != None:
                        c = init_c

                    if samplern in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                        samples = sampler_fn(
                            c=c,
                            uc=uc,
                            C=C,
                            H=H,
                            f=f,
                            W=W,
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
                            z_enc = torch.randn([n_samples, C, H // f, W // f], device=device)
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
                        if GFPGAN:
                            image = FACE_RESTORATION(image, bg_upsampling, upscale).astype(np.uint8)
                            image = Image.fromarray(image)
                        else:
                            image = image


                        results.append(image)
    return results


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

#def render_image_batch(batch_args):
        #    display.clear_output(wait=True)
        #    display.display(grid_image)


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

def batch_dict(b_prompts, b_name, b_outdir, b_GFPGAN, b_bg_upsampling, b_upscale, b_W, b_H, b_steps, b_scale, b_seed_behavior, b_seed, b_sampler, b_save_grid, b_save_settings, b_save_samples, b_n_batch, b_n_samples, b_ddim_eta, b_use_init, b_init_image, b_strength, b_make_grid):
    return locals()

def run_batch(b_prompts, b_name, b_outdir, b_GFPGAN, b_bg_upsampling, b_upscale, b_W, b_H, b_steps, b_scale, b_seed_behavior, b_seed, b_sampler, b_save_grid, b_save_settings, b_save_samples, b_n_batch, b_n_samples, b_ddim_eta, b_use_init, b_init_image, b_strength, b_make_grid, b_init_img_array):

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
                                                b_seed,
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

        model.to('cuda')
        #b_prompts = prompts

        b_prompts = list(b_prompts.split("\n"))

        # create output folder for the batch
        os.makedirs(b_outdir, exist_ok=True)
        if b_save_settings or b_save_samples:
            print(f"Saving to {b_outdir}_*")

        #save settings for the batch
        if b_save_settings:
            filename = os.path.join(b_outdir, f"{b_name}_{b_seed}_{random.randint(10000, 99999)}_settings.txt")
            with open(filename, "w+", encoding="utf-8") as f:
                json.dump(dict(batchargs.__dict__), f, ensure_ascii=False, indent=4)

        index = 0
        all_images = []
        b_outputs = []

        # function for init image batching
        init_array = []
        #Defaults needed by def generate()
        b_init_latent = None
        b_init_sample = None
        b_init_c = None
        b_mask_contrast_adjust = 1.0
        b_mask_brightness_adjust = 1.0
        b_invert_mask = False
        b_use_mask = False
        dynamic_threshold = None
        static_threshold = None
        precision = 'autocast'
        #fixed_code = True
        C = 4
        f = 8


        if b_init_img_array != None:
            initdir = f'{b_outdir}/init'
            os.makedirs(initdir, exist_ok=True)
            b_init_image = f'{b_outdir}/init/init.png'
            b_mask_file = f'{b_outdir}/init/mask.png'
            b_init_img_array['image'].save(os.path.join(b_outdir, b_init_image))
            b_init_img_array['mask'].save(os.path.join(b_outdir, b_mask_file))
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
            b_sanitized = f'{b_sanitized[:128]}_{b_seed}_{random.randint(10000, 99999)}'


            for batch_index in range(b_n_batch):
                #if clear_between_batches:
                #    display.clear_output(wait=True)
                #print(f"Batch {batch_index+1} of {b_n_batch}")

                for image in init_array: # iterates the init images
                    b_init_image = image
                    print(f'USING SEED FOR BATCH:{b_seed}')

                    results = generate(prompt=b_prompts[iprompt], name=b_name, outdir=b_outdir,
                                       GFPGAN=b_GFPGAN, bg_upsampling=b_bg_upsampling, upscale=b_upscale,
                                       W=b_W, H=b_H, steps=b_steps, scale=b_scale, seed=b_seed,
                                       samplern=b_sampler, n_batch=b_n_batch, n_samples=b_n_samples, ddim_eta=b_ddim_eta,
                                       use_init=b_use_init, init_image=b_init_image,
                                       init_sample=b_init_sample, strength=b_strength,
                                       use_mask=b_use_mask, mask_file=b_mask_file,
                                       mask_contrast_adjust=b_mask_contrast_adjust,
                                       mask_brightness_adjust=b_mask_brightness_adjust, invert_mask=b_invert_mask,
                                       dynamic_threshold=None, static_threshold=None, C=4, f=8, init_c=None)


                    for image in results:
                        #all_images.append(results[image])
                        if b_make_grid:
                            all_images.append(T.functional.pil_to_tensor(image))
                        if b_save_samples:
                            print(f"{b_outdir}/{b_sanitized}_{index:05}_{b_seed}.png")

                            b_filename = (f"{b_sanitized}_{index:05}.png")
                            b_fpath = (f"{b_outdir}/{b_filename}")
                            print(f"File at: {b_fpath}")

                            image.save(os.path.join(b_outdir, b_filename))
                            b_outputs.append(b_fpath)
                        #if b_display_samples:
                        #    display.display(image)
                        index += 1
                    if b_seed_behavior != 'fixed':
                        b_seed = next_seed(b_seed, b_seed_behavior)

        print(f"Filepath List: {b_outputs}")

        if b_make_grid:
            b_grid_rows = 2
            grid = mkgrid(all_images, nrow=int(len(all_images)/b_grid_rows))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid_filename = f"{b_sanitized}_{b_name}_{iprompt:05d}_grid_{random.randint(10000, 99999)}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            grid_image.save(os.path.join(b_outdir, grid_filename))
            grid_path = f"{b_outdir}/{grid_filename}"
            b_outputs.append(grid_path)
        return b_outputs

def anim_dict(animation_prompts, prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, translation_z, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring, make_grid, GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, init_img_array, use_mask, mask_file, invert_mask, mask_brightness_adjust, mask_contrast_adjust):

    precision = 'autocast'
    fixed_code = True
    C = 4
    f = 8
    dynamic_threshold = None
    static_threshold = None
    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None
    return locals()


def anim(animation_mode: str, animation_prompts: str, key_frames: bool, prompts: str, batch_name: str, outdir: str, max_frames: int, GFPGAN: bool, bg_upsampling: bool, upscale: int, W: int, H: int, steps: int, scale: int, angle: str, zoom: str, translation_x: str, translation_y: str, translation_z: str, rotation_3d_x: str, rotation_3d_y: str, rotation_3d_z: str, use_depth_warping: bool, midas_weight: float, near_plane: int, far_plane: int, fov: int, padding_mode: str, sampling_mode: str, seed_behavior: str, seed: str, interp_spline: str, noise_schedule: str, strength_schedule: str, contrast_schedule: str, sampler: str, extract_nth_frame: int, interpolate_x_frames: int, border: str, color_coherence: str, previous_frame_noise: float, previous_frame_strength: float, video_init_path: str, save_grid: bool, save_settings: bool, save_samples: bool, display_samples: bool, n_batch: int, n_samples: int, ddim_eta: float, use_init: bool, init_image: str, strength: float, timestring: str, resume_from_timestring: bool, resume_timestring: str, make_grid: bool, init_img_array, use_mask, mask_file, invert_mask, mask_brightness_adjust, mask_contrast_adjust):


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
    dynamic_threshold = None
    static_threshold = None
    precision = 'autocast'
    #fixed_code = True
    C = 4
    f = 8

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
        args.mtransform = midas_transform

        return midas_model, midas_transform

    def anim_frame_warp_2d(prev_img_cv2, args, frame_idx):
        angle = args.angle_series[frame_idx]
        zoom = args.zoom_series[frame_idx]
        translation_x = args.translation_x_series[frame_idx]
        translation_y = args.translation_y_series[frame_idx]

        center = (args.W // 2, args.H // 2)
        trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
        trans_mat = np.vstack([trans_mat, [0,0,1]])
        rot_mat = np.vstack([rot_mat, [0,0,1]])
        xform = np.matmul(rot_mat, trans_mat)

        return cv2.warpPerspective(
            prev_img_cv2,
            xform,
            (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
            borderMode=cv2.BORDER_WRAP if args.border == 'wrap' else cv2.BORDER_REPLICATE
        )

    def anim_frame_warp_3d(prev_img_cv2, args, frame_idx, adabins_helper, midas_model):
        TRANSLATION_SCALE = 1.0/200.0 # matches Disco
        translate_xyz = [
            -args.translation_x_series[frame_idx] * TRANSLATION_SCALE,
            args.translation_y_series[frame_idx] * TRANSLATION_SCALE,
            -args.translation_z_series[frame_idx] * TRANSLATION_SCALE
        ]
        rotate_xyz = [
            math.radians(args.rotation_3d_x_series[frame_idx]),
            math.radians(args.rotation_3d_y_series[frame_idx]),
            math.radians(args.rotation_3d_z_series[frame_idx])
        ]
        rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
        result = transform_image_3d(prev_img_cv2, adabins_helper, midas_model, args.mtransform, rot_mat, translate_xyz, args)
        torch.cuda.empty_cache()
        return result

    def DeformAnimKeys(angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, noise_schedule, strength_schedule, contrast_schedule):
        angle_series = get_inbetweens(parse_key_frames(args.angle))
        zoom_series = get_inbetweens(parse_key_frames(args.zoom))
        translation_x_series = get_inbetweens(parse_key_frames(args.translation_x))
        translation_y_series = get_inbetweens(parse_key_frames(args.translation_y))
        translation_z_series = get_inbetweens(parse_key_frames(args.translation_z))
        rotation_3d_x_series = get_inbetweens(parse_key_frames(args.rotation_3d_x))
        rotation_3d_y_series = get_inbetweens(parse_key_frames(args.rotation_3d_y))
        rotation_3d_z_series = get_inbetweens(parse_key_frames(args.rotation_3d_z))
        noise_schedule_series = get_inbetweens(parse_key_frames(args.noise_schedule))
        strength_schedule_series = get_inbetweens(parse_key_frames(args.strength_schedule))
        contrast_schedule_series = get_inbetweens(parse_key_frames(args.contrast_schedule))

    def render_animation(args):
        prom = animation_prompts
        key = prompts

        new_prom = list(prom.split("\n"))
        new_key = list(key.split("\n"))

        prompts = dict(zip(new_key, new_prom))
        print (prompts)
        # animations use key framed prompts
        #args.prompts = animation_prompts
        DeformAnimKeys(args)
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
        #settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        #with open(settings_filename, "w+", encoding="utf-8") as f:
        #    s = {**dict(args.__dict__)}
        #    json.dump(s, f, ensure_ascii=False, indent=4)

        # resume from timestring
        if args.resume_from_timestring:
            args.timestring = args.resume_timestring
        #prompt_series = args.prompts
        # expand prompts out to per-frame
        #prompt_series = {}
        #prompt_series = pd.Series([np.nan for a in range(args.max_frames)])

        promptList = list(args.animation_prompts.split("\n"))
        #keyList = list(args.prompts.split("\n"))
        #anim_prompts = dict(zip(new_key, new_prom))

        #for i in range (len(keyList)):
        #  n = int(keyList[i])
        #  prompt_series[n] = promptList[i]
        #prompt_series = prompt_series.ffill().bfill()
        prompt_series = pd.Series([np.nan for a in range(args.max_frames)])
        for i, prompt in prompts.items():
            n = int(i)
            prompt_series[n] = prompt

        prompt_series = prompt_series.ffill().bfill()


        print("PROMPT SERIES")

        print(prompt_series)

        print("END OF PROMPT SERIES")

        # check for video inits
        using_vid_init = args.animation_mode == 'Video Input'

        # load depth model for 3D
        if args.animation_mode == '3D' and args.use_depth_warping:
            download_depth_models()
            adabins_helper = InferenceHelper(dataset='nyu', device=device)
            midas_model, midas_transform = load_depth_model()
            #load_depth_model()
        else:
            adabins_helper, midas_model, midas_transform = None, None, None

        args.n_samples = 1
        prev_sample = None
        color_match_sample = None
        for frame_idx in range(start_frame,args.max_frames):
            print(f"Rendering animation frame {frame_idx} of {args.max_frames}")
            noise = args.noise_schedule_series[frame_idx]
            strength = args.strength_schedule_series[frame_idx]
            contrast = args.contrast_schedule_series[frame_idx]

            # resume animation
            if args.resume_from_timestring:
                path = os.path.join(args.outdir,f"{args.timestring}_{frame_idx-1:05}.png")
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                prev_sample = sample_from_cv2(img)

            # apply transforms to previous frame
            if prev_sample is not None:

                if args.animation_mode == '2D':
                    prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), args, frame_idx)
                else: # '3D'
                    prev_img = anim_frame_warp_3d(sample_to_cv2(prev_sample), args, frame_idx, adabins_helper, midas_model)

                # apply color matching
                if args.color_coherence != 'None':
                    if color_match_sample is None:
                        color_match_sample = prev_img.copy()
                    else:
                        prev_img = maintain_colors(prev_img, color_match_sample, args.color_coherence)

                # apply scaling
                contrast_sample = prev_img * contrast
                # apply frame noising
                noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

                # use transformed previous frame as init for current
                args.use_init = True
                #args.init_sample = noised_sample.half().to(device)
                if half_precision:
                    args.init_sample = noised_sample.half().to(device)
                else:
                    args.init_sample = noised_sample.to(device)
                args.strength = max(0.0, min(1.0, strength))

            # grab prompt for current frame
            args.prompt = prompt_series[frame_idx]
            print(f"{args.prompt} {args.seed}")

            # grab init image for current frame
            if using_vid_init:
                init_frame = os.path.join(args.outdir, 'inputframes', f"{frame_idx+1:04}.jpg")
                print(f"Using video init frame {init_frame}")
                args.init_image = init_frame

            # sample the diffusion model
            results = generate(seed, outdir, sampler, n_samples, prompt, init_latent, init_sample, use_init, init_image, strength, use_mask, mask_file, mask_contrast_adjust, mask_brightness_adjust, invert_mask, steps, ddim_eta, dynamic_threshold, static_threshold, C, f, H, W, scale, init_c, b_GFPGAN, b_bg_upsampling, b_upscale, return_latent=False, return_sample=True)

            sample, image = results[0], results[1]

            filename = f"{args.timestring}_{frame_idx:05}.png"
            image.save(os.path.join(args.outdir, filename))
            if not using_vid_init:
                prev_sample = sample

            args.seed = next_seed(args)


    def make_xform_2d(width, height, translation_x, translation_y, angle, scale):
        center = (width // 2, height // 2)
        trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        trans_mat = np.vstack([trans_mat, [0,0,1]])
        rot_mat = np.vstack([rot_mat, [0,0,1]])
        return np.matmul(rot_mat, trans_mat)

    def render_input_video(args):
        # create a folder for the video input frames to live in
        video_in_frame_path = os.path.join(args.outdir, 'inputframes')
        os.makedirs(os.path.join(args.outdir, video_in_frame_path), exist_ok=True)

        # save the video frames from input video
        print(f"Exporting Video Frames (1 every {args.extract_nth_frame}) frames to {video_in_frame_path}...")
        try:
            for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
                f.unlink()
        except:
            pass
        vf = r'select=not(mod(n\,'+str(args.extract_nth_frame)+'))'
        subprocess.run([
            'ffmpeg', '-i', f'{args.video_init_path}',
            '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2',
            '-loglevel', 'error', '-stats',
            os.path.join(video_in_frame_path, '%04d.jpg')
        ], stdout=subprocess.PIPE).stdout.decode('utf-8')

        # determine max frames from length of input frames
        args.max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])

        args.use_init = True
        print(f"Loading {args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}")
        render_animation(args)

    def render_interpolation(args):
        # animations use key framed prompts
        args.prompts = animation_prompts

        # create output folder for the batch
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Saving animation frames to {args.outdir}")

        # save settings for the batch
        settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(args.__dict__), **dict(args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)

        # Interpolation Settings
        args.n_samples = 1
        args.seed_behavior = 'fixed' # force fix seed at the moment bc only 1 seed is available
        prompts_c_s = [] # cache all the text embeddings

        print(f"Preparing for interpolation of the following...")

        for i, prompt in animation_prompts.items():
          args.prompt = prompt

          # sample the diffusion model
          results = generate(args, return_c=True)
          c, image = results[0], results[1]
          prompts_c_s.append(c)

          # display.clear_output(wait=True)
          #display.display(image)

          args.seed = next_seed(args)

        display.clear_output(wait=True)
        print(f"Interpolation start...")

        frame_idx = 0

        if args.interpolate_key_frames:
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
                args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1/dist_frames))

                # sample the diffusion model
                results = generate(args)
                image = results[0]

                filename = f"{args.timestring}_{frame_idx:05}.png"
                image.save(os.path.join(args.outdir, filename))
                frame_idx += 1

                display.clear_output(wait=True)
                display.display(image)

                args.seed = next_seed(args)

        else:
          for i in range(len(prompts_c_s)-1):
            for j in range(args.interpolate_x_frames+1):
              # interpolate the text embedding
              prompt1_c = prompts_c_s[i]
              prompt2_c = prompts_c_s[i+1]
              args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1/(args.interpolate_x_frames+1)))

              # sample the diffusion model
              results = generate(args)
              image = results[0]

              filename = f"{args.timestring}_{frame_idx:05}.png"
              image.save(os.path.join(args.outdir, filename))
              frame_idx += 1

              display.clear_output(wait=True)
              display.display(image)

              args.seed = next_seed(args)

        # generate the last prompt
        args.init_c = prompts_c_s[-1]
        results = generate(args)
        image = results[0]
        filename = f"{args.timestring}_{frame_idx:05}.png"
        image.save(os.path.join(args.outdir, filename))

        display.clear_output(wait=True)
        display.display(image)
        args.seed = next_seed(args)

        #clear init_c
        args.init_c = None

    #animation_prompts = dict(zip(new_key, new_prom))
    print (prompts)
    print (animation_prompts)
    #animation_mode = animation_mode
    anim_dict(animation_prompts, prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, translation_z, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring, make_grid, GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, init_img_array, use_mask, mask_file, invert_mask, mask_brightness_adjust, mask_contrast_adjust)


    anim_args = SimpleNamespace(**arger(animation_prompts, prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, translation_z, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring, make_grid, GFPGAN, bg_upsampling, upscale, rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode))


    args.outputs = []
    if args.animation_mode == 'None':
        args.max_frames = 1

    #if args.key_frames:
    #    angle_series = get_inbetweens(parse_key_frames(args.angle))
    #    zoom_series = get_inbetweens(parse_key_frames(args.zoom))
    #    translation_x_series = get_inbetweens(parse_key_frames(args.translation_x))
    #    translation_y_series = get_inbetweens(parse_key_frames(args.translation_y))
    #    noise_schedule_series = get_inbetweens(parse_key_frames(args.noise_schedule))
    #    strength_schedule_series = get_inbetweens(parse_key_frames(args.strength_schedule))
    #    contrast_schedule_series = get_inbetweens(parse_key_frames(args.contrast_schedule))

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.outdir = f'{args.outdir}/{args.timestring}'
    args.strength = max(0.0, min(1.0, args.strength))
    args.returns = {}
    if args.seed == -1:
        args.seed = random.randint(0, 2**32)
    if args.animation_mode == 'Video Input':
        args.use_init = True
    if not args.use_init:
        args.init_image = None
        args.strength = 0
    if args.sampler == 'plms' and (args.use_init or args.animation_mode != 'None'):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = 'klms'
    if args.sampler != 'ddim':
        args.ddim_eta = 0

    if args.animation_mode == '2D' or args.animation_mode == '3D':
        render_animation(args)
        makevideo(args)
        #mem = torch.cuda.memory_allocated()/1e6
        #model.to("cpu")

        #while(torch.cuda.memory_allocated()/1e6 >= mem):
        #    time.sleep(1)
        torch_gc()
        return args.mp4_path
    elif args.animation_mode == 'Video Input':
        render_input_video(args)
        makevideo(args)
        #mem = torch.cuda.memory_allocated()/1e6
        #model.to("cpu")

        #while(torch.cuda.memory_allocated()/1e6 >= mem):
        #    time.sleep(1)
        torch_gc()
        return args.mp4_path
    elif args.animation_mode == 'Interpolation':
        render_interpolation(args)
        makevideo(args)
        #mem = torch.cuda.memory_allocated()/1e6
        #model.to("cpu")

        #while(torch.cuda.memory_allocated()/1e6 >= mem):
        #    time.sleep(1)
        torch_gc()
        return args.mp4_path
    else:
        render_image_batch(args)
        mem = torch.cuda.memory_allocated()/1e6
        #model.to("cpu")

        #while(torch.cuda.memory_allocated()/1e6 >= mem):
        #    time.sleep(1)
        torch_gc()
        return args.outputs

def refresh(choice):
    print(choice)
    #choice = None
    return choice

torch_gc()
inPaint=None

demo = gr.Blocks()

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

with demo:
    with gr.Tabs():
        with gr.TabItem('Animation'):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim', interactive=True)#batch_name
                    outdir = gr.Textbox(label='Output Dir',  placeholder='/content', lines=1, value='/gdrive/MyDrive/sd_anims', interactive=True)#outdir
                    animation_prompts = gr.Textbox(label='Prompts - divided by enter',
                                                    placeholder='a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation',
                                                    lines=5, interactive=True)#animation_prompts
                    key_frames = gr.Checkbox(label='KeyFrames',
                                            value=True,
                                            visible=False, interactive=True)#key_frames
                    prompts = gr.Textbox(label='Keyframes - numbers divided by enter',
                                        placeholder='0',
                                        lines=5,
                                        value='0', interactive=True)#prompts
                    use_init = gr.Checkbox(label='Use Init', value=False, visible=True, interactive=True)#use_init
                    init_image = gr.Textbox(label='Init Image link',  placeholder='https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg', lines=1, interactive=True)#init_image
                    anim_btn = gr.Button('Generate')
                with gr.Column(scale=1.6):
                        mp4_paths = gr.Video(label='Generated Video')

                        #output = gr.Text()
                with gr.Column(scale=2.5):
                    with gr.TabItem('Movements'):
                        with gr.Row():
                            with gr.Column(scale=0.13):
                                angle = gr.Textbox(label='Angles',  placeholder='0:(0)', lines=1, value='0:(0)')#angle
                                zoom = gr.Textbox(label='Zoom',  placeholder='0: (1.04)', lines=1, value='0:(1.0)')#zoom
                                translation_x = gr.Textbox(label='Translation X (+ is Camera Left, large values [1 - 50])',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_x
                                translation_y = gr.Textbox(label='Translation Y + = R',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_y
                                translation_z = gr.Textbox(label='Translation Z + = FW',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#translation_y
                                rotation_3d_x = gr.Textbox(label='Rotation 3D X (+ is Up)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_x
                                rotation_3d_y = gr.Textbox(label='Rotation 3D Y (+ is Right)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_y
                            with gr.Column(scale=0.13):
                                rotation_3d_z = gr.Textbox(label='Rotation 3D Z (+ is Clockwise)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_z
                                use_depth_warping = gr.Checkbox(label='Depth Warping', value=True, visible=True)#use_depth_warping
                                midas_weight = gr.Slider(minimum=0, maximum=5, step=0.1, label='Midas Weight', value=0.3, visible=True)#midas_weight
                                near_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Near Plane', value=200, visible=True)#near_plane
                                far_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Far Plane', value=1000, visible=True)#far_plane
                                fov = gr.Slider(minimum=0, maximum=360, step=1, label='FOV', value=40, visible=True)#fov
                                padding_mode = gr.Dropdown(label='Padding Mode', choices=['border', 'reflection', 'zeros'], value='border', visible=True)#padding_mode
                                sampling_mode = gr.Dropdown(label='Sampling Mode', choices=['bicubic', 'bilinear', 'nearest'], value='bicubic', visible=True)#sampling_mode
                    with gr.TabItem('Video / Init Video / Interpolation settings'):
                      sampler = gr.Radio(label='Sampler',
                                          choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                          value='klms', interactive=True)#sampler
                      with gr.Row():
                          GFPGAN = gr.Checkbox(label='GFPGAN, Upscaler', value=False)
                          bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                          upscale = gr.Slider(minimum=1, maximum=8, step=1, label='Upscaler, 1 to turn off', value=1, interactive=True)
                      W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                      H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height
                      steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100, interactive=True)#steps
                      scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11, interactive=True)#scale
                      video_init_path = gr.Textbox(label='Video init path',  placeholder='/content/video_in.mp4', lines=1)#video_init_path
                      with gr.Row():
                          extract_nth_frame = gr.Slider(minimum=1, maximum=100, step=1, label='Extract n-th frame', value=1)#extract_nth_frame
                          interpolate_x_frames = gr.Slider(minimum=1, maximum=25, step=1, label='Interpolate n frames', value=4)#interpolate_x_frames
                      with gr.Row():
                          previous_frame_noise = gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Noise', value=0.02)#previous_frame_noise
                          previous_frame_strength = gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Strength', value=0.4)#previous_frame_strength
                    with gr.TabItem('Anim Settings'):
                        with gr.Row():
                            with gr.Column(scale=0.15):
                                animation_mode = gr.Dropdown(label='Animation Mode',
                                                                choices=['None', '2D', '3D', 'Video Input', 'Interpolation'],
                                                                value='3D')#animation_mode

                                max_frames = gr.Slider(minimum=1, maximum=1000, step=1, label='Frames to render', value=100)#max_frames
                                seed_behavior = gr.Dropdown(label='Seed Behavior', choices=['iter', 'fixed', 'random'], value='iter')#seed_behavior
                                seed = gr.Number(label='Seed',  placeholder='SEED HERE', value='-1')#seed
                                interp_spline = gr.Dropdown(label='Spline Interpolation', choices=['Linear', 'Quadratic', 'Cubic'], value='Linear')#interp_spline
                                noise_schedule = gr.Textbox(label='Noise Schedule',  placeholder='0:(0)', lines=1, value='0:(0.02)')#noise_schedule
                                strength_schedule = gr.Textbox(label='Strength_Schedule',  placeholder='0:(0)', lines=1, value='0:(0.65)')#strength_schedule
                                contrast_schedule = gr.Textbox(label='Contrast Schedule',  placeholder='0:(0)', lines=1, value='0:(1.0)')#contrast_schedule
                            with gr.Column(scale=0.15):
                                color_coherence = gr.Dropdown(label='Color Coherence', choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'], value='Match Frame 0 RGB')#color_coherence
                                save_settings = gr.Checkbox(label='Save Settings', value=True, visible=True)#save_settings
                                border = gr.Dropdown(label='Border', choices=['wrap', 'replicate'], value='wrap')#border
                                timestring = gr.Textbox(label='Timestring',  placeholder='timestring', lines=1, value='')#timestring
                                resume_from_timestring = gr.Checkbox(label='Resume from Timestring', value=False, visible=True)#resume_from_timestring
                                resume_timestring = gr.Textbox(label='Resume from:',  placeholder='20220829210106', lines=1, value='20220829')
                                save_grid = gr.Checkbox(label='Save Grid', value=False, visible=True)#save_grid
                                make_grid = gr.Checkbox(label='Make Grid', value=False, visible=False)#make_grid
                                save_samples = gr.Checkbox(label='Save Samples', value=True, visible=False)#save_samples
                                display_samples = gr.Checkbox(label='Display Samples', value=False, visible=False)#display_samples
                                n_batch = gr.Slider(minimum=1, maximum=25, step=1, label='Number of Batches', value=1, visible=False)#n_batch
                                n_samples = gr.Slider(minimum=1, maximum=4, step=1, label='Samples (keep on 1)', value=1, visible=False)#n_samples
                                ddim_eta = gr.Slider(minimum=0, maximum=1.0, step=0.1, label='DDIM ETA', value=0.0)#ddim_eta
                                strength = gr.Slider(minimum=0, maximum=1, step=0.1, label='Init Image Strength', value=0.5)#strength
                                resume_timestring = gr.Textbox(label='Resume from:',  placeholder='20220829210106', lines=1, value='20220829')
        with gr.TabItem('Batch Prompts'):
            with gr.Row():
                with gr.Column():
                    b_init_img_array = gr.Image(visible=False)

                    b_sampler = gr.Radio(label='Sampler',
                                        choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                        value='klms',
                                        interactive=True)#sampler
                    b_prompts = gr.Textbox(label='Prompts',
                                                    placeholder='a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation',
                                                    lines=5)#animation_prompts
                    b_seed_behavior = gr.Dropdown(label='Seed Behavior', choices=['iter', 'fixed', 'random'], value='iter', interactive=True)#seed_behavior
                    b_seed = gr.Number(label='Seed',  placeholder='SEED HERE', value='-1', interactive=True)#seed
                    b_save_settings = gr.Checkbox(label='Save Settings', value=True, visible=True)#save_settings
                    b_save_samples = gr.Checkbox(label='Save Samples', value=True, visible=True)#save_samples
                    b_n_batch = gr.Slider(minimum=1, maximum=25, step=1, label='Number of Batches', value=1, visible=True)#n_batch
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
                    batch_outputs = gr.Gallery()
                    b_GFPGAN = gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False)
                    b_bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                    b_upscale = gr.Slider(minimum=1, maximum=8, step=1, label='Upscaler, 1 to turn off', value=1, interactive=True)
                    b_W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                    b_H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height
                    b_steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100, interactive=True)#steps
                    b_scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11, interactive=True)#scale
                    b_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim', interactive=True)#batch_name
                    b_outdir = gr.Textbox(label='Output Dir',  placeholder='/content/', lines=1, value='/gdrive/MyDrive/sd_anims', interactive=True)#outdir
                    batch_btn = gr.Button('Generate')
        with gr.TabItem('InPainting'):
            with gr.Row():
                with gr.Column():

                    refresh_btn = gr.Button('Refresh')
                    inPaint = gr.Image(value=inPaint, source="upload", interactive=True,
                                                                      type="pil", tool="sketch", visible=True,
                                                                      elem_id="mask")
                    i_animation_prompts = gr.Textbox(label='Prompts',
                                                    placeholder='a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation',
                                                    lines=1)#animation_prompts
                    inPaint_btn = gr.Button('Generate')
                    i_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='Init Image Strength', value=0.00, interactive=True)#strength
                    i_batch_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim', interactive=True)#batch_name
                    i_outdir = gr.Textbox(label='Output Dir',  placeholder='/content/', lines=1, value='/gdrive/MyDrive/sd_anims/', interactive=True)#outdir
                    use_mask = gr.Checkbox(label='Use Mask Path', value=True, visible=False) #@param {type:"boolean"}
                    mask_file = gr.Textbox(label='Mask File', placeholder='https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg', interactive=True) #@param {type:"string"}
                    with gr.Row():
                        i_use_init = gr.Checkbox(label='use_init', value=True, visible=False)
                        i_init_image = gr.Textbox(label='Init Image link',  placeholder='https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg', lines=1)#init_image
                with gr.Column():
                    inPainted = gr.Gallery()
                    i_sampler = gr.Radio(label='Sampler',
                                     choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                     value='klms', interactive=True)#sampler
                    with gr.Row():
                        i_GFPGAN = gr.Checkbox(label='GFPGAN, Upscaler', value=False)
                        i_bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                        i_upscale = gr.Slider(minimum=1, maximum=8, step=1, label='Upscaler, 1 to turn off', value=1, interactive=True)
                    with gr.Row():
                        i_W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                        i_H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height
                    i_steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100, interactive=True)#steps
                    i_scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11, interactive=True)#scale
                    invert_mask = gr.Checkbox(label='Invert Mask', value=True, interactive=True) #@param {type:"boolean"}
                    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                    with gr.Row():

                        mask_brightness_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Brightness', value=1.0, interactive=True)
                        mask_contrast_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Contrast', value=1.0, interactive=True)
                    #
                    i_animation_mode = gr.Dropdown(label='Animation Mode',
                                                      choices=['None', '2D', '3D', 'Video Input', 'Interpolation'],
                                                      value='None',
                                                      visible=False)#animation_mode
                    i_max_frames = gr.Slider(minimum=1, maximum=1, step=1, label='Steps', value=1, visible=False)#inpaint_frames=0
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
                    var_outdir = gr.Textbox(label='Output Folder',  value='/gdrive/MyDrive/variations', lines=1)
                    v_ddim_eta = gr.Slider(minimum=0, maximum=1, step=0.01, label='DDIM ETA', value=1.0, interactive=True)#scale
                    with gr.Row():

                        v_cfg_scale = gr.Slider(minimum=0, maximum=25, step=0.1, label='Cfg Scale', value=3.0, interactive=True)#scale
                        v_steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100, interactive=True)#steps

                    with gr.Row():
                        v_W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                        v_H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height

                    var_btn = gr.Button('Variations')
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

    var_inputs = [input_var, var_outdir, var_samples, var_plms, v_cfg_scale, v_steps, v_W, v_H, v_ddim_eta, v_GFPGAN, v_bg_upsampling, v_upscale]
    var_outputs = [output_var]

    soup_inputs = [input_prompt]
    soup_outputs = [output_prompt]





    anim_func = anim
    anim_inputs = [animation_mode, animation_prompts, key_frames,
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
    resume_from_timestring, resume_timestring, make_grid, inPaint, b_use_mask,
    b_mask_file, invert_mask, mask_brightness_adjust, mask_contrast_adjust]


    batch_inputs = [b_prompts, b_name, b_outdir, b_GFPGAN, b_bg_upsampling, b_upscale, b_W, b_H, b_steps, b_scale, b_seed_behavior, b_seed, b_sampler, b_save_grid, b_save_settings, b_save_samples, b_n_batch, b_n_samples, b_ddim_eta, b_use_init, b_init_image, b_strength, b_make_grid, b_init_img_array]

    mask_inputs = [i_animation_mode, i_animation_prompts, key_frames,
    prompts, i_batch_name, i_outdir, i_max_frames, i_GFPGAN,
    i_bg_upsampling, i_upscale, i_W, i_H, i_steps, i_scale,
    angle, zoom, translation_x, translation_y, translation_z,
    rotation_3d_x, rotation_3d_y, rotation_3d_z, use_depth_warping,
    midas_weight, near_plane, far_plane, fov, padding_mode,
    sampling_mode, seed_behavior, seed, interp_spline, noise_schedule,
    strength_schedule, contrast_schedule, sampler, extract_nth_frame,
    interpolate_x_frames, border, color_coherence, previous_frame_noise,
    previous_frame_strength, video_init_path, save_grid, save_settings,
    save_samples, display_samples, n_batch, n_samples, ddim_eta,
    i_use_init, i_init_image, i_strength, timestring,
    resume_from_timestring, resume_timestring, make_grid, inPaint, use_mask,
    mask_file, invert_mask, mask_brightness_adjust, mask_contrast_adjust]

    anim_outputs = [mp4_paths]
    batch_outputs = [batch_outputs]
    inPaint_outputs = [inPainted]

    var_btn.click(variations, inputs=var_inputs, outputs=var_outputs)
    soup_btn.click(fn=process_noodle_soup, inputs=soup_inputs, outputs=soup_outputs)

    refresh_btn.click(refresh, inputs=inPaint, outputs=inPaint)
    inPaint_btn.click(fn=anim, inputs=mask_inputs, outputs=inPaint_outputs)
    anim_btn.click(fn=anim, inputs=anim_inputs, outputs=anim_outputs)
    batch_btn.click(fn=run_batch, inputs=batch_inputs, outputs=batch_outputs)

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
            'show_error': True
        }
        #if not opt.share:
        demo.queue(concurrency_count=3)
        #if opt.share and opt.share_password:
        #    gradio_params['auth'] = ('webui', opt.share_password)

        # Check to see if Port 7860 is open
        port_status = 1
        while port_status != 0:
            try:
                self.demo.launch(**gradio_params)
            except (OSError) as e:
                print (f'Error: Port: {opt.port} is not open yet. Please wait, this may take upwards of 60 seconds...')
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

if __name__ == '__main__':
    #if opt.cli is None:
        launch_server()
    #else:
    #    run_headless()
