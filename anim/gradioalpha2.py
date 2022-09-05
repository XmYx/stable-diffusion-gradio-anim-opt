import json
from IPython import display
import os
import threading, asyncio
import argparse, glob, os, pathlib, subprocess, sys, time
import cv2
import numpy as np
import pandas as pd
import random
import requests
import pynvml
import shutil
import torch
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
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from types import SimpleNamespace
from torch import autocast
import gradio as gr


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help="choose which GPU to use if you have multiple", default=0)
parser.add_argument("--cli", type=str, help="don't launch web server, take Python function kwargs from this file.", default=None)
opt = parser.parse_args()


sys.path.append('./src/taming-transformers')
sys.path.append('./src/clip')
sys.path.append('./stable-diffusion/')
sys.path.append('./k-diffusion')

from helpers import save_samples, sampler_fn
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser


models_path = "/gdrive/MyDrive/" #@param {type:"string"}
output_path = "/content/output" #@param {type:"string"}

#@markdown **Google Drive Path Variables (Optional)**
mount_google_drive = False #@param {type:"boolean"}
force_remount = False
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

def torch_gc():
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

def make_callback(sampler, dynamic_threshold=None, static_threshold=None):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1*static_threshold, static_threshold)

    if sampler in ["plms","ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

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


model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
model_checkpoint =  "model.ckpt" #@param ["custom","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt"]
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
    ckpt_config_path = "/content/sdtest/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
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

def load_model_from_config(config, ckpt, verbose=False, device='cuda', half_precision=True):
    map_location = "cuda" #@param ["cpu", "cuda"]
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=map_location)
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

    if half_precision:
        model = model.half().to(device)
    else:
        model = model.to(device)
    model.eval()
    return model

if load_on_run_all and ckpt_valid:
    local_config = OmegaConf.load(f"{ckpt_config_path}")
    model = load_model_from_config(local_config, f"{ckpt_path}",half_precision=half_precision)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

def arger(animation_prompts, prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring):
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


def anim(animation_prompts: str, prompts: str, animation_mode: str, strength: float, max_frames: int, border: str, key_frames: bool, interp_spline: str, angle: str, zoom: str, translation_x: str, translation_y: str, color_coherence: str, previous_frame_noise: float, previous_frame_strength: float, video_init_path: str, extract_nth_frame: int, interpolate_x_frames: int, batch_name: str, outdir: str, save_grid: bool, save_settings: bool, save_samples: bool, display_samples: bool, n_samples: int, W: int, H: int, init_image: str, seed: str, sampler: str, steps: int, scale: int, ddim_eta: float, seed_behavior: str, n_batch: int, use_init: bool, timestring: str, noise_schedule: str, strength_schedule: str, contrast_schedule: str, resume_from_timestring: bool, resume_timestring: str):



    images = []

    def render_animation(args):
        print (args.prompts)
        # animations use key framed prompts
        #args.prompts = animation_prompts

        # resume animation
        start_frame = 1
        if args.resume_from_timestring:
            for tmp in os.listdir(args.outdir):
                if tmp.split("_")[0] == args.resume_timestring:
                    start_frame += 1
            start_frame = start_frame - 1

        # create output folder for the batch
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Saving animation frames to {args.outdir}")

        # save settings for the batch
        settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(args.__dict__), **dict(args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)

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

        args.n_samples = 1
        prev_sample = None
        color_match_sample = None
        images = []
        print(f'Max Frames: {args.max_frames}')
        print(f'Start Frame: {start_frame}')
        print(range(start_frame,args.max_frames))


        for frame_idx in range(start_frame,args.max_frames):
            print(f"Rendering animation frame {frame_idx} of {args.max_frames}")
            print(frame_idx)
            # resume animation
            if args.resume_from_timestring:
                path = os.path.join(args.outdir,f"{args.timestring}_{frame_idx-1:05}.png")
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                prev_sample = sample_from_cv2(img)

            # apply transforms to previous frame
            if prev_sample is not None:
                if args.key_frames:
                    angle = angle_series[frame_idx]
                    zoom = zoom_series[frame_idx]
                    translation_x = translation_x_series[frame_idx]
                    translation_y = translation_y_series[frame_idx]
                    noise = noise_schedule_series[frame_idx]
                    strength = strength_schedule_series[frame_idx]
                    contrast = contrast_schedule_series[frame_idx]
                    print(
                        f'angle: {angle}',
                        f'zoom: {zoom}',
                        f'translation_x: {translation_x}',
                        f'translation_y: {translation_y}',
                        f'noise: {noise}',
                        f'strength: {strength}',
                        f'contrast: {contrast}',
                    )
                xform = make_xform_2d(args.W, args.H, translation_x, translation_y, angle, zoom)

                # transform previous frame
                prev_img = sample_to_cv2(prev_sample)
                prev_img = cv2.warpPerspective(
                    prev_img,
                    xform,
                    (prev_img.shape[1], prev_img.shape[0]),
                    borderMode=cv2.BORDER_WRAP if args.border == 'wrap' else cv2.BORDER_REPLICATE
                )

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
                args.init_sample = noised_sample.half().to(device)
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
            results = generate(args, return_latent=False, return_sample=True)
            sample, image = results[0], results[1]
            images.append(results[1])

            filename = f"{args.timestring}_{frame_idx:05}.png"
            image.save(os.path.join(args.outdir, filename))
            if not using_vid_init:
                prev_sample = sample

            #display.clear_output(wait=True)
            #display.display(image)

            args.seed = next_seed(args)
            #return images

    def next_seed(args):
        if args.seed_behavior == 'iter':
            args.seed += 1
        elif args.seed_behavior == 'fixed':
            pass # always keep seed the same
        else:
            args.seed = random.randint(0, 2**32)
        return args.seed

    def make_xform_2d(width, height, translation_x, translation_y, angle, scale):
        center = (width // 2, height // 2)
        trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        trans_mat = np.vstack([trans_mat, [0,0,1]])
        rot_mat = np.vstack([rot_mat, [0,0,1]])
        return np.matmul(rot_mat, trans_mat)

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

    def get_inbetweens(key_frames, integer=False):
        key_frame_series = pd.Series([np.nan for a in range(args.max_frames)])

        for i, value in key_frames.items():
            key_frame_series[i] = value
        key_frame_series = key_frame_series.astype(float)

        interp_method = args.interp_spline
        if interp_method == 'Cubic' and len(key_frames.items()) <=3:
          interp_method = 'Quadratic'
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
          interp_method = 'Linear'

        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[args.max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
        key_frame_series = key_frame_series.interpolate(method=interp_method.lower(),limit_direction='both')
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    def generate(args, return_latent=False, return_sample=False, return_c=False):

        torch_gc()
        # start time after garbage collection (or before?)
        start_time = time.time()

        mem_mon = MemUsageMonitor('MemMon')
        mem_mon.start()



        seed_everything(args.seed)
        os.makedirs(args.outdir, exist_ok=True)

        if args.sampler == 'plms':
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        model_wrap = CompVisDenoiser(model)
        batch_size = args.n_samples
        prompt = args.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

        init_latent = None
        if args.init_latent is not None:
            init_latent = args.init_latent
        elif args.init_sample is not None:
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(args.init_sample))
        elif args.init_image != None and args.init_image != '':
            init_image = load_img(args.init_image, shape=(args.W, args.H)).to(device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, verbose=False)

        t_enc = int((1.0-args.strength) * args.steps)

        start_code = None
        if args.fixed_code and init_latent == None:
            start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)

        callback = make_callback(sampler=args.sampler,
                                dynamic_threshold=args.dynamic_threshold,
                                static_threshold=args.static_threshold)

        results = []
        precision_scope = autocast if args.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for prompts in data:
                        uc = None
                        if args.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        if args.init_c != None:
                            c = args.init_c

                        if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                            samples = sampler_fn(
                                c=c,
                                uc=uc,
                                args=args,
                                model_wrap=model_wrap,
                                init_latent=init_latent,
                                t_enc=t_enc,
                                device=device,
                                cb=callback)
                        else:

                            if init_latent != None:
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=args.scale,
                                                        unconditional_conditioning=uc,)
                            else:
                                if args.sampler == 'plms' or args.sampler == 'ddim':
                                    shape = [args.C, args.H // args.f, args.W // args.f]
                                    samples, _ = sampler.sample(S=args.steps,
                                                                    conditioning=c,
                                                                    batch_size=args.n_samples,
                                                                    shape=shape,
                                                                    verbose=False,
                                                                    unconditional_guidance_scale=args.scale,
                                                                    unconditional_conditioning=uc,
                                                                    eta=args.ddim_eta,
                                                                    x_T=start_code,
                                                                    img_callback=callback)

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
                            images.append(image)
                            results.append(image)
        torch_gc()
        return results

    prom = animation_prompts
    key = prompts

    new_prom = list(prom.split("\n"))
    new_key = list(key.split("\n"))

    prompts = dict(zip(new_key, new_prom))
    #animation_prompts = dict(zip(new_key, new_prom))

    print (prompts)

    #animation_mode = animation_mode
    arger(animation_prompts, prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring)
    args = SimpleNamespace(**arger(animation_prompts, prompts, animation_mode, strength, max_frames, border, key_frames, interp_spline, angle, zoom, translation_x, translation_y, color_coherence, previous_frame_noise, previous_frame_strength, video_init_path, extract_nth_frame, interpolate_x_frames, batch_name, outdir, save_grid, save_settings, save_samples, display_samples, n_samples, W, H, init_image, seed, sampler, steps, scale, ddim_eta, seed_behavior, n_batch, use_init, timestring, noise_schedule, strength_schedule, contrast_schedule, resume_from_timestring, resume_timestring))






    #prompts = [
    #    "a beautiful forest by Asher Brown Durand, trending on Artstation", #the first prompt I want
    #    "a beautiful portrait of a woman by Artgerm, trending on Artstation", #the second prompt I want
    #    #"the third prompt I don't want it I commented it with an",
    #]

    #animation_prompts = {
    #    0: "a beautiful apple, trending on Artstation",
    #    20: "a beautiful banana, trending on Artstation",
    #    30: "a beautiful coconut, trending on Artstation",
    #    40: "a beautiful durian, trending on Artstation",
    #}


    if args.animation_mode == 'None':
        args.max_frames = 1

    if args.key_frames:
        angle_series = get_inbetweens(parse_key_frames(args.angle))
        zoom_series = get_inbetweens(parse_key_frames(args.zoom))
        translation_x_series = get_inbetweens(parse_key_frames(args.translation_x))
        translation_y_series = get_inbetweens(parse_key_frames(args.translation_y))
        noise_schedule_series = get_inbetweens(parse_key_frames(args.noise_schedule))
        strength_schedule_series = get_inbetweens(parse_key_frames(args.strength_schedule))
        contrast_schedule_series = get_inbetweens(parse_key_frames(args.contrast_schedule))

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))


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

    if args.animation_mode == '2D':
        render_animation(args)
    elif args.animation_mode == 'Video Input':
        render_animation(args)
    elif args.animation_mode == 'Interpolation':
        render_animation(args)
    else:
        render_image_batch(args)

    print(angle_series)
    return images





anim = gr.Interface(
    anim,
    inputs=[
        gr.Textbox(label='Prompts',  placeholder="a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation", lines=5),
        gr.Textbox(label='Keyframes or Prompts for batch',  placeholder="0\n5 ", lines=5, value="0\n5"),
        gr.Dropdown(label='Animation Mode', choices=["None", "2D", "Video Input", "Interpolation"], value="2D"),
        gr.Slider(minimum=0, maximum=1, step=0.1, label='Max frames', value=0.5),
        gr.Slider(minimum=1, maximum=1000, step=1, label='Max frames', value=100),
        gr.Dropdown(label='Border', choices=["wrap", "replicate"], value="wrap"),
        gr.Checkbox(label='KeyFrames', value=True, visible=False),
        gr.Dropdown(label='Spline Interpolation', choices=["Linear", "Quadratic", "Cubic"], value="Linear"),
        gr.Textbox(label='Angles',  placeholder="0:(0)", lines=1, value="0:(0)"),
        gr.Textbox(label='Zoom',  placeholder="0: (1.04)", lines=1, value="0:(1.04)"),
        gr.Textbox(label='Translation X',  placeholder="0: (0)", lines=1, value="0:(0)"),
        gr.Textbox(label='Translation Y',  placeholder="0: (0)", lines=1, value="0:(0)"),
        gr.Dropdown(label='Color Coherence', choices=["None", "MatchFrame0"], value="MatchFrame0"),
        gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Noise', value=0.02),
        gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Strength', value=0.4),
        gr.Textbox(label='Video init path',  placeholder='/content/video_in.mp4', lines=1),
        gr.Slider(minimum=1, maximum=100, step=1, label='Extract n-th frame', value=1),
        gr.Slider(minimum=1, maximum=25, step=1, label='Interpolate n frames', value=4),
        gr.Textbox(label='Batch Name',  placeholder="Batch_001", lines=1, value="SDAnim"),
        gr.Textbox(label='Output Dir',  placeholder="/content/", lines=1, value='/gdrive/MyDrive/sd_anims/'),
        gr.Checkbox(label='Save Grid', value=False, visible=False),
        gr.Checkbox(label='Save Settings', value=True, visible=True),
        gr.Checkbox(label='Save Samples', value=True, visible=True),
        gr.Checkbox(label='Display Samples', value=True, visible=False),
        gr.Slider(minimum=1, maximum=4, step=1, label='Samples (keep on 1)', value=1),
        gr.Slider(minimum=256, maximum=1024, step=64, label='Width', value=512),
        gr.Slider(minimum=256, maximum=1024, step=64, label='Height', value=512),
        gr.Textbox(label='Init Image link',  placeholder="https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg", lines=5),
        gr.Number(label='Seed',  placeholder="SEED HERE", value='-1'),
        gr.Radio(label='Sampler', choices=["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"], value="klms"),
        gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100),
        gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11),
        gr.Slider(minimum=0, maximum=1.0, step=0.1, label='DDIM ETA', value=0.0),
        gr.Dropdown(label='Seed Behavior', choices=["iter", "fixed", "random"], value="iter"),
        gr.Slider(minimum=1, maximum=25, step=1, label='Number of Batches', value=1),
        gr.Checkbox(label='Use Init', value=False, visible=True),
        gr.Textbox(label='Timestring',  placeholder="timestring", lines=1, value=''),
        gr.Textbox(label='Noise Schedule',  placeholder="0:(0)", lines=1, value="0:(0.02)"),
        gr.Textbox(label='Strength_Schedule',  placeholder="0:(0)", lines=1, value="0:(0.65)"),
        gr.Textbox(label='Contrast Schedule',  placeholder="0:(0)", lines=1, value="0:(1.0)"),
        gr.Checkbox(label='Resume from Timestring', value=False, visible=True),
        gr.Textbox(label='Resume from:',  placeholder="20220829210106", lines=1, value="20220829210106"),


    ],
    outputs=[
        gr.Gallery(),
    ],
    title="Stable Diffusion Animation",
    description="",
)

demo = gr.TabbedInterface(interface_list=[anim], tab_names=["Anim"])

#demo.launch(share=True, enable_queue=True)

class ServerLauncher(threading.Thread):
    def __init__(self, demo):
        threading.Thread.__init__(self)
        self.name = 'Gradio Server Thread'
        self.demo = demo

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gradio_params = {
            'show_error': True,
            'server_name': '0.0.0.0'#,
            #'share': opt.share
        }
        #if not opt.share:
        demo.queue(concurrency_count=1)
        #if opt.share and opt.share_password:
        #    gradio_params['auth'] = ('webui', opt.share_password)
        self.demo.launch(**gradio_params)

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
