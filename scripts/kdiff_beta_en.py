import PIL
import gradio as gr
import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from tqdm import tqdm, trange
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
import threading, asyncio

mimetypes.init()
mimetypes.add_type('application/javascript', '.js')


import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def FACE_RESTORATION(image, bg_upsampling, upscale, mpth):
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
    model_path = os.path.join(mpth)
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
    del bg_upsampler
    del restorer
    del model
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


# class KDiffusionSampler:
    # def __init__(self, m, sampler):
        # self.model = m
        # self.model_wrap = K.external.CompVisDenoiser(m)
        # self.schedule = sampler

    # def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        # sigmas = self.model_wrap.get_sigmas(S)
        # x = x_T * sigmas[0]
        # model_wrap_cfg = CFGDenoiser(self.model_wrap)

        # samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

        # return samples_ddim, None


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./configs/stable-diffusion/v1-inference.yaml", type=str)
parser.add_argument("--ckpt", default="/gdrive/MyDrive/model.ckpt", type=str)
parser.add_argument("--precision", default="autocast", type=str)
#parser.add_argument("--outdir", default="./outputs/txt2img-samples", type=str)
parser.add_argument("--GFPGAN", default='/content/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth', type=str)

args = parser.parse_args()


config = OmegaConf.load(args.config)
model = load_model_from_config(config, args.ckpt)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.half().to(device)


def resize_image(resize_mode, im, width, height):
    print(resize_mode, width, height)
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res

def check_prompt_length(prompt, comments):
    """this function tests if prompt is too long, and if so, adds a message to comments"""

    tokenizer = model.cond_stage_model.tokenizer
    max_length = model.cond_stage_model.max_length

    info = model.cond_stage_model.tokenizer([prompt], truncation=True, max_length=max_length, return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
    ovf = info['overflowing_tokens'][0]
    overflowing_count = ovf.shape[0]
    if overflowing_count == 0:
        return

    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    overflowing_words = [vocab.get(int(x), "") for x in ovf]
    overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

    comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

def dream(prompt: str, mask_mode, init_img_arr, keep_mask, mask_blur_strength, ddim_steps: int, sampler_name: str, fixed_code: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scales: str, denoising_strength: float, seed: int, height: int, width: int, same_seed: bool, resize_mode, GFPGAN: bool, bg_upsampling: bool, upscale: int, outdir: str, precision = args.precision, GFPGANth = args.GFPGAN):


    if mask_mode == 'Mask':
        init_img = init_img_arr['image']
        init_img.save('init_img_1.png')
        init_mask = init_img_arr['mask']
        init_mask.save('init_mask_1.png')
    elif mask_mode == 'Crop':
        init_img = init_img_arr
        init_mask = None
    else:
        init_img = None
        init_mask = None






    torch.cuda.empty_cache()
    if init_img_arr != None:
        init_img = init_img.convert("RGB")
        init_img = resize_image(resize_mode, init_img, width, height)
    if init_mask != None and mask_mode == 'Mask':
        init_mask = init_mask.convert("RGB")
        init_mask = resize_image(resize_mode, init_mask, width, height)
        init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
        init_mask.save('init_mask_2.png')
    if init_img != None: back_img = init_img


    H = height
    W = width
    C = 4
    f = 8
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    rng_seed = seed_everything(seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    if sampler_name == 'k_dpm_2_a':
        sampler_name = 'dpm_2_ancestral'
    elif sampler_name == 'k_dpm_2':
        sampler_name = 'dpm_2'
    elif sampler_name == 'k_euler_a':
        sampler_name = 'euler_ancestral'
    elif sampler_name == 'k_euler':
        sampler_name = 'euler'
    elif sampler_name == 'k_heun':
        sampler_name = 'heun'
    elif sampler_name == 'k_lms':
        sampler_name = 'lms'
    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    prompt = prompt
    assert prompt is not None
    data = [batch_size * [prompt]]


    sample_path = os.path.join(outpath)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    seedit = 0

    if fixed_code and init_img == None:
        start_code = torch.randn([n_samples, C, height // f, width // f], device=device)


    if len(cfg_scales) > 1: cfg_scales = list(map(float, cfg_scales.split(' ')))
    output_images = []
    precision_scope = autocast if precision == "autocast" else nullcontext
    ff = []
    message = ''
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        with precision_scope("cuda"):
            if init_img_arr != None:
                init_img = np.array(init_img).astype(np.float32) / 255.0
                init_img = init_img[None].transpose(0, 3, 1, 2)
                init_img = torch.from_numpy(init_img)
                init_img = 2.*init_img - 1.
                init_img = init_img.to(device)
                init_img = repeat(init_img, '1 ... -> b ...', b=batch_size)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_img))  # move to latent space

                #print(f"target t_enc is {t_enc} steps")
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                        output_images.append([])

                        aaa = f'{rng_seed + seedit}{random.randint(8, 10000)}'
                        output_images[-1].append(aaa)
                        output_images[-1].append(prompts[0])

                        os.makedirs(f'{outpath}/prompts', exist_ok=True)
                        for n in trange(n_iter, desc="Sampling", disable=not accelerator.is_main_process):


                            with open(f'{outpath}/prompts/{aaa}_prompt.txt', 'w') as fff:
                                fff.write(prompts[0])

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
                                model_wrap_cfg = CFGDenoiser(model_wrap)
                                t_enc = int(denoising_strength * ddim_steps)
                                if init_img_arr == None:

                                    shape = [C, height // f, width // f]

                                    x = torch.randn([n_samples, *shape], device=device) * sigmas[0] # for GPU draw
                                else:
                                    x0 = init_latent
                                    noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]
                                    x = x0 + noise
                                    sigmas = sigmas[ddim_steps - t_enc - 1:]
                                    #sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

                                    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'



                                #extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}
                                samples_ddim = K.sampling.__dict__[f'sample_{sampler_name}'](model_wrap_cfg, x, sigmas, extra_args={'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}, disable=False)
                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = accelerator.gather(x_samples_ddim)
                                #del samples_ddim, model_wrap_cfg
                                if accelerator.is_main_process:
                                    for x_sample in x_samples_ddim:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        image = Image.fromarray(x_sample.astype(np.uint8))
                                        if isinstance(init_img_arr, dict) and init_img_arr != None:

                                            init_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
                                            init_mask = init_mask.convert('L')
                                            init_img = back_img.convert('RGB')
                                            image = image.convert('RGB')

                                            image = Image.composite(init_img, image, init_mask)

                                        output_images[-1].append(image)
                                        # aaa = output_images[-1][0]
                                        # message+= f'Запрос "{output_images[-1][1]}" находится в папке {outpath}/{aaa}/ \n'
                                        # pt = f'{outpath}/{aaa}/{cfg_scale} {random.randint(1, 200000)}.jpg'
                                        # gc.collect()
                                        # torch.cuda.empty_cache()
                                        # if GFPGAN:
                                            # (Image.fromarray(FACE_RESTORATION(output_images[-1][-1], bg_upsampling, upscale, GFPGANth).astype(np.uint8))).save(pt, format = 'JPEG', mize = True)
                                        # else:
                                            # output_images[-1][-1].save(pt, format = 'JPEG', mize = True)
                                        # #del output_images[-1][-1]
                                        # ff.append(pt)
                                        base_count += 1
                                        if not same_seed: seedit += 1
                                    #del x_samples_ddim


                toc = time.time()
                gc.collect()
                torch.cuda.empty_cache()

    for i in trange(len(output_images), desc="prompts"):
        z = 0
        aaa = output_images[i][0]
        message+= f'Запрос "{output_images[i][1]}" находится в папке {outpath}/{aaa}/ \n'
        for k in range(2, len(output_images[i])):
            if k%batch_size == 0:
                cfg=cfg_scales[z]
                if z+1> len(cfg_scales)-1:
                    z=0
                else:
                    z+=1
            pt = f'{outpath}/{aaa}_{cfg}_{random.randint(1, 200000)}.jpg'
            if GFPGAN:
                (Image.fromarray(FACE_RESTORATION(output_images[i][k], bg_upsampling, upscale, GFPGANth).astype(np.uint8))).save(pt, format = 'JPEG', mize = True)
            else:
                output_images[i][k].save(pt, format = 'JPEG', mize = True)
            ff.append(pt)

    return ff, rng_seed, message

def dev_dream(prompt: str, init_img, use_img: bool, ddim_steps: int, plms: bool, fixed_code: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scales: str, denoising_strength: float, seed: int, height: int, width: int, same_seed: bool, GFPGAN: bool, bg_upsampling: bool, upscale: int):
    prompts = list(map(str, prompt.split('|')))
    if not use_img:
        init_img=None
    f, rng_seed, message = [], [], ''
    for prompt in tqdm(prompts, desc="prompts"):
        ff, rng_seedf, messagef = dream(prompt, init_img, ddim_steps, plms, fixed_code, ddim_eta, n_iter, n_samples, cfg_scales, denoising_strength, seed, height, width, same_seed, GFPGAN, bg_upsampling, upscale)
        for i in ff:
            f.append(i)
        rng_seed=rng_seedf
        message+=messagef
    return f, rng_seed, message


image_mode = 'sketch'

def crash(e, s):
    global model
    global device

    print(s, '\n', e)

    del model
    del device

    print('exiting...calling os._exit(0)')
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()

def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

css = '[data-testid="image"] {min-height: 512px !important}'

with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
    with gr.Tabs():
        with gr.TabItem("Stable Diffusion Text-to-Image"):
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    gr.Markdown("Thank Momicro for his hard work on these files, please use this translation with joy")
                    prompt = gr.Textbox(label='Text request. Supports weighting query parts with ":number" (the space after the number is mandatory). Normal request is also supported.',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1)
                    steps = gr.Slider(minimum=1, maximum=500, step=1, label="Diffusion steps, ideal - 100.", value=50)
                    sampling = gr.Radio(label='Sampler. k_euler_a - fast, medium quality, k_dpm_2_a - slow (up to 2x), high quality', choices=['k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value='k_euler_a')
                    ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False)
                    batch_count = gr.Slider(minimum=1, maximum=50, step=1, label='How many times to generate on request (sequentially)', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='How many pictures at a time (simultaneously). EATS A LOT OF MEMORY', value=1)
                    cfg = gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scales, one or more numbers separated by spaces will generate a sequence, has to be float e.g.: 7.0', lines=1, value=9.0)
                    seed = gr.Number(label='Seed', value=-1)
                    same_seed = gr.Checkbox(label='The same seed every time. In order to generate the same thing with the same request.', value=False)
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Шidth", value=512)
                    GFPGAN = gr.Checkbox(label='GFPGAN, restores faces, can upscale. All settings are below it.', value=True)
                    bg_upsampling = gr.Checkbox(label='BG Enhancement', value=True)
                    upscale = gr.Slider(minimum=1, maximum=8, step=1, label="Upscaling. 1 = OFF.", value=2)
                    outdir =  gr.Textbox(label='Save Dir:', value = "/gdrive/My Drive/GradIO_out/")

                    btn = gr.Button("Generate")
                    placeholder_none = gr.Variable(value=None, visible=False)
                    placeholder_false = gr.Variable(value=False, visible=False)
                    placeholder_0 = gr.Variable(value=0.0, visible=False)
                with gr.Column():
                    output_txt2img_gallery = gr.Gallery(label="Images")
                    output_txt2img_seed = gr.Number(label='Seed')
                    output_txt2img_stats = gr.Textbox(label='To download the folder with the results, open the files on the left side of the colab and download the specified folders')

            btn.click(
                dream,
                [prompt, placeholder_none, placeholder_none, placeholder_false, placeholder_none, steps, sampling, placeholder_false, ddim_eta, batch_count, batch_size, cfg, placeholder_0, seed, height, width, same_seed, placeholder_none, GFPGAN, bg_upsampling, upscale, outdir],
                [output_txt2img_gallery, output_txt2img_seed, output_txt2img_stats]
            )
        with gr.TabItem("Stable Diffusion Image-to-Image"):
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    gr.Markdown("IMG2IMG with Stable Diffusion - Momicro's Fork")
                    prompt = gr.Textbox(label='Text request. Supports weighting query parts with ":number" (the space after the number is mandatory). Normal request is also supported.',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1)
                    image_editor_mode = gr.Radio(choices=["Mask", "Crop"], label="Image change mode, you may have to close the image and reload when changing", value="Crop")
                    with gr.Row():
                        painterro_btn = gr.Button("Аdvanced Editor")
                        copy_from_painterro_btn = gr.Button(value="Get image from Advanced Editor")
                    image = gr.Image(value=None, source="upload", interactive=True, type="pil", tool="select")
                    image_mask = gr.Image(value=None, source="upload", interactive=True, type="pil", tool="sketch", visible=False)
                    mask = gr.Radio(choices=["Regen Masked Area", "Keep Only Masked area"], label="Mask Mode", type="index", value="Regen Masked Area", visible=False)
                    mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1, label="How much blurry should the mask be? (to avoid hard edges)", value=3, visible=False)
                    steps = gr.Slider(minimum=1, maximum=500, step=1, label="Diffusion steps, ideal - 100.", value=50)
                    sampling = gr.Radio(label='Sampler', choices=['k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value='k_euler_a')
                    ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False)
                    batch_count = gr.Slider(minimum=1, maximum=50, step=1, label='How many times to generate on request (sequentially)', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='How many pictures at a time (simultaneously). EATS A LOT OF MEMORY', value=1)
                    cfg = gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scales, one or more numbers separated by spaces will generate a sequence, has to be float e.g.: 7.0', lines=1, value=9.0)
                    denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Freedom of AI', value=0.75)
                    seed = gr.Number(label='Seed', value=-1)
                    same_seed = gr.Checkbox(label='The same seed every time. In order to generate the same thing with the same request.', value=False)

                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Шidth", value=512)
                    resize = gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Crop and resize")
                    GFPGAN = gr.Checkbox(label='GFPGAN, restores faces, can upscale. All settings are below it.', value=True)
                    bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                    upscale = gr.Slider(minimum=1, maximum=8, step=1, label="Upscale, 1 = OFF", value=1)
                    outdir =  gr.Textbox(label='Save Dir:', value = "/gdrive/My Drive/GradIO_out/")

                    btn_mask = gr.Button("Generate").style(full_width=True)
                    btn_crop = gr.Button("Generate", visible=False).style(full_width=True)
                    placeholder_none = gr.Variable(value=None, visible=False)
                    placeholder_false = gr.Variable(value=False, visible=False)
                    placeholder_0 = gr.Variable(value=0.0, visible=False)
                with gr.Column():
                    output_img2img_gallery = gr.Gallery(label="Images")
                    output_img2img_seed = gr.Number(label='Seed')
                    output_img2img_stats = gr.Textbox(label='To download the folder with the results, open the files on the left side of the colab and download the specified folders')

            image_editor_mode.change(
                change_image_editor_mode,
                [image_editor_mode, image, resize, width, height],
                [image, image_mask, btn_crop, btn_mask, painterro_btn, copy_from_painterro_btn, mask, mask_blur_strength]
            )

            image.edit(
                update_image_mask,
                [image, resize, width, height],
                image_mask
            )
            btn_mask.click(
                dream,
                [prompt, image_editor_mode, image_mask, mask, mask_blur_strength, steps, sampling, placeholder_false, ddim_eta, batch_count, batch_size, cfg, denoising, seed, height, width, same_seed, resize, GFPGAN, bg_upsampling, upscale, outdir],
                [output_img2img_gallery, output_img2img_seed, output_img2img_stats]            )

            btn_crop.click(
                dream,
                [prompt, image_editor_mode, image, mask, mask_blur_strength, steps, sampling, placeholder_false, ddim_eta, batch_count, batch_size, cfg, denoising, seed, height, width, same_seed, resize, GFPGAN, bg_upsampling, upscale, outdir],
                [output_img2img_gallery, output_img2img_seed, output_img2img_stats]
                )

            painterro_btn.click(None, [image], None, _js="""(img) => {
                try {
                    Painterro({
                        hiddenTools: ['arrow'],
                        saveHandler: function (image, done) {
                            localStorage.setItem('painterro-image', image.asDataURL());
                            done(true);
                        },
                    }).show(Array.isArray(img) ? img[0] : img);
                } catch(e) {
                    const script = document.createElement('script');
                    script.src = 'https://unpkg.com/painterro@1.2.78/build/painterro.min.js';
                    document.head.appendChild(script);
                    const style = document.createElement('style');
                    style.appendChild(document.createTextNode('.ptro-holder-wrapper { z-index: 9999 !important; }'));
                    document.head.appendChild(style);
                }
                return [];
            }""")
            copy_from_painterro_btn.click(None, None, [image, image_mask], _js="""() => {
                const image = localStorage.getItem('painterro-image')
                return [image, image];
            }""")

#demo.queue(concurrency_count=1)
demo.launch(show_error=True, share=True)
