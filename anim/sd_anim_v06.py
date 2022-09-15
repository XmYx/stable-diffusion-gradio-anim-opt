#Imports
import json
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
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps, ImageChops
from PIL.PngImagePlugin import PngInfo
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

#Load Default Config
defaultspath = "/content/stable-diffusion-gradio-anim-opt/configs/animgui/defaults.yaml"
defaults = OmegaConf.load(defaultspath)
loaded_models = []


"""
defaults = OmegaConf.load("configs/webui/webui_streamlit.yaml")
if (os.path.exists("configs/webui/userconfig_streamlit.yaml")):
	user_defaults = OmegaConf.load("configs/webui/userconfig_streamlit.yaml");
	defaults = OmegaConf.merge(defaults, user_defaults)
defaults.general.grid_format
"""


#Definitions

def model_loader(models):
    if "Stable Diffusion 1.4" in models:
		if "sd14" not in loaded_models:
			model = load_model_from_config(defaults.configs.sd, defaults.models.sd)
			loaded_models.append("sd14")
	if "GFPGAN" in models:
		if "gfpgan" not in loaded_models:
			model = load_model_from_config(defaults.configs.gfpgan, defaults.models.gfpgan)
			loaded_models.append("gfpgan")




#UI
