import gradio as gr
import os
from sorcery import dict_of
import json
from types import SimpleNamespace

args = {}

def create_dict(*args):
    return SimpleNamespace(**args)

def save_json(dictin, outdir):
    os.makedirs(outdir, exist_ok=True)
    fp = f'{outdir}/config.json'
    with open(fp, "w+") as write_file:
        json.dump(dictin, write_file, indent=4)


def save_settings(test_1, outdir):



def load_settings(config):
    if config == "":
        config = '/gdrive/MyDrive/configs/default.json'
        if not config.exists():
            print("No default config found, please save as Default first")
    with open(f'{config}/config.json') as f:
        settings = json.load(f)
    return settings.test_1

demo = gr.Blocks()

def pingpong(img):
    print(img)

with demo:
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
                        anim_btn = gr.Button('Generate')
                        with gr.Row():
                            save_cfg_btn = gr.Button('save config snapshot')
                            load_cfg_btn = gr.Button('load config snapshot')
                        cfg_snapshots = gr.Dropdown(label = 'config snapshots (loading is WIP)', choices = list1, interactive=True)
                    with gr.Column(scale=1.6):
                            mp4_paths = gr.Video(label='Generated Video')
                            with gr.Accordion("keyframe builder test"):
                                with gr.Row():
                                  kb_frame = gr.Textbox(label = 'Key', interactive = True)
                                  kb_value = gr.Textbox(label = 'Key', interactive = True)
                                  kb_btn = gr.Button('build')
                                  kb_string = gr.Textbox(label = 'Key', interactive = True)

                            #output = gr.Text()
                    with gr.Column(scale=2.5):
                        with gr.TabItem('Animation'):
                            with gr.Accordion(label = 'Render Settings', open=False):
                                sampler = gr.Radio(label='Sampler',
                                                  choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                                  value='klms', interactive=True)#sampler
                                max_frames = gr.Slider(minimum=1, maximum=2500, step=1, label='Frames to render', value=20)#max_frames
                                steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=20, interactive=True)#steps
                                scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11, interactive=True)#scale
                                W = gr.Slider(minimum=256, maximum=8192, step=64, label='Width', value=512, interactive=True)#width
                                H = gr.Slider(minimum=256, maximum=8192, step=64, label='Height', value=512, interactive=True)#height
                                with gr.Row():
                                    GFPGAN = gr.Checkbox(label='GFPGAN, Upscaler', value=False)
                                    bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                                upscale = gr.Slider(minimum=1, maximum=8, step=1, label='Upscaler, 1 to turn off', value=1, interactive=True)


                                n_batch = gr.Slider(minimum=1, maximum=25, step=1, label='Number of Batches', value=1, visible=False)#n_batch
                                n_samples = gr.Slider(minimum=1, maximum=4, step=1, label='Samples (keep on 1)', value=1, visible=False)#n_samples
                                ddim_eta = gr.Slider(minimum=0, maximum=1.0, step=0.1, label='DDIM ETA', value=0.0)#ddim_eta
                                resume_timestring = gr.Textbox(label='Resume from:',  placeholder='20220829210106', lines=1, value='', interactive = True)
                                timestring = gr.Textbox(label='Timestring',  placeholder='timestring', lines=1, value='')#timestring
                            with gr.Accordion(label = 'Animation Settings', open=False):
                                animation_mode = gr.Dropdown(label='Animation Mode',
                                                                choices=['None', '2D', '3D', 'Video Input', 'Interpolation'],
                                                                value='3D')#animation_mode
                                with gr.Row():
                                    seed_behavior = gr.Dropdown(label='Seed Behavior', choices=['iter', 'fixed', 'random'], value='iter')#seed_behavior
                                    seed = gr.Number(label='Seed',  placeholder='SEED HERE', value='-1')#seed
                                with gr.Row():
                                    interp_spline = gr.Dropdown(label='Spline Interpolation', choices=['Linear', 'Quadratic', 'Cubic'], value='Linear')#interp_spline
                                    color_coherence = gr.Dropdown(label='Color Coherence', choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'], value='Match Frame 0 RGB')#color_coherence
                                noise_schedule = gr.Textbox(label='Noise Schedule',  placeholder='0:(0)', lines=1, value='0:(0.02)')#noise_schedule
                                strength_schedule = gr.Textbox(label='Strength_Schedule',  placeholder='0:(0)', lines=1, value='0:(0.65)')#strength_schedule
                                contrast_schedule = gr.Textbox(label='Contrast Schedule',  placeholder='0:(0)', lines=1, value='0:(1.0)')#contrast_schedule
                                border = gr.Dropdown(label='Border', choices=['wrap', 'replicate'], value='wrap')#border

                            with gr.Accordion(label = 'Movements', open=False):
                                with gr.Column(scale=0.1):
                                    angle = gr.Textbox(label='Angles',  placeholder='0:(0)', lines=1, value='0:(0)')#angle
                                    zoom = gr.Textbox(label='Zoom',  placeholder='0: (1.04)', lines=1, value='0:(1.0)')#zoom
                                    translation_x = gr.Textbox(label='Translation X (+ is Camera Left, large values [1 - 50])',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_x
                                    translation_y = gr.Textbox(label='Translation Y + = R',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_y
                                    translation_z = gr.Textbox(label='Translation Z + = FW',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#translation_y
                                with gr.Column(scale=0.1):
                                    rotation_3d_x = gr.Textbox(label='Rotation 3D X (+ is Up)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_x
                                    rotation_3d_y = gr.Textbox(label='Rotation 3D Y (+ is Right)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_y
                                    rotation_3d_z = gr.Textbox(label='Rotation 3D Z (+ is Clockwise)',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_z
                                    midas_weight = gr.Slider(minimum=0, maximum=5, step=0.1, label='Midas Weight', value=0.3, visible=True)#midas_weight
                            with gr.Accordion('3D Settings', open=False):
                                use_depth_warping = gr.Checkbox(label='Depth Warping', value=True, visible=True)#use_depth_warping
                                near_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Near Plane', value=200, visible=True)#near_plane
                                far_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Far Plane', value=1000, visible=True)#far_plane
                                fov = gr.Slider(minimum=0, maximum=360, step=1, label='FOV', value=40, visible=True)#fov
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

            with gr.TabItem('Animation Director'):
                with gr.Column():
                    add_cfg_btn = gr.Button('add config snapshot to sequence')
                    cfg_seq_snapshots = gr.Dropdown(label = 'select snapshot to add', choices = list2, interactive=True)
                with gr.Column():
                    sequence = gr.Textbox(label='sequence', lines = 10, interactive=True)

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
                        with gr.Row():
                          b_mask_brightness_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Brightness', value=1.0, interactive=True)
                          b_mask_contrast_adjust = gr.Slider(minimum=0, maximum=2, step=0.1, label='Mask Contrast', value=1.0, interactive=True)
                        b_invert_mask = gr.Checkbox(label='Invert Mask', value=True, interactive=True) #@param {type:"boolean"}

            with gr.TabItem('InPainting'):
                with gr.Row():
                    with gr.Column():

                        refresh_btn = gr.Button('Refresh')
                        i_init_img_array = gr.Image(value=inPaint, source="upload", interactive=True,
                                                                          type="pil", tool="sketch", visible=True,
                                                                          elem_id="mask")
                        i_prompts = gr.Textbox(label='Prompts',
                                    placeholder='a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation',
                                    lines=1)#animation_prompts
                        inPaint_btn = gr.Button('Generate')
                        i_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='Init Image Strength', value=0.00, interactive=True)#strength
                        i_batch_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim', interactive=True)#batch_name
                        i_outdir = gr.Textbox(label='Output Dir',  placeholder='/content/', lines=1, value='/gdrive/MyDrive/sd_anims/', interactive=True)#outdir
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
                            i_n_batch = gr.Slider(minimum=1, maximum=20, step=1, label='Batches', value=1, visible=True)




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
demo.launch(debug=True, share=True)
