import gradio as gr
text = "test"
def anim(text):
    print("text")

demo = gr.Blocks()

with demo:
    gr.Markdown('Stable Diffusion 1.4 GUI v0.2')
    with gr.Tabs():
        with gr.TabItem('Animation'):
            with gr.Row():
                with gr.Column(scale=1):
                    animation_mode = gr.Dropdown(label='Animation Mode',
                                                    choices=['None', '2D', 'Video Input', 'Interpolation'],
                                                    value='2D')#animation_mode
                    animation_mode = gr.Dropdown(label='Animation Mode',
                                                                choices=['RealESRGAN_x4plus',
                                                                          'RealESRGAN_x4plus_anime_6B'],
                                                                value='RealESRGAN_x4plus',
                                                                visible=True)
                    sampler = gr.Radio(label='Sampler',
                                        choices=['klms','dpm2','dpm2_ancestral','heun','euler','euler_ancestral','plms', 'ddim'],
                                        value='klms')#sampler
                    animation_prompts = gr.Textbox(label='Prompts',
                                        placeholder='a beautiful forest by Asher Brown Durand, trending on Artstation\na beautiful city by Asher Brown Durand, trending on Artstation',
                                        lines=5)#animation_prompts
                    key_frames = gr.Checkbox(label='KeyFrames',
                                            value=True,
                                            visible=True)#key_frames
                    prompts = gr.Textbox(label='Keyframes or Prompts for batch',  placeholder='0\n5 ', lines=5, value='0\n5')#prompts


                with gr.Column(scale=2):

                        mp4_paths = gr.Video(label='Generated Video')
                        GFPGAN = gr.Checkbox(label='GFPGAN, Face Resto, Upscale', value=False)
                        bg_upsampling = gr.Checkbox(label='BG Enhancement', value=False)
                        upscale = gr.Slider(minimum=1, maximum=8, step=1, label='Upscaler, 1 to turn off', value=1)
                        W = gr.Slider(minimum=256, maximum=1024, step=64, label='Width', value=512)#width
                        H = gr.Slider(minimum=256, maximum=1024, step=64, label='Height', value=512)#height
                        steps = gr.Slider(minimum=1, maximum=300, step=1, label='Steps', value=100)#steps
                        scale = gr.Slider(minimum=1, maximum=25, step=1, label='Scale', value=11)#scale
                        batch_name = gr.Textbox(label='Batch Name',  placeholder='Batch_001', lines=1, value='SDAnim')#batch_name
                        outdir = gr.Textbox(label='Output Dir',  placeholder='/content/', lines=1, value='/gdrive/MyDrive/sd_anims/')#outdir
                        anim_btn = gr.Button('Generate')
                        #output = gr.Text()
                with gr.Column(scale=2):
                    with gr.Tab('Movements'):
                        angle = gr.Textbox(label='Angles',  placeholder='0:(0)', lines=1, value='0:(0)')#angle
                        zoom = gr.Textbox(label='Zoom',  placeholder='0: (1.04)', lines=1, value='0:(1.04)')#zoom
                        translation_x = gr.Textbox(label='Translation X (+ is Camera Left, large values [1 - 50])',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_x
                        translation_y = gr.Textbox(label='Translation Y',  placeholder='0: (0)', lines=1, value='0:(0)')#translation_y
                        translation_z = gr.Textbox(label='Translation Z',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#translation_y
                        rotation_3d_x = gr.Textbox(label='Rotation 3D X (+ is )',  placeholder='0: (0)', lines=1, value='0:(5)', visible=True)#rotation_3d_x
                        rotation_3d_y = gr.Textbox(label='Rotation 3D Y (+ is )',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_y
                        rotation_3d_z = gr.Textbox(label='Rotation 3D Z (+ is )',  placeholder='0: (0)', lines=1, value='0:(0)', visible=True)#rotation_3d_z
                        use_depth_warping = gr.Checkbox(label='Depth Warping', value=True, visible=True)#use_depth_warping
                        midas_weight = gr.Slider(minimum=0, maximum=5, step=0.1, label='Midas Weight', value=0.3, visible=True)#midas_weight
                        near_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Near Plane', value=200, visible=True)#near_plane
                        far_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Far Plane', value=1000, visible=True)#far_plane
                        fov = gr.Slider(minimum=0, maximum=360, step=1, label='FOV', value=40, visible=True)#fov
                        padding_mode = gr.Dropdown(label='Padding Mode', choices=['border', 'reflection', 'zeros'], value='border', visible=True)#padding_mode
                        sampling_mode = gr.Dropdown(label='Sampling Mode', choices=['bicubic', 'bilinear', 'nearest'], value='bicubic', visible=True)#sampling_mode
                    with gr.Tab('Video Init / Interpolation'):

                        video_init_path = gr.Textbox(label='Video init path',  placeholder='/content/video_in.mp4', lines=1)#video_init_path
                        extract_nth_frame = gr.Slider(minimum=1, maximum=100, step=1, label='Extract n-th frame', value=1)#extract_nth_frame
                        interpolate_x_frames = gr.Slider(minimum=1, maximum=25, step=1, label='Interpolate n frames', value=4)#interpolate_x_frames
                        previous_frame_noise = gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Noise', value=0.02)#previous_frame_noise
                        previous_frame_strength = gr.Slider(minimum=0.01, maximum=1.00, step=0.01, label='Prev Frame Strength', value=0.4)#previous_frame_strength

                    with gr.Tab('Anim Settings'):

                        color_coherence = gr.Dropdown(label='Color Coherence', choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'], value='Match Frame 0 RGB')#color_coherence
                        max_frames = gr.Slider(minimum=1, maximum=1000, step=1, label='Frames to render', value=100)#max_frames
                        with gr.Rows():
                            with gr.Column():
                                seed_behavior = gr.Dropdown(label='Seed Behavior', choices=['iter', 'fixed', 'random'], value='iter')#seed_behavior
                                seed = gr.Number(label='Seed',  placeholder='SEED HERE', value='-1')#seed
                                interp_spline = gr.Dropdown(label='Spline Interpolation', choices=['Linear', 'Quadratic', 'Cubic'], value='Linear')#interp_spline
                                noise_schedule = gr.Textbox(label='Noise Schedule',  placeholder='0:(0)', lines=1, value='0:(0.02)')#noise_schedule
                                strength_schedule = gr.Textbox(label='Strength_Schedule',  placeholder='0:(0)', lines=1, value='0:(0.65)')#strength_schedule
                                contrast_schedule = gr.Textbox(label='Contrast Schedule',  placeholder='0:(0)', lines=1, value='0:(1.0)')#contrast_schedule
                                border = gr.Dropdown(label='Border', choices=['wrap', 'replicate'], value='wrap')#border
                                timestring = gr.Textbox(label='Timestring',  placeholder='timestring', lines=1, value='')#timestring
                                resume_from_timestring = gr.Checkbox(label='Resume from Timestring', value=False, visible=True)#resume_from_timestring
                                resume_timestring = gr.Textbox(label='Resume from:',  placeholder='20220829210106', lines=1, value='20220829')
                                save_grid = gr.Checkbox(label='Save Grid', value=False, visible=True)#save_grid
                            with gr.Column():
                                save_settings = gr.Checkbox(label='Save Settings', value=True, visible=True)#save_settings
                                save_samples = gr.Checkbox(label='Save Samples', value=True, visible=True)#save_samples
                                display_samples = gr.Checkbox(label='Display Samples', value=False, visible=True)#display_samples
                                n_batch = gr.Slider(minimum=1, maximum=25, step=1, label='Number of Batches', value=1, visible=True)#n_batch
                                n_samples = gr.Slider(minimum=1, maximum=4, step=1, label='Samples (keep on 1)', value=1)#n_samples
                                ddim_eta = gr.Slider(minimum=0, maximum=1.0, step=0.1, label='DDIM ETA', value=0.0)#ddim_eta
                                use_init = gr.Checkbox(label='Use Init', value=False, visible=True)#use_init
                                init_image = gr.Textbox(label='Init Image link',  placeholder='https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg', lines=1)#init_image
                                strength = gr.Slider(minimum=0, maximum=1, step=0.1, label='Init Image Strength', value=0.5)#strength
                                resume_timestring = gr.Textbox(label='Resume from:',  placeholder='20220829210106', lines=1, value='20220829')
                                make_grid = gr.Checkbox(label='Make Grid', value=False, visible=True)#make_grid
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
    resume_from_timestring, resume_timestring, make_grid]
    anim_outputs = [mp4_paths]
    #print(anim_output)
    #print(anim_outputs)
    #mp4_paths.append('/gdrive/MyDrive/sd_anims/upscales/20220902151451/20220902151451.mp4')

    #print(f'orig: {mp4_paths}')
    #print(f'list: {list(mp4_paths)}')

    anim_btn.click(fn=anim, inputs=anim_inputs, outputs=anim_outputs)


demo.launch(debug=False, share=True)
