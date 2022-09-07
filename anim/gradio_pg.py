import gradio as gr
import os
from sorcery import dict_of

def create_dict(*args):
    return dict(((k, eval(k)) for k in args))

def save_json(dict, outdir)
    os.makedirs(outdir, exist_ok=True)
    fp = f'{outdir}/config.json'
    with open(fp, "w") as write_file
        json.dump(developer, write_file, indent=4)

def save_settings(*args):
    settings = create_dict(*args)
    save_json(settings, 'config')

def load_settings(config)
    if config = ""
        config = '/gdrive/MyDrive/configs/default.json'
        if not config.exists():
            print("No default config found, please save as Default first")
    with open(config) as f:
        settings = json.load(f)

demo = gr.Blocks()

def pingpong(img):
    print(img)

with demo:
    with gr.Column():
        input = gr.Gallery()
        test_1 = gr.Textbox(label='label')
        output = gr.Gallery()
        btn = gr.Button()
        save_btn = gr.Button(label='save')
        load_btn = gr.Button(label='load')

save_in = [test_1]
save_out = []

load_in = []
load_out = [test_1]

inputs = [input]
outputs = [output]

save_btn.click(create_dict, save_in, save_out)
btn.click(pingpong, inputs, outputs)

demo.launch(debug=True, share=True)
