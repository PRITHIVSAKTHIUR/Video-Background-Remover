import spaces
import gradio as gr
import cv2
import numpy as np
import time
import random
from PIL import Image
import torch

torch.jit.script = lambda f: f

from transparent_background import Remover

@spaces.GPU()
def doo(video, mode, progress=gr.Progress()):
    
    if mode == 'Fast':
        remover = Remover(mode='fast')
    else:
        remover = Remover()

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames
    writer = None
    tmpname = random.randint(111111111, 999999999)
    processed_frames = 0
    start_time = time.time()
    

    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break

        if time.time() - start_time >= 20 * 60 - 5:
            print("GPU Timing Out")
            cap.release()
            writer.release()
            return str(tmpname) + '.mp4'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).convert('RGB')
        

        if writer is None:
            writer = cv2.VideoWriter(str(tmpname) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), img.size)

        processed_frames += 1
        print(f"Processing frame {processed_frames}")
        progress(processed_frames / total_frames, desc=f"Processing frame {processed_frames}/{total_frames}")
        out = remover.process(img, type='green')
        writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))       

    cap.release()
    writer.release()
    return str(tmpname) + '.mp4'
    


description="Bigger the file size, Longer the time takes. May got GPU timeout ( Abort / Error )"
examples = [['./input2.mp4'],['./input.mp4']]

iface = gr.Interface(
    fn=doo,
    inputs=["video", gr.components.Radio(['Standard', 'Quick'], label='Select mode', value='Normal', info='Standard is more accurate but takes longer⏪, while quick is quicker but less accurate.⏩')],
    outputs="video",
    examples=examples,
    description=description
)
iface.launch()
