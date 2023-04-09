import gradio as gr 
import pandas as pd 

from PIL import Image
from torchkeras import plots 
from torchkeras.data import get_url_img

from pathlib import Path
from ultralytics import YOLO
import ultralytics
from ultralytics.yolo.data import utils 

model = YOLO('yolov8n.pt')


#load class_names
yaml_path = str(Path(ultralytics.__file__).parent/'datasets/coco128.yaml') 
class_names = utils.yaml_load(yaml_path)['names']

def detect(img):
    if isinstance(img,str):
        img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
    result = model.predict(source=img)
    if len(result[0].boxes.boxes)>0:
        vis = plots.plot_detection(img,boxes=result[0].boxes.boxes,
                     class_names=class_names, min_score=0.2)
    else:
        vis = img
    return vis

with gr.Blocks() as demo:

    with gr.Tab("捕捉摄像头喔"):
        input_img = gr.Image(source='webcam',type='pil')
        button = gr.Button("执行检测",variant="primary")
        
        gr.Markdown("## 预测输出")
        out_img = gr.Image(type='pil')
        
        button.click(detect,
                     inputs=input_img, 
                     outputs=out_img)
        
    with gr.Tab("输入图片链接"):
        default_url = 'https://t7.baidu.com/it/u=3601447414,1764260638&fm=193&f=GIF'
        url = gr.Textbox(value=default_url)
        button = gr.Button("执行检测",variant="primary")
        
        gr.Markdown("## 预测输出")
        out_img = gr.Image(type='pil')
        
        button.click(detect,
                     inputs=url, 
                     outputs=out_img)
        
    with gr.Tab("上传本地图片"):
        input_img = gr.Image(type='pil')
        button = gr.Button("执行检测",variant="primary")
        
        gr.Markdown("## 预测输出")
        out_img = gr.Image(type='pil')
        
        button.click(detect,
                     inputs=input_img, 
                     outputs=out_img)
        

        
gr.close_all() 
demo.queue()
demo.launch()