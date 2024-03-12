import gradio as gr
import paddle
import requests
import json
from ppdiffusers import StableDiffusionInpaintPipeline
from ppdiffusers import StableDiffusionImg2ImgPipeline
from ppdiffusers import PaintByExamplePipeline
from ppdiffusers import VersatileDiffusionDualGuidedPipeline
from ppdiffusers.utils import load_image

pipe = None  #全局变量，用于模型的导入
API_KEY = "j5HodGgjG2iQ87MenXrw2hot"
SECRET_KEY = "Ea1AYc1kjzv2MNExEZeMAEwzanDDlsdK"
def select_model(choice):#用于选择模型
    global pipe
    pipe = None
    paddle.device.cuda.empty_cache() #释放GPU显存
    if choice == '文本引导图片风格变化':
        pipe1 = StableDiffusionImg2ImgPipeline.from_pretrained("/home/aistudio/model1")
    elif choice == '文本引导图像区域生成':
        pipe1 = StableDiffusionInpaintPipeline.from_pretrained("/home/aistudio/model2")
    elif choice == '文图引导图像区域生成':
        pipe1 = PaintByExamplePipeline.from_pretrained("/home/aistudio/model3")
    elif choice == '文图双引导生成':
        pipe1 = VersatileDiffusionDualGuidedPipeline.from_pretrained("/home/aistudio/model4")
        pipe1.remove_unused_weights()
    pipe = pipe1
    return pipe1  #用于输出导入的模型相关信息

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def translation(content):  #可接受中文prompt 转换成英文
    url = "https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=" + get_access_token()
    payload = json.dumps({
        "from": "zh",
        "to": "en",
        "q": content
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    i = response.text.find("dst")+6
    j = response.text.find("src")-2
    return response.text[i:j]


def diffusion_image(image,text,strength,guidance_scale,choice):
    global pipe
    text = translation(text)
    if choice == '文本引导图片风格变化':
        image = load_image(image).resize((768, 512))
        with paddle.amp.auto_cast(True): # 使用fp16加快生成速度
            img = pipe(prompt=text, image=image, strength=strength, guidance_scale=guidance_scale).images[0]
    elif choice == '文本引导图像区域生成':
        init_image = load_image(image).resize((512, 512))
        mask_image = load_image(strength).resize((512, 512))
        with paddle.amp.auto_cast(True): # 使用fp16加快生成速度
            img = pipe(prompt=text, image=init_image, mask_image=mask_image).images[0]
    elif choice == '文图引导图像区域生成':
        init_image = load_image(image).resize((512, 512))
        mask_image = load_image(strength).resize((512, 512))
        example_image = load_image(guidance_scale).resize((512, 512))
        with paddle.amp.auto_cast(True): # 使用fp16加快生成速度
            img = pipe(image=init_image, mask_image=mask_image, example_image=example_image).images[0]
    elif choice == '文图双引导生成':
        image = load_image(image)
        with paddle.amp.auto_cast(True): # 使用fp16加快生成速度
            image = pipe(prompt=text, image=image, text_to_image_strength=strength).images[0]
    return img

options = ["文本引导图片风格变化", "文本引导图像区域生成", "文图引导图像区域生成",'文图双引导生成'] # 定义下拉选项

def select_option(choice):
    return select_model(choice)


with gr.Blocks(title='功能选择') as demo:
    gr.Markdown("# 功能介绍")
    gr.Markdown("根据提示输入，得到理想图像！")
    with gr.Row():
        pipeText = gr.Textbox(label ='模型信息')
        with gr.Tab('加载模型'):
            dropdown = gr.inputs.Dropdown(choices=options, label="选项") # 创建下拉选择框输入组件
            sub_model = gr.Button("加载模型")
            with gr.Row():
                with gr.Tab('文本引导图片风格变化'):
                    in_img_1 = gr.Image(label='输入图片',type="filepath")
                    text_1 = gr.Textbox(label="输入文字")
                    strength_1 = gr.Slider(label="强度", minimum=0, maximum=1, step=0.01,value=0.75,
                                    info="控制条件或指导文本的强度")
                    guidance_scale_1 = gr.Slider(label="权重", minimum=0, maximum=15, step=0.1,value=7.5,
                                    info="条件或指导的权重")
                    sub_img_1 = gr.Button("生成图片")
                with gr.Tab('文本引导图像区域生成'):
                    in_img_2 = gr.Image(label='输入图片',type="filepath")
                    strength_2 = gr.Image(label='输入掩码',type="filepath")
                    text_2 = gr.Textbox(label="输入文字")
                    sub_img_2 = gr.Button("生成图片")
                with gr.Tab('文图引导图像区域生成'):
                    in_img_3 = gr.Image(label='输入图片',type="filepath")
                    strength_3 = gr.Image(label='输入掩码',type="filepath")
                    guidance_scale_3 = gr.Image(label='输入目标图片',type="filepath")
                    sub_img_3 = gr.Button("生成图片")
    out_img=gr.Image(label='输出图片')
    sub_model.click(fn =select_option,inputs=dropdown,outputs=pipeText)
    sub_img_1.click(fn=diffusion_image, inputs=[in_img_1,text_1,strength_1,guidance_scale_1,dropdown], outputs=out_img)
    sub_img_2.click(fn=diffusion_image, inputs=[in_img_2,text_2,strength_2,guidance_scale_1,dropdown], outputs=out_img)
    sub_img_3.click(fn=diffusion_image, inputs=[in_img_3,text_1,strength_3,guidance_scale_3,dropdown], outputs=out_img)
demo.launch()