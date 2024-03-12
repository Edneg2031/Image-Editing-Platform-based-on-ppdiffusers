# Image-Editing-Platform-based-on-ppdiffusers
## 创作动机

<font size=5>当前市场上存在一些图像处理的产品或服务，大部分功能十分强大，但是也让其对于新手来说，不是那么好入门，它们通常缺乏文本引导的功能和基于大模型的复杂任务处理能力。有时仅仅是想对图片进行简单的编辑，耗费的时间和难度也是不低的。这便是我们开发这款应用的动机——<font color=Blue>乘着大模型的东风，让小白也能快速上手图片编辑。</font>

同样，多引导的图像生成是当前图像编辑在实际应用中一个非常有实用价值的研究方向，可以应用于广告设计、新闻宣传、学习教育、国防军事等重要领域。同时，当前对该技术的研究仍然存在生成精度与效率的问题，特别是在文本输入有限的情况下，模型的体量和计算速度对于该技术的实用化十分关键。因此，本项目的研究具有较高的科学价值和实用价值。

## 市场情况
<font size=5>通过市场调研显示，大约有80%以上的人认为现有的Photoshop等相关图像处理软件的操作过于复杂，难以在短时间内轻松上手，70%以上的人明确表示希望有一种通过输入自定义的文本指令来引导图像处理的功能来帮助人们提高工作效率。

下面是一些市场上已有产品或服务的例子。
* 传统图像处理软件：例如Adobe Photoshop、GIMP等，这些软件提供了广泛的图像处理功能，但用户需要熟悉复杂的界面和参数设置，并且缺乏文本引导的功能。

 * 在线图像编辑工具：例如Canva、Pixlr等，这些工具通常提供一些简单的图像编辑功能，但在处理复杂任务和个性化需求方面受限较多。
 
 * 生成网络（GAN）应用：一些应用利用GAN技术提供了图像风格迁移、图像变换等功能，但用户通常需要上传和选择样本图像进行处理，不能通过文本直接进行引导和个性化控制。

 * ## 项目特色
* <font size=5>基于国产的百度PaddlePaddle平台开发，依托于PaddlePaddle框架、Paddle自然语言处理开发库以及PPDiffusers，借助现有的基础模型，根据需求功能调整模型的超参数，提升模型的精度和稳定性。同时添加了translate功能，支持中文文本引导图像变换。

* 使用Gradio将基于生成模型的图像编辑处理功能集成到一个平台上以供使用，极大简化用户的操作，包括可视化的strength和guidance_scale参数的调整，并且由于PPDiffusers的强大拓展能力，我们的平台可以根据用户需要，快速开发添加新功能。</font>

# 二、技术方案
## 总体框架
如图3所示，我们采用paddleNLP与PPDiffusers，根据不同任务的需要导入模型利用其中的模型功能完成任务，将先进的自然语言处理和计算机视觉模型借助gradio等工具集成到我们的图像处理平台中。实现通过文本引导的图像处理功能，并提供更灵活、智能化的用户体验。
以上述提到的通过文本引导图片变换风格为例，首先将文字通过paddleNLP处理，得到特征向量然后将图片一起输入到PPDiffusers中进行处理，最后得到变换风格后的图像。

<div align="center">
<img src=https://ai-studio-static-online.cdn.bcebos.com/b16e2572a914436287e81f34ec482b17b847d47f1a9e4a1aac8bf9d69038971c  width=500/>
</div>

## 自然语言处理模块
通过**paddleNLP**导入模型，为解决特定任务提供良好的文本处理能力。**paddleNLP**有着强大的API可以快速完成大部分任务，并且封装灵活，低耦合高内聚，保证用户可以通过继承和改写满足特定的数据处理需求。同样内置丰富的中文数据集，如*中文诗歌古典文集数据、中文对联数据集*等，可以针对比较特殊的中文处理任务，
## 计算机视觉模块
通过**PPDiffusers**导入模型，为解决特定任务提供良好的图像生成或编辑能力。对于图像风格变换任务来说，
我们使用**StableDiffusion**模型来进行图像变换，调整以下参数：
* 扩散步数（num_diffusion_timesteps）：通过增加或减少步数，来控制扩散过程的迭代次数。较多的步数可以生成更高质量的图像，但会增加计算时间和资源消耗。较少的步数可能会导致生成结果缺乏细节。根据任务需求和资源限制，逐步调整扩散步数以获得最佳效果。
* 扩散温度（beta_schedule）：扩散温度控制了噪声的强度，从而影响生成图像的样式和多样性。较高的扩散温度会产生更随机多样化的图像，而较低的扩散温度则会产生更加清晰和锐利的图像。
* 图像尺寸和批次大小：调整输入图像的尺寸和批次大小来控制模型的计算和内存消耗。较大的图像尺寸和较小的批次大小会产生更高质量的生成结果，但也会增加计算时间和资源占用。根据任务要求资源限制，选择合适的图像尺寸和批次大小。

同样对于特定的任务，可以采用自己的数据集进行微调，通过对提示词的调整，让用户经常输入的提示词可以被模型学习到。从而达到更好的效果，收集相关领域的对话数据，并将其作为微调数据集。然后，将这个数据集与基础模型结合，进行进一步训练。这样，模型就能够更好地理解和回答与与之相关的问题，并能够更准确地生成与相关的内容。
# 三、环境设置
* <font size=5>安装ppdiffusers</font>
* <font size=5>安装translate</font>

!pip install --upgrade pip ppdiffusers requests

# 四、程序实现
## 1. 导入必要的包
import paddle
from ppdiffusers import StableDiffusionInpaintPipeline
from ppdiffusers import StableDiffusionImg2ImgPipeline
from ppdiffusers import PaintByExamplePipeline
from ppdiffusers.utils import load_image

## 2.下载必要的模型
paddle.device.cuda.empty_cache() #释放GPU显存
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5") #下载模型
pipe.save_pretrained('./model1') # 保存模型到本地
paddle.device.cuda.empty_cache()
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
pipe.save_pretrained('./model2')
paddle.device.cuda.empty_cache()
pipe = PaintByExamplePipeline.from_pretrained("Fantasy-Studio/Paint-by-Example")
pipe.save_pretrained('./model3')
paddle.device.cuda.empty_cache()
pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
pipe.save_pretrained('./model4')
# 五、进入UI界面
<font size=5>以上步骤完成后打开左侧的main.gradio.py文件运行即可</font>
![](https://ai-studio-static-online.cdn.bcebos.com/8bb516457f68423fbcea3e1ab4209ad22eae6aebac2f4025a62d5d787d58170d)

<font size=5 color=blue>我们点击下拉框会有多个模型选择</font>


![](https://ai-studio-static-online.cdn.bcebos.com/372d66afbb3042d0963ec20de1af6bdaee14aa77657e4bdcbadbdb7fd322bca9)
<font size=5>比如我们选择第二个，然后点击加载模型，等待10s左右，左侧就会出现加载模型的信息。</font>
![](https://ai-studio-static-online.cdn.bcebos.com/d61a6b4fc5994b308e0bd87b3c0e7f394fa9691ac44b4a898763afcf418aeee1)
<font size=5>然后我们输入图片和掩码，以及想要其变换的背景，比如</font>
<font size=5>产生的结果：</font>
![](https://ai-studio-static-online.cdn.bcebos.com/d70d78abb9084f8eb273c2a958f39052818a88c687cf48f5bf5132e8f26ce924)
# 六、一些帮助
<font size=5>此外，由于jupyter不能实现交互功能，不能用鼠标画出掩码，只能自己提供，因为我们在此给出掩码获取代码：</font>
import cv2
import numpy as np
class draw:
    def __init__(self,img,pointsMax,maxnum):
        self.num=0
        self.maxnum=maxnum
        self.img = img
        self.lsPointsChoose = []
        self.tpPointsChoose = []
        self.tp=[]
        self.ls=[]
        self.pointsCount = 0
        self.count = 0
        self.pointsMax = pointsMax
    def on_mouse(self,event, x, y, flags, param):
        self.img2 = self.img.copy()  # 此行代码保证每次都重新再原图画  避免画多了
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            self.pointsCount = self.pointsCount + 1
            #print('pointsCount:', self.pointsCount)
            point1 = (x, y)
            #print(x, y)
            cv2.circle(self.img2, point1, 10, (0, 255, 0), 2)
            self.lsPointsChoose.append([x, y])  # 用于转化为darry 提取多边形ROI
            self.tpPointsChoose.append((x, y))  # 用于画点
            #print(len(self.tpPointsChoose))
            for i in range(len(self.tpPointsChoose) - 1):
                #print('i', i)
                cv2.line(self.img2, self.tpPointsChoose[i], self.tpPointsChoose[i + 1], (0, 0, 255), 2)
            # 绘制区域
            if self.pointsCount == self.pointsMax:
                self.num += 1
                self.pointsCount=0
                self.ls.append(self.lsPointsChoose)
                self.tp.append(self.tpPointsChoose)
                self.lsPointsChoose = []
                self.tpPointsChoose = []
                if self.num == self.maxnum:
                    self.ROI_byMouse()
                    self.ROI_bymouse_flag = 1
            cv2.imshow('src', self.img2)
        if event == cv2.EVENT_RBUTTONDOWN:
            print("right-mouse")
            self.pointsCount = 0
            tpPointsChoose = []
            self.lsPointsChoose = []
            #print(len(tpPointsChoose))
            for i in range(len(tpPointsChoose) - 1):
                #print('i', i)
                cv2.line(self.img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
            #cv2.imshow('src', self.img2)
    def ROI_byMouse(self):
        p = []
        for i in self.ls:
            pts = np.array([i], np.int32)
            pts = pts.reshape((-1, 1, 2))
            p.append(pts)
        mask = np.zeros(self.img.shape, np.uint8)
        mask = cv2.polylines(mask, p, True, (255, 255, 255))
        # 绘制多边形
        self.mask2 = cv2.fillPoly(mask, p, (255, 255, 255))
        cv2.imshow('mask', self.mask2)
        cv2.imwrite(r'E:\Pycharm\data\MNIST\112.png', self.mask2) #保存掩码
        self.ROI = cv2.bitwise_and(self.mask2, self.img)
    def diffuser(self):
        self.ROI = self.img.copy()
        cv2.namedWindow('src')
        cv2.setMouseCallback('src', self.on_mouse)
        cv2.imshow('src', self.img)
        cv2.waitKey(0)

img = cv2.imread(r'E:\Pycharm\data\MNIST\12.jpg') #读入图片

a,b = 40,1
i = draw(img,a,b) # a表示点的点数，b表示抠图的数量
i.diffuser()
