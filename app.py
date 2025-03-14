import gradio as gr
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import requests
import os  # 获取 Render 提供的端口号

# 获取 ImageNet 1000 个类别的标签
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS = json.loads(requests.get(LABELS_URL).text)

# 加载预训练的 ResNet18 模型（修正 deprecated `pretrained=True`）
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.eval()

# 图像预处理（与训练时相同）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预测函数
def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
    
    return f"预测类别: {LABELS[predicted_idx]} (置信度: {probabilities[predicted_idx]:.2f})"

# **获取 Render 自动分配的端口**
PORT = int(os.getenv("PORT", 8080))  # 默认为 8080，防止 Render 没有设置 PORT 变量

# Gradio 接口
iface = gr.Interface(fn=predict_image, inputs=gr.Image(type="pil"), outputs="text")

# **修正端口问题**
iface.launch(server_name="0.0.0.0", server_port=PORT)
