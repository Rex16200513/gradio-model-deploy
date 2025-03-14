import gradio as gr
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import requests
import os  

# **获取 ImageNet 1000 个类别的标签**
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS = json.loads(requests.get(LABELS_URL).text)

# **修正 ResNet18 加载方式**
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# **图像预处理**
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 修正图片尺寸，避免 PIL 兼容性问题
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# **预测函数**
def predict_image(image):
    try:
        image = transform(image).unsqueeze(0)  # 处理图片
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
        
        return f"预测类别: {LABELS[predicted_idx]} (置信度: {probabilities[predicted_idx]:.2f})"
    except Exception as e:
        return f"预测错误: {str(e)}"  # 捕获错误并返回

# **获取 Render 自动分配的端口**
PORT = int(os.getenv("PORT", 8080))  # 8080 更稳定

# **Gradio 服务器绑定 0.0.0.0**
iface = gr.Interface(fn=predict_image, inputs=gr.Image(type="pil"), outputs="text")
iface.launch(server_name="0.0.0.0", server_port=PORT, debug=True)
