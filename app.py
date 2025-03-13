import gradio as gr
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 加载预训练的ResNet18模型
model = torchvision.models.resnet18(pretrained=True)
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
    # 预处理图片
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        # 通过模型预测
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    # 获取预测结果的标签
    labels = torchvision.datasets.CIFAR100(root='.', download=True).classes
    return f"预测类别: {labels[predicted_idx]} (置信度: {probabilities[predicted_idx]:.2f})"


# Gradio 接口
iface = gr.Interface(fn=predict_image, inputs=gr.Image(type="pil"), outputs="text")
iface.launch()
