import os
import argparse
from pathlib import Path
from pathlib import Path

from ultralytics import YOLO
from PIL import Image

# 获取当前脚本的绝对路径
script_path = Path(__file__).absolute()
# 获取脚本所在目录
script_dir = script_path.parent

# 切换到脚本所在目录
os.chdir(script_dir)

# 加载预训练的 YOLOv8 模型
model = YOLO('./yolov8n.pt')  # 加载官方预训练模型

# 设置为评估模式
model.eval()

# 使用模型进行预测
image_path = './front_short_camera_record.jpg'
results = model.predict(source=image_path, save=True, project='../outputs/infer_single_image_results', ) 
