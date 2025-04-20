#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8模型导出脚本
支持导出为ONNX、TorchScript、TensorRT等格式
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def export_yolov8(
    model_path,
    format="onnx",
    half=False,
    imgsz=640,
    batch=1,
    device="cpu",
    simplify=True,
    output_dir=None
):
    """
    导出YOLOv8模型为其他格式
    
    参数:
        model_path (str): YOLOv8模型路径 (.pt 文件)
        format (str): 导出格式，可选 'onnx', 'openvino', 'engine', 'coreml', 
                     'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs',
                     'paddle', 'torchscript'
        half (bool): 是否使用FP16精度
        imgsz (int): 输入图像大小
        batch (int): 批处理大小
        device (str): 设备 ('cpu' 或 '0' 或 '0,1,2,3' 等)
        simplify (bool): 是否简化ONNX模型
        output_dir (str): 输出目录，默认为模型所在目录
    
    返回:
        str: 导出模型的路径
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 设置导出目录
    if output_dir is None:
        output_dir = Path(model_path).parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # 导出模型
    path = model.export(
        format=format,
        half=half,
        imgsz=imgsz,
        batch=batch,
        device=device,
        simplify=simplify,
        nms=True,
        opset=12,  # ONNX 算子集版本
    )
    
    print(f"模型已导出为 {format} 格式，保存在: {path}")
    return path


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="导出YOLOv8模型为各种格式")
    parser.add_argument("--model", 
                        type=str, 
                        default="yolov8n.pt",
                        required=False, 
                        help="YOLOv8模型路径 (.pt 文件)")
    parser.add_argument(
        "--format", 
        type=str, 
        default="onnx", 
        choices=[
            "onnx", "openvino", "engine", "coreml",
            "saved_model", "pb", "tflite", "edgetpu", "tfjs",
            "paddle", "torchscript"
        ],
        help="导出格式"
    )
    parser.add_argument("--half", action="store_true", help="是否使用FP16精度")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小")
    parser.add_argument("--batch", type=int, default=1, help="批处理大小")
    parser.add_argument("--device", type=str, default="cpu", help="设备 ('cpu' 或 '0' 或 '0,1,2,3' 等)")
    parser.add_argument("--simplify", action="store_true", help="是否简化ONNX模型")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，默认为模型所在目录")
    
    args = parser.parse_args()
    
    # 导出模型
    export_yolov8(
        model_path=args.model,
        format=args.format,
        half=args.half,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        simplify=args.simplify,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
