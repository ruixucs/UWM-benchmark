"""
评估BAGEL模型在Smart Watch Benchmark上的性能
用于测试统一视觉语言模型的泛化能力
"""

import argparse
import json
import os
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np

# 导入BAGEL相关模块
import sys
sys.path.append('..')
from inferencer import load_model_and_tokenizer, build_transform
from data.transforms import get_transform_config


class SmartWatchEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # 加载BAGEL模型
        print("Loading BAGEL model...")
        self.model, self.tokenizer, self.new_token_ids = load_model_and_tokenizer(args)
        self.image_transform = build_transform()
        
        print(f'Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B')
        
        # 加载数据集
        print(f"Loading dataset from {args.data_path}...")
        with open(args.data_path, 'r') as f:
            self.data = json.load(f)
        
        # 根据任务类型过滤数据
        if args.task_type == 'vqa':
            self.data = [d for d in self.data if d['task'] in ['vqa', 'caption']]
        elif args.task_type == 'generation':
            self.data = [d for d in self.data if d['task'] == 'generation']
        
        print(f"Loaded {len(self.data)} samples for {args.task_type} task")
        
        # 结果存储
        self.results = []
        
    def run_evaluation(self):
        """运行完整评估流程"""
        if self.args.task_type == 'vqa':
            self.evaluate_vqa()
            self.compute_vqa_metrics()
        elif self.args.task_type == 'generation':
            self.evaluate_generation()
            self.compute_generation_metrics()
        else:  # all
            self.evaluate_vqa()
            self.compute_vqa_metrics()
            self.evaluate_generation()
            self.compute_generation_metrics()
    
    def evaluate_vqa(self):
        """评估VQA任务"""
        print("\n" + "="*50)
        print("Evaluating VQA Task")
        print("="*50)
        
        vqa_results = []
        
        for idx, sample in enumerate(tqdm(self.data, desc="VQA Inference")):
            if sample['task'] not in ['vqa', 'caption']:
                continue
            
            # 限制评估样本数量（可选）
            if self.args.max_samples > 0 and idx >= self.args.max_samples:
                break
            
            # 加载图像
            image_path = os.path.join(self.args.image_folder, sample['image'])
            image = Image.open(image_path).convert('RGB')
            
            # 构建问题（移除<image>标记，BAGEL有自己的处理方式）
            question = sample['conversations'][0]['value'].replace('<image>\n', '').replace('<image>', '')
            ground_truth = sample['conversations'][1]['value']
            
            # 使用BAGEL进行推理
            try:
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    response = self.model.chat(
                        self.tokenizer,
                        self.new_token_ids,
                        self.image_transform,
                        images=[image],
                        prompt=question,
                        max_length=128,
                        do_sample=False,
                        temperature=0.0,
                    )
            except Exception as e:
                print(f"Error in inference for sample {idx}: {e}")
                response = ""
            
            # 保存结果
            result = {
                'image': sample['image'],
                'question': question,
                'ground_truth': ground_truth,
                'prediction': response,
                'task': sample['task']
            }
            vqa_results.append(result)
        
        # 保存VQA结果
        vqa_output_file = os.path.join(self.args.output_dir, 'vqa_results.json')
        with open(vqa_output_file, 'w') as f:
            json.dump(vqa_results, f, indent=2, ensure_ascii=False)
        
        print(f"VQA results saved to {vqa_output_file}")
        self.vqa_results = vqa_results
    
    def compute_vqa_metrics(self):
        """计算VQA评估指标"""
        print("\n" + "="*50)
        print("Computing VQA Metrics")
        print("="*50)
        
        # 初始化统计
        stats = {
            'time': {'count': 0, 'score': 0},
            'weather': {'count': 0, 'score': 0},
            'position': {'count': 0, 'score': 0},
            'battery': {'count': 0, 'score': 0}
        }
        
        for result in self.vqa_results:
            ground_truth = result['ground_truth']
            answer = result['prediction']
            
            # 根据ground_truth格式判断问题类型
            if ':' in ground_truth and ' ' not in ground_truth:
                # 时间类问题
                stats['time']['count'] += 1
                pattern = r"(\d{2}):(\d{2}):(\d{2})"
                gt_match = re.search(pattern, ground_truth)
                ans_match = re.search(pattern, answer)
                
                if gt_match and ans_match:
                    gt_h, gt_m, gt_s = int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3))
                    ans_h, ans_m, ans_s = int(ans_match.group(1)), int(ans_match.group(2)), int(ans_match.group(3))
                    
                    err_h = min(abs(ans_h - gt_h), 12 - abs(ans_h - gt_h))
                    err_m = min(abs(ans_m - gt_m), 60 - abs(ans_m - gt_m))
                    err_s = min(abs(ans_s - gt_s), 60 - abs(ans_s - gt_s))
                    err = (err_h / 6.0 + err_m / 30.0 + err_s / 30.0) / 3.0
                    
                    stats['time']['score'] += 1 - err
                    
            elif 'sunny' in ground_truth or 'raining' in ground_truth or 'cloudy' in ground_truth:
                # 天气类问题
                stats['weather']['count'] += 1
                if ground_truth.lower() in answer.lower():
                    stats['weather']['score'] += 1
                    
            elif '-' in ground_truth and 'left' in ground_truth or 'right' in ground_truth:
                # 位置类问题
                stats['position']['count'] += 1
                if ground_truth in answer:
                    stats['position']['score'] += 1
                    
            elif '%' in ground_truth:
                # 电量类问题
                stats['battery']['count'] += 1
                gt_match = re.search(r'\b(100|[1-9]\d?|0)%', ground_truth)
                ans_match = re.search(r'\b(100|[1-9]\d?|0)%', answer)
                
                if gt_match and ans_match:
                    gt = int(gt_match.group(1)) / 100
                    ans = int(ans_match.group(1)) / 100
                    err = abs(ans - gt)
                    stats['battery']['score'] += 1 - err
        
        # 计算准确率
        metrics = {}
        for category in stats:
            if stats[category]['count'] > 0:
                acc = stats[category]['score'] / stats[category]['count']
                metrics[f'{category}_acc'] = acc
                print(f"{category.capitalize()} Accuracy: {acc:.4f} ({stats[category]['score']:.2f}/{stats[category]['count']})")
        
        # 计算总体准确率
        total_score = sum(stats[c]['score'] for c in stats)
        total_count = sum(stats[c]['count'] for c in stats)
        if total_count > 0:
            total_acc = total_score / total_count
            metrics['total_acc'] = total_acc
            print(f"\nTotal Accuracy: {total_acc:.4f}")
        
        # 保存指标
        metrics_file = os.path.join(self.args.output_dir, 'vqa_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_file}")
        return metrics
    
    def evaluate_generation(self):
        """评估图像生成任务"""
        print("\n" + "="*50)
        print("Evaluating Image Generation Task")
        print("="*50)
        print("Note: Image generation evaluation is not yet fully supported.")
        print("BAGEL's generation capability requires specific prompts and decoding.")
        print("This is a placeholder for future implementation.")
        
        # TODO: 实现图像生成评估
        # BAGEL的图像生成需要特殊的prompt格式和解码流程
        # 这需要参考BAGEL的图像生成示例代码
        
    def compute_generation_metrics(self):
        """计算图像生成指标（FID等）"""
        print("\n图像生成指标计算将在未来版本中实现")
        # TODO: 实现FID计算


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BAGEL on Smart Watch Benchmark")
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to BAGEL model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to smart watch test data JSON file')
    parser.add_argument('--image-folder', type=str, required=True,
                        help='Path to smart watch test images folder')
    
    # 评估参数
    parser.add_argument('--task-type', type=str, default='vqa',
                        choices=['vqa', 'generation', 'all'],
                        help='Task type to evaluate (default: vqa)')
    parser.add_argument('--max-samples', type=int, default=-1,
                        help='Maximum number of samples to evaluate (-1 for all)')
    parser.add_argument('--output-dir', type=str, default='./bagel_smartwatch_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def main():
    args = parse_args()
    
    print("="*50)
    print("BAGEL Smart Watch Benchmark Evaluation")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Task: {args.task_type}")
    print("="*50 + "\n")
    
    # 创建评估器
    evaluator = SmartWatchEvaluator(args)
    
    # 运行评估
    evaluator.run_evaluation()
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print(f"Results saved to {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()

