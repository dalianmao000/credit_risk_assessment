import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from data_exploration import load_data, explore_data, preprocess_data, save_processed_data
from model_training import (
    load_processed_data, prepare_data, train_model,
    evaluate_model, cross_validate_model, feature_importance
)
from predict import predict_single_sample, predict_batch
from config import *

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='信用风险预测系统')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['explore', 'train', 'predict'],
                      help='运行模式: explore(数据探索), train(模型训练), predict(预测)')
    parser.add_argument('--model_type', type=str, default='lightgbm',
                      choices=['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost'],
                      help='模型类型')
    parser.add_argument('--input_file', type=str, help='输入数据文件路径')
    parser.add_argument('--output_file', type=str, help='输出结果文件路径')
    return parser.parse_args()

def run_data_exploration():
    """运行数据探索流程"""
    print("\n=== 开始数据探索 ===")
    
    # 加载原始数据
    train_df = load_data(TRAIN_DATA_PATH)
    test_df = load_data(TEST_DATA_PATH)
    
    # 探索数据
    explore_data(train_df)
    
    # 预处理数据
    train_processed = preprocess_data(train_df)
    test_processed = preprocess_data(test_df)
    
    # 保存处理后的数据
    save_processed_data(train_processed, test_processed)
    
    print("\n数据探索完成！处理后的数据已保存到 processed 目录")

def run_model_training(model_type):
    """运行模型训练流程"""
    print(f"\n=== 开始训练 {model_type} 模型 ===")
    
    # 加载处理后的数据
    df = load_processed_data()
    
    # 准备数据
    X_train, X_val, y_train, y_val = prepare_data(df)
    
    # 训练模型
    model = train_model(X_train, X_val, y_train, y_val, model_type)
    
    # 评估模型
    metrics = evaluate_model(model, X_val, y_val, model_type)
    
    # 交叉验证
    cv_scores = cross_validate_model(X_train, y_train, model_type)
    
    # 分析特征重要性
    importance = feature_importance(model, model_type)
    
    print(f"\n{model_type} 模型训练完成！")
    print(f"测试集 AUC: {metrics['auc']:.4f}")
    print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

def run_prediction(model_type, input_file=None, output_file=None):
    """运行预测流程"""
    print(f"\n=== 开始使用 {model_type} 模型进行预测 ===")
    
    if input_file:
        # 从文件加载数据
        data = pd.read_csv(input_file)
        results = predict_batch(data, model_type=model_type)
        
        if output_file:
            results.to_csv(output_file, index=False)
            print(f"\n预测结果已保存到: {output_file}")
        else:
            print("\n预测结果:")
            print(results)
    else:
        # 使用示例数据
        sample_data = {
            'RevolvingUtilizationOfUnsecuredLines': 0.5,
            'age': 35,
            'NumberOfTime30-59DaysPastDueNotWorse': 0,
            'DebtRatio': 0.3,
            'MonthlyIncome': 5000,
            'NumberOfOpenCreditLinesAndLoans': 5,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 2
        }
        
        result = predict_single_sample(sample_data, model_type=model_type)
        print("\n示例预测结果:")
        print(result)

def main():
    """主函数"""
    args = parse_args()
    
    if args.mode == 'explore':
        run_data_exploration()
    elif args.mode == 'train':
        run_model_training(args.model_type)
    elif args.mode == 'predict':
        run_prediction(args.model_type, args.input_file, args.output_file)
    else:
        raise ValueError(f"不支持的运行模式: {args.mode}")

if __name__ == "__main__":
    main() 