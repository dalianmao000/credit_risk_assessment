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
from predict import predict_single_sample, predict_batch, predict_all_models
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
    parser.add_argument('--use_all_models', action='store_true',
                      help='是否使用所有模型进行预测')
    parser.add_argument('--train_all', action='store_true',
                      help='是否训练所有模型')
    return parser.parse_args()

def run_data_exploration():
    """运行数据探索流程"""
    print("\n=== 开始数据探索 ===")
    
    # 加载数据
    train_df = load_data(TRAIN_DATA_PATH)
    test_df = load_data(TEST_DATA_PATH)
    
    # 数据探索
    explore_data(train_df)
    
    # 数据预处理
    train_processed = preprocess_data(train_df)
    test_processed = preprocess_data(test_df)
    
    # 保存处理后的数据
    save_processed_data(train_processed, 'processed_train.csv')
    save_processed_data(test_processed, 'processed_test.csv')
    
    print("\n数据探索完成！")

def run_model_training(model_type=None, train_all=False):
    """运行模型训练流程
    Args:
        model_type: 模型类型，如果train_all为True则忽略此参数
        train_all: 是否训练所有模型
    """
    # 加载处理后的数据
    df = load_processed_data()
    
    # 准备数据
    X_train, X_val, y_train, y_val = prepare_data(df)
    
    if train_all:
        print("\n=== 开始训练所有模型 ===")
        models_to_train = ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost']
        results = {}
        best_auc = 0
        best_model = None
        
        # 训练和评估每个模型
        for model_type in models_to_train:
            print(f"\n--- 训练 {model_type} 模型 ---")
            try:
                # 训练模型
                model = train_model(X_train, X_val, y_train, y_val, model_type)
                
                # 评估模型
                metrics = evaluate_model(model, X_val, y_val, model_type)
                cv_scores = cross_validate_model(X_train, y_train, model_type)
                importance = feature_importance(model, model_type)
                
                results[model_type] = {
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'importance': importance
                }
                
                # 更新最佳模型
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_model = model_type
                
                print(f"\n{model_type} 模型评估结果:")
                print(f"验证集 AUC: {metrics['auc']:.4f}")
                print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            except Exception as e:
                print(f"\n{model_type} 模型训练失败: {str(e)}")
        
        # 保存模型比较结果
        results_df = pd.DataFrame({
            model_type: {
                'AUC': results[model_type]['metrics']['auc'],
                'PR AUC': results[model_type]['metrics']['pr_auc'],
                'Precision': results[model_type]['metrics']['precision'],
                'Recall': results[model_type]['metrics']['recall'],
                'F1 Score': results[model_type]['metrics']['f1'],
                'CV AUC Mean': results[model_type]['cv_scores'].mean(),
                'CV AUC Std': results[model_type]['cv_scores'].std()
            }
            for model_type in results.keys()
        }).T
        
        results_df.to_csv(RESULT_DIR / 'model_comparison.csv')
        print("\n所有模型训练完成！")
        print(f"最佳模型: {best_model} (AUC: {best_auc:.4f})")
        print("\n模型比较结果已保存到 results/model_comparison.csv")
        
        # 绘制ROC曲线对比图
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        
        for model_type in results:
            fpr, tpr, roc_auc = results[model_type]['metrics']['roc_curve']
            plt.plot(fpr, tpr, label=f'{model_type} (AUC = {roc_auc:.4f})')
        
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title('不同模型的ROC曲线对比')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(RESULT_DIR / 'roc_curves_comparison.png')
        plt.close()
    
    else:
        print(f"\n=== 开始训练 {model_type} 模型 ===")
        # 训练模型
        model = train_model(X_train, X_val, y_train, y_val, model_type)
        
        # 评估模型
        metrics = evaluate_model(model, X_val, y_val, model_type)
        cv_scores = cross_validate_model(X_train, y_train, model_type)
        importance = feature_importance(model, model_type)
        
        print(f"\n{model_type} 模型训练完成！")
        print(f"验证集 AUC: {metrics['auc']:.4f}")
        print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

def run_prediction(model_type, input_file=None, output_file=None, use_all_models=False):
    """运行预测流程
    Args:
        model_type: 模型类型
        input_file: 输入文件路径
        output_file: 输出文件路径
        use_all_models: 是否使用所有模型预测
    """
    if use_all_models:
        print("\n=== 开始使用所有模型进行预测 ===")
    else:
        print(f"\n=== 开始使用 {model_type} 模型进行预测 ===")
    
    if input_file:
        # 从文件加载数据
        data = pd.read_csv(input_file)
        if use_all_models:
            results = predict_all_models(data)
        else:
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
        
        if use_all_models:
            results = predict_all_models(sample_data)
            print("\n所有模型预测结果:")
            print(results)
        else:
            result = predict_single_sample(sample_data, model_type=model_type)
            print("\n单模型预测结果:")
            print(result)

def main():
    """主函数"""
    args = parse_args()
    
    if args.mode == 'explore':
        run_data_exploration()
    elif args.mode == 'train':
        run_model_training(args.model_type, args.train_all)
    elif args.mode == 'predict':
        run_prediction(args.model_type, args.input_file, args.output_file, args.use_all_models)
    else:
        raise ValueError(f"不支持的运行模式: {args.mode}")

if __name__ == "__main__":
    main() 