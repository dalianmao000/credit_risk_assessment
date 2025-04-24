import pandas as pd
import numpy as np
from predict import predict_single_sample, predict_batch
from config import *

def create_sample_data():
    """创建示例数据"""
    # 创建不同风险等级的样本
    samples = [
        # 低风险样本
        {
            'RevolvingUtilizationOfUnsecuredLines': 0.2,
            'age': 45,
            'NumberOfTime30-59DaysPastDueNotWorse': 0,
            'DebtRatio': 0.1,
            'MonthlyIncome': 8000,
            'NumberOfOpenCreditLinesAndLoans': 3,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 1
        },
        # 中低风险样本
        {
            'RevolvingUtilizationOfUnsecuredLines': 0.5,
            'age': 35,
            'NumberOfTime30-59DaysPastDueNotWorse': 1,
            'DebtRatio': 0.3,
            'MonthlyIncome': 5000,
            'NumberOfOpenCreditLinesAndLoans': 5,
            'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 2
        },
        # 中高风险样本
        {
            'RevolvingUtilizationOfUnsecuredLines': 0.8,
            'age': 30,
            'NumberOfTime30-59DaysPastDueNotWorse': 2,
            'DebtRatio': 0.6,
            'MonthlyIncome': 3000,
            'NumberOfOpenCreditLinesAndLoans': 7,
            'NumberOfTimes90DaysLate': 1,
            'NumberRealEstateLoansOrLines': 2,
            'NumberOfTime60-89DaysPastDueNotWorse': 1,
            'NumberOfDependents': 3
        },
        # 高风险样本
        {
            'RevolvingUtilizationOfUnsecuredLines': 0.95,
            'age': 25,
            'NumberOfTime30-59DaysPastDueNotWorse': 3,
            'DebtRatio': 0.9,
            'MonthlyIncome': 2000,
            'NumberOfOpenCreditLinesAndLoans': 10,
            'NumberOfTimes90DaysLate': 2,
            'NumberRealEstateLoansOrLines': 3,
            'NumberOfTime60-89DaysPastDueNotWorse': 2,
            'NumberOfDependents': 4
        }
    ]
    return pd.DataFrame(samples)

def run_single_prediction_example():
    """运行单个样本预测示例"""
    print("\n=== 单个样本预测示例 ===")
    
    # 创建示例数据
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
    
    # 使用不同模型进行预测
    for model_type in ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost']:
        try:
            print(f"\n使用 {model_type} 模型预测:")
            result = predict_single_sample(sample_data, model_type=model_type)
            print("预测结果:", result)
        except Exception as e:
            print(f"使用 {model_type} 模型预测时出错: {str(e)}")

def run_batch_prediction_example():
    """运行批量预测示例"""
    print("\n=== 批量预测示例 ===")
    
    # 创建示例数据
    samples_df = create_sample_data()
    
    # 使用不同模型进行批量预测
    for model_type in ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost']:
        try:
            print(f"\n使用 {model_type} 模型批量预测:")
            results = predict_batch(samples_df, model_type=model_type)
            print("\n预测结果:")
            print(results)
            
            # 保存预测结果
            output_file = f"results/predictions_{model_type}.csv"
            results.to_csv(output_file, index=False)
            print(f"预测结果已保存到: {output_file}")
        except Exception as e:
            print(f"使用 {model_type} 模型批量预测时出错: {str(e)}")

def main():
    """主函数"""
    print("=== 信用风险预测示例 ===")
    
    # 运行单个样本预测示例
    run_single_prediction_example()
    
    # 运行批量预测示例
    run_batch_prediction_example()
    
    print("\n=== 示例运行完成 ===")

if __name__ == "__main__":
    main() 