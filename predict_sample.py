import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from predict import predict_single_sample, predict_batch
import pandas as pd

def main():
    print("信用风险预测演示")
    
    # 示例数据
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
    
    # 使用XGBoost模型预测
    print("\n使用XGBoost模型预测:")
    xgb_result = predict_single_sample(sample_data, model_type='xgboost')
    print("预测结果:", xgb_result)
    
    # 使用LightGBM模型预测
    print("\n使用LightGBM模型预测:")
    lgb_result = predict_single_sample(sample_data, model_type='lightgbm')
    print("预测结果:", lgb_result)
    
    # 批量预测示例
    print("\n批量预测示例:")
    # 创建一些不同的样本
    samples = [
        {**sample_data, 'age': 25, 'MonthlyIncome': 3000},  # 年轻低收入
        {**sample_data, 'age': 45, 'MonthlyIncome': 8000},  # 中年高收入
        {**sample_data, 'NumberOfTimes90DaysLate': 2, 'DebtRatio': 0.8}  # 有逾期记录高负债
    ]
    
    batch_data = pd.DataFrame(samples)
    batch_results = predict_batch(batch_data)
    print("\n批量预测结果:")
    print(batch_results)

if __name__ == "__main__":
    main() 