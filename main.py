import os
import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from data_exploration import load_data, explore_data, preprocess_data, save_processed_data
from model_training import (
    load_processed_data as load_processed,
    prepare_data,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
    feature_importance
)

def main():
    print("开始信用评分模型训练流程...")
    
    # 步骤1: 数据探索和预处理
    print("\n=== 步骤1: 数据探索和预处理 ===")
    df = load_data()
    explore_data(df)
    processed_df = preprocess_data(df)
    save_processed_data(processed_df)
    
    # 步骤2: 模型训练和评估
    print("\n=== 步骤2: 模型训练和评估 ===")
    df = load_processed()
    X_train, X_val, y_train, y_val = prepare_data(df)
    
    # 训练XGBoost模型
    print("\n--- 训练XGBoost模型 ---")
    xgb_model = train_xgboost(X_train, X_val, y_train, y_val)
    xgb_auc, xgb_pr_auc = evaluate_model(xgb_model, X_val, y_val, 'xgboost')
    xgb_importance = feature_importance(xgb_model, 'xgboost')
    
    # 训练LightGBM模型
    print("\n--- 训练LightGBM模型 ---")
    lgb_model = train_lightgbm(X_train, X_val, y_train, y_val)
    lgb_auc, lgb_pr_auc = evaluate_model(lgb_model, X_val, y_val, 'lightgbm')
    lgb_importance = feature_importance(lgb_model, 'lightgbm')
    
    print("\n模型训练流程完成！")

if __name__ == "__main__":
    main() 