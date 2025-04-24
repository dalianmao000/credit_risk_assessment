import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

def load_data():
    """加载训练数据"""
    df = pd.read_csv(TRAIN_DATA_PATH)
    return df

def explore_data(df):
    """数据探索"""
    print("\n=== 数据基本信息 ===")
    print(f"数据形状: {df.shape}")
    print("\n=== 数据概览 ===")
    print(df.info())
    print("\n=== 描述性统计 ===")
    print(df.describe())
    
    # 检查缺失值
    print("\n=== 缺失值统计 ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # 检查目标变量分布
    print("\n=== 目标变量分布 ===")
    target_dist = df[TARGET].value_counts(normalize=True)
    print(target_dist)
    
    # 绘制目标变量分布图
    plt.figure(figsize=(8, 6))
    sns.countplot(x=TARGET, data=df)
    plt.title('目标变量分布')
    plt.savefig(RESULT_DIR / 'target_distribution.png')
    plt.close()
    
    # 绘制特征相关性热力图
    plt.figure(figsize=(12, 10))
    corr_matrix = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'correlation_heatmap.png')
    plt.close()

def preprocess_data(df):
    """数据预处理"""
    # 处理缺失值
    # 对于数值型特征，使用中位数填充
    numeric_features = df[FEATURES].select_dtypes(include=['int64', 'float64']).columns
    for feature in numeric_features:
        df[feature] = df[feature].fillna(df[feature].median())
    
    # 处理异常值
    # 对于年龄，移除小于18岁的记录
    df = df[df['age'] >= 18]
    
    # 对于逾期次数，将大于90的值截断为90
    overdue_features = ['NumberOfTime30-59DaysPastDueNotWorse', 
                       'NumberOfTime60-89DaysPastDueNotWorse',
                       'NumberOfTimes90DaysLate']
    for feature in overdue_features:
        df[feature] = df[feature].clip(upper=90)
    
    return df

def save_processed_data(df, filename='processed_train.csv'):
    """保存处理后的数据"""
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)

if __name__ == "__main__":
    # 加载数据
    df = load_data()
    
    # 数据探索
    explore_data(df)
    
    # 数据预处理
    processed_df = preprocess_data(df)
    
    # 保存处理后的数据
    save_processed_data(processed_df)
    
    print("\n数据处理完成！") 