import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from config import *

def feature_engineering(df):
    """特征工程"""
    df = df.copy()
    
    # 年龄分段
    df['age_group'] = pd.cut(
        df['age'],
        bins=FEATURE_ENGINEERING['age_bins'],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
    
    # 收入分段
    df['income_group'] = pd.cut(
        df['MonthlyIncome'],
        bins=FEATURE_ENGINEERING['income_bins'],
        labels=['<2k', '2k-4k', '4k-6k', '6k-8k', '8k-10k', '>10k']
    )
    
    # 债务比率分段
    df['debt_ratio_group'] = pd.cut(
        df['DebtRatio'],
        bins=FEATURE_ENGINEERING['debt_ratio_bins'],
        labels=['<0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '>1.0']
    )
    
    # 创建逾期次数总和特征
    df['total_overdue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    
    # 新增特征：债务收入比
    df['debt_to_income'] = df['DebtRatio'] * df['MonthlyIncome']
    
    # 新增特征：信用额度使用率与债务比率的交互
    df['utilization_debt_ratio'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    
    # 新增特征：逾期严重程度
    df['severe_overdue_ratio'] = df['NumberOfTimes90DaysLate'] / (df['total_overdue'] + 1)
    
    # 新增特征：信用历史长度（假设年龄越大，信用历史越长）
    df['credit_history_length'] = df['age'] - 18  # 假设18岁开始有信用记录
    
    # 新增特征：贷款数量与收入比
    df['loans_to_income'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['MonthlyIncome'] + 1)
    
    # 新增特征：房产贷款占比
    df['real_estate_loan_ratio'] = df['NumberRealEstateLoansOrLines'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)
    
    # 新增特征：家庭负担（抚养人数与收入比）
    df['family_burden'] = df['NumberOfDependents'] / (df['MonthlyIncome'] + 1)
    
    # 新增特征：信用使用效率
    df['credit_utilization_efficiency'] = df['RevolvingUtilizationOfUnsecuredLines'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)
    
    return df

class CreditRiskPredictor:
    def __init__(self, model_type='xgboost'):
        """
        初始化预测器
        Args:
            model_type: 模型类型，可选 'xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost'
        """
        self.model_type = model_type
        self.model = self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        if self.model_type == 'xgboost':
            model_path = MODEL_DIR / 'xgboost_model.json'
            model = xgb.Booster()
            model.load_model(str(model_path))
        elif self.model_type == 'lightgbm':
            model_path = MODEL_DIR / 'lightgbm_model.txt'
            model = lgb.Booster(model_file=str(model_path))
        elif self.model_type == 'random_forest':
            model_path = MODEL_DIR / 'random_forest_model.joblib'
            model = joblib.load(model_path)
        elif self.model_type == 'logistic_regression':
            model_path = MODEL_DIR / 'logistic_regression_model.joblib'
            model = joblib.load(model_path)
        elif self.model_type == 'catboost':
            model_path = MODEL_DIR / 'catboost_model.cbm'
            model = cb.CatBoostClassifier()
            model.load_model(str(model_path))
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        return model
    
    def preprocess_input(self, data):
        """预处理输入数据"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是字典或DataFrame格式")
        
        # 应用特征工程
        processed_data = feature_engineering(data)
        
        # 选择特征
        processed_data = processed_data[FEATURES]
        
        # 处理缺失值
        for col in processed_data.columns:
            if processed_data[col].dtype in ['int64', 'float64']:
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
            else:
                processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
        
        return processed_data
    
    def predict(self, data, threshold=0.5):
        """
        预测违约概率
        Args:
            data: 输入数据
            threshold: 分类阈值，默认0.5
        Returns:
            dict: 包含预测结果的字典
        """
        # 预处理数据
        processed_data = self.preprocess_input(data)
        
        # 预测概率
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(processed_data)
            proba = self.model.predict(dmatrix)
        elif self.model_type == 'lightgbm':
            proba = self.model.predict(processed_data)
        else:
            proba = self.model.predict_proba(processed_data)[:, 1]
            
        # 转换为二分类结果
        prediction = (proba >= threshold).astype(int)
        
        # 返回结果
        result = {
            'probability': float(proba[0]),
            'prediction': int(prediction[0]),
            'risk_level': self._get_risk_level(proba[0])
        }
        
        return result
    
    def _get_risk_level(self, probability):
        """根据概率确定风险等级"""
        if probability < THRESHOLDS['low_risk']:
            return '低风险'
        elif probability < THRESHOLDS['medium_low_risk']:
            return '中低风险'
        elif probability < THRESHOLDS['medium_high_risk']:
            return '中高风险'
        else:
            return '高风险'
    
    def batch_predict(self, data, threshold=0.5):
        """
        批量预测
        Args:
            data: 输入数据DataFrame
            threshold: 分类阈值
        Returns:
            DataFrame: 包含预测结果的DataFrame
        """
        # 预处理数据
        processed_data = self.preprocess_input(data)
        
        # 预测概率
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(processed_data)
            proba = self.model.predict(dmatrix)
        elif self.model_type == 'lightgbm':
            proba = self.model.predict(processed_data)
        else:
            proba = self.model.predict_proba(processed_data)[:, 1]
            
        # 转换为二分类结果
        prediction = (proba >= threshold).astype(int)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'probability': proba,
            'prediction': prediction,
            'risk_level': [self._get_risk_level(p) for p in proba]
        })
        
        return results

def predict_single_sample(sample_data, model_type='xgboost'):
    """
    预测单个样本的便捷函数
    Args:
        sample_data: 单个样本的数据（字典格式）
        model_type: 模型类型
    Returns:
        dict: 预测结果
    """
    predictor = CreditRiskPredictor(model_type=model_type)
    return predictor.predict(sample_data)

def predict_batch(data, model_type='xgboost'):
    """
    批量预测的便捷函数
    Args:
        data: 批量数据（DataFrame格式）
        model_type: 模型类型
    Returns:
        DataFrame: 预测结果
    """
    predictor = CreditRiskPredictor(model_type=model_type)
    return predictor.batch_predict(data)

if __name__ == "__main__":
    # 示例用法
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
    
    # 测试所有模型
    for model_type in ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost']:
        try:
            print(f"\n使用 {model_type} 模型预测:")
            result = predict_single_sample(sample_data, model_type=model_type)
            print("预测结果:", result)
        except Exception as e:
            print(f"使用 {model_type} 模型预测时出错: {str(e)}")
    
    # 批量预测示例
    print("\n批量预测示例:")
    # 创建一些不同的样本
    samples = [
        {**sample_data, 'age': 25, 'MonthlyIncome': 3000},  # 年轻低收入
        {**sample_data, 'age': 45, 'MonthlyIncome': 8000},  # 中年高收入
        {**sample_data, 'NumberOfTimes90DaysLate': 2, 'DebtRatio': 0.8}  # 有逾期记录高负债
    ]
    
    batch_data = pd.DataFrame(samples)
    print("\n批量预测结果:")
    for model_type in ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost']:
        try:
            print(f"\n使用 {model_type} 模型批量预测:")
            batch_results = predict_batch(batch_data, model_type=model_type)
            print(batch_results)
        except Exception as e:
            print(f"使用 {model_type} 模型批量预测时出错: {str(e)}")