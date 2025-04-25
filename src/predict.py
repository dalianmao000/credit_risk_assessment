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
from data_exploration import preprocess_data
from model_training import feature_engineering

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
    def __init__(self, model_type='lightgbm'):
        """初始化预测器
        Args:
            model_type: 模型类型，可选 'xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost'
        """
        self.model_type = model_type
        self.model = self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.model_type == 'xgboost':
            model = xgb.Booster()
            model.load_model(str(MODEL_DIR / 'xgboost_model.json'))
        elif self.model_type == 'lightgbm':
            model = lgb.Booster(model_file=str(MODEL_DIR / 'lightgbm_model.txt'))
        elif self.model_type == 'random_forest':
            model = joblib.load(MODEL_DIR / 'random_forest_model.joblib')
        elif self.model_type == 'logistic_regression':
            model = joblib.load(MODEL_DIR / 'logistic_regression_model.joblib')
        elif self.model_type == 'catboost':
            model = cb.CatBoostClassifier()
            model.load_model(MODEL_DIR / 'catboost_model.cbm')
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        return model
    
    def preprocess_input(self, data):
        """预处理输入数据"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # 数据预处理
        data = preprocess_data(data)
        # 特征工程
        data = feature_engineering(data)
        # 选择特征
        data = data[FEATURES]
        
        return data
    
    def predict_proba(self, data):
        """预测违约概率"""
        data = self.preprocess_input(data)
        
        if self.model_type == 'xgboost':
            ddata = xgb.DMatrix(data)
            proba = self.model.predict(ddata)
        elif self.model_type == 'lightgbm':
            proba = self.model.predict(data)
        else:
            proba = self.model.predict_proba(data)[:, 1]
        
        return proba
    
    def predict_risk_level(self, data):
        """预测风险等级"""
        proba = self.predict_proba(data)
        
        def get_risk_level(p):
            if p < RISK_THRESHOLDS['low']:
                return '低风险'
            elif p < RISK_THRESHOLDS['medium']:
                return '中风险'
            else:
                return '高风险'
        
        if isinstance(proba, np.ndarray):
            return [get_risk_level(p) for p in proba]
        else:
            return get_risk_level(proba)

class EnsembleCreditRiskPredictor:
    def __init__(self):
        """初始化集成预测器，加载所有可用模型"""
        self.models = {}
        self.model_types = ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost']
        
        for model_type in self.model_types:
            try:
                self.models[model_type] = CreditRiskPredictor(model_type)
                print(f"成功加载 {model_type} 模型")
            except Exception as e:
                print(f"加载 {model_type} 模型失败: {str(e)}")
    
    def predict_all(self, data):
        """使用所有模型进行预测
        Args:
            data: 字典或DataFrame形式的输入数据
        Returns:
            DataFrame: 包含所有模型的预测结果
        """
        results = {}
        
        # 对每个模型进行预测
        for model_type, predictor in self.models.items():
            try:
                proba = predictor.predict_proba(data)
                risk_level = predictor.predict_risk_level(data)
                
                if isinstance(data, dict):
                    results[f'{model_type}_probability'] = [proba]
                    results[f'{model_type}_risk_level'] = [risk_level]
                else:
                    results[f'{model_type}_probability'] = proba
                    results[f'{model_type}_risk_level'] = risk_level
            except Exception as e:
                print(f"{model_type} 模型预测失败: {str(e)}")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 如果是单样本预测，添加输入特征
        if isinstance(data, dict):
            for key, value in data.items():
                results_df[key] = value
        
        # 计算综合预测结果
        proba_columns = [col for col in results_df.columns if col.endswith('_probability')]
        results_df['ensemble_probability'] = results_df[proba_columns].mean(axis=1)
        
        def get_risk_level(p):
            if p < RISK_THRESHOLDS['low']:
                return '低风险'
            elif p < RISK_THRESHOLDS['medium']:
                return '中风险'
            else:
                return '高风险'
        
        results_df['ensemble_risk_level'] = results_df['ensemble_probability'].apply(get_risk_level)
        
        return results_df

def predict_single_sample(data, model_type='lightgbm'):
    """预测单个样本"""
    predictor = CreditRiskPredictor(model_type)
    proba = predictor.predict_proba(data)
    risk_level = predictor.predict_risk_level(data)
    
    result = {
        'probability': float(proba),
        'risk_level': risk_level
    }
    return result

def predict_batch(data, model_type='lightgbm'):
    """批量预测"""
    predictor = CreditRiskPredictor(model_type)
    probas = predictor.predict_proba(data)
    risk_levels = predictor.predict_risk_level(data)
    
    results = pd.DataFrame({
        'probability': probas,
        'risk_level': risk_levels
    })
    return results

def predict_all_models(data):
    """使用所有模型进行预测
    Args:
        data: 字典（单样本）或DataFrame（批量样本）形式的输入数据
    Returns:
        DataFrame: 包含所有模型预测结果的数据框
    """
    predictor = EnsembleCreditRiskPredictor()
    return predictor.predict_all(data)

if __name__ == "__main__":
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
    
    # 测试单模型预测
    print("\n=== 单模型预测结果 ===")
    result = predict_single_sample(sample_data, 'lightgbm')
    print(f"LightGBM预测结果: {result}")
    
    # 测试所有模型预测
    print("\n=== 所有模型预测结果 ===")
    results = predict_all_models(sample_data)
    print(results)