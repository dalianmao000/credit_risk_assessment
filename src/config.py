import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 模型目录
MODEL_DIR = ROOT_DIR / "models"

# 结果目录
RESULT_DIR = ROOT_DIR / "results"

# 创建必要的目录
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据文件路径
TRAIN_DATA_PATH = RAW_DATA_DIR / "cs-training.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "cs-test.csv"

# 随机种子
RANDOM_SEED = 42

# 特征列表
FEATURES = [
    # 原始特征
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    # 衍生特征
    'total_overdue',
    'debt_to_income',
    'utilization_debt_ratio',
    'severe_overdue_ratio',
    'credit_history_length',
    'loans_to_income',
    'real_estate_loan_ratio',
    'family_burden',
    'credit_utilization_efficiency'
]

# 目标变量
TARGET = 'SeriousDlqin2yrs'

# 模型参数
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'scale_pos_weight': 1.0,  # 处理类别不平衡
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss']
    },
    'lightgbm': {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_SEED
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED
    },
    'logistic_regression': {
        'C': 1.0,
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED,
        'max_iter': 1000
    },
    'catboost': {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_state': RANDOM_SEED,
        'class_weights': [1, 1],
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 100
    }
}

# 交叉验证参数
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': RANDOM_SEED
}

# 评估指标阈值
THRESHOLDS = {
    'low_risk': 0.2,
    'medium_low_risk': 0.5,
    'medium_high_risk': 0.8
}

# 特征工程参数
FEATURE_ENGINEERING = {
    'age_bins': [18, 25, 35, 45, 55, 65, 100],
    'income_bins': [0, 2000, 4000, 6000, 8000, 10000, float('inf')],
    'debt_ratio_bins': [0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
}

# 风险阈值配置
RISK_THRESHOLDS = {
    'low': 0.3,     # 低风险阈值
    'medium': 0.6,  # 中风险阈值
    # 大于medium的为高风险
}