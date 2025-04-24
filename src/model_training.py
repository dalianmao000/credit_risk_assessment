import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    average_precision_score, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_processed_data():
    """加载处理后的数据"""
    df = pd.read_csv(PROCESSED_DATA_DIR / 'processed_train.csv')
    return df

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

def prepare_data(df):
    """准备训练数据"""
    # 特征工程
    df = feature_engineering(df)
    
    # 选择特征
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    
    # 处理缺失值
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # 处理类别不平衡
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_val, y_train_resampled, y_val

def train_model(X_train, X_val, y_train, y_val, model_type='xgboost'):
    """训练指定类型的模型"""
    if model_type == 'xgboost':
        return train_xgboost(X_train, X_val, y_train, y_val)
    elif model_type == 'lightgbm':
        return train_lightgbm(X_train, X_val, y_train, y_val)
    elif model_type == 'random_forest':
        return train_random_forest(X_train, X_val, y_train, y_val)
    elif model_type == 'logistic_regression':
        return train_logistic_regression(X_train, X_val, y_train, y_val)
    elif model_type == 'catboost':
        return train_catboost(X_train, X_val, y_train, y_val)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def train_xgboost(X_train, X_val, y_train, y_val):
    """训练XGBoost模型"""
    print("\n训练XGBoost模型...")
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # 训练模型
    model = xgb.train(
        MODEL_PARAMS['xgboost'],
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # 保存模型
    model.save_model(MODEL_DIR / 'xgboost_model.json')
    
    return model

def train_lightgbm(X_train, X_val, y_train, y_val):
    """训练LightGBM模型"""
    print("\n训练LightGBM模型...")
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 训练模型
    model = lgb.train(
        MODEL_PARAMS['lightgbm'],
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 保存模型
    model.save_model(MODEL_DIR / 'lightgbm_model.txt')
    
    return model

def train_random_forest(X_train, X_val, y_train, y_val):
    """训练随机森林模型"""
    print("\n训练随机森林模型...")
    
    model = RandomForestClassifier(**MODEL_PARAMS['random_forest'])
    model.fit(X_train, y_train)
    
    # 保存模型
    joblib.dump(model, MODEL_DIR / 'random_forest_model.joblib')
    
    return model

def train_logistic_regression(X_train, X_val, y_train, y_val):
    """训练逻辑回归模型"""
    print("\n训练逻辑回归模型...")
    
    model = LogisticRegression(**MODEL_PARAMS['logistic_regression'])
    model.fit(X_train, y_train)
    
    # 保存模型
    joblib.dump(model, MODEL_DIR / 'logistic_regression_model.joblib')
    
    return model

def train_catboost(X_train, X_val, y_train, y_val):
    """训练CatBoost模型"""
    print("\n训练CatBoost模型...")
    
    model = cb.CatBoostClassifier(**MODEL_PARAMS['catboost'])
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100
    )
    
    # 保存模型
    model.save_model(MODEL_DIR / 'catboost_model.cbm')
    
    return model

def evaluate_model(model, X_val, y_val, model_type='xgboost'):
    """评估模型性能"""
    # 预测概率
    if model_type == 'xgboost':
        dval = xgb.DMatrix(X_val)
        y_pred_proba = model.predict(dval)
    elif model_type == 'lightgbm':
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    else:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 计算各种评估指标
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'auc': roc_auc,
        'pr_auc': average_precision_score(y_val, y_pred_proba),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'confusion_matrix': confusion_matrix(y_val, y_pred),
        'roc_curve': (fpr, tpr, roc_auc)  # 保存ROC曲线数据
    }
    
    # 打印评估结果
    print("\n模型评估结果:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.savefig(RESULT_DIR / f'confusion_matrix_{model_type}.png')
    plt.close()
    
    return metrics

def cross_validate_model(X, y, model_type='xgboost'):
    """交叉验证评估模型"""
    cv = StratifiedKFold(**CV_PARAMS)
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(**MODEL_PARAMS['xgboost'])
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**MODEL_PARAMS['lightgbm'])
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**MODEL_PARAMS['random_forest'])
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**MODEL_PARAMS['logistic_regression'])
    elif model_type == 'catboost':
        model = cb.CatBoostClassifier(**MODEL_PARAMS['catboost'])
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 计算交叉验证得分
    cv_scores = cross_val_score(
        model, X, y, cv=cv, scoring='roc_auc'
    )
    
    print(f"\n{model_type} 交叉验证结果:")
    print(f"平均AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def feature_importance(model, model_type='xgboost'):
    """分析特征重要性"""
    if model_type == 'xgboost':
        importance = model.get_score(importance_type='weight')
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    elif model_type == 'lightgbm':
        importance = list(zip(FEATURES, model.feature_importance()))
        importance = sorted(importance, key=lambda x: x[1], reverse=True)
    elif model_type == 'random_forest':
        importance = list(zip(FEATURES, model.feature_importances_))
        importance = sorted(importance, key=lambda x: x[1], reverse=True)
    elif model_type == 'logistic_regression':
        # 逻辑回归使用系数作为特征重要性，保留正负符号
        importance = list(zip(FEATURES, model.coef_[0]))
        importance = sorted(importance, key=lambda x: abs(x[1]), reverse=True)  # 按绝对值排序
    elif model_type == 'catboost':
        importance = list(zip(FEATURES, model.get_feature_importance()))
        importance = sorted(importance, key=lambda x: x[1], reverse=True)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    print("\n特征重要性:")
    for feature, score in importance:
        print(f"{feature}: {score}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    features, scores = zip(*importance)
    
    if model_type == 'logistic_regression':
        # 为逻辑回归创建带有正负符号的条形图
        colors = ['red' if x < 0 else 'blue' for x in scores]
        plt.barh(features, scores, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)  # 添加零线
        plt.title(f'{model_type} 特征重要性（带正负符号）')
    else:
        plt.barh(features, scores)
        plt.title(f'{model_type} 特征重要性')
    
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f'feature_importance_{model_type}.png')
    plt.close()
    
    return importance

if __name__ == "__main__":
    # 加载数据
    df = load_processed_data()
    
    # 准备数据
    X_train, X_val, y_train, y_val = prepare_data(df)
    
    # 定义要训练的模型
    models_to_train = ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'catboost']
    
    # 训练和评估每个模型
    results = {}
    roc_data = {}  # 存储所有模型的ROC曲线数据
    
    for model_type in models_to_train:
        print(f"\n=== 训练和评估 {model_type} 模型 ===")
        model = train_model(X_train, X_val, y_train, y_val, model_type)
        metrics = evaluate_model(model, X_val, y_val, model_type)
        cv_scores = cross_validate_model(X_train, y_train, model_type)
        importance = feature_importance(model, model_type)
        
        results[model_type] = {
            'metrics': metrics,
            'cv_scores': cv_scores,
            'importance': importance
        }
        
        # 保存ROC曲线数据
        roc_data[model_type] = metrics['roc_curve']
    
    print("\n所有模型训练完成！")
    
    # 绘制ROC曲线对比图
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    
    for model_type, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{model_type} (AUC = {auc_score:.4f})')
    
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('不同模型的ROC曲线对比')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(RESULT_DIR / 'roc_curves_comparison.png')
    plt.close()
    
    # 保存评估结果
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
        for model_type in models_to_train
    }).T
    
    results_df.to_csv(RESULT_DIR / 'model_comparison.csv')
    print("\n模型比较结果已保存到 results/model_comparison.csv")