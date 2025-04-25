# 信用风险评估系统

## 项目概述
GiveMeSomeCredit 是一个基于机器学习的信用风险评估系统，旨在帮助金融机构评估客户的信用风险。系统使用多种机器学习算法，包括 XGBoost、LightGBM、CatBoost、随机森林和逻辑回归，以提供准确的信用风险评估。

## 数据来源
https://www.kaggle.com/c/GiveMeSomeCredit/data

## 功能特点
- 数据探索和可视化
- 特征工程和预处理
- 多种机器学习模型训练
- 模型评估和比较
- 预测服务
- 模型监控和评估

### 支持的模型
- XGBoost
- LightGBM
- Random Forest
- Logistic Regression
- CatBoost

### 特征工程
#### 基础特征
- 年龄分段：将年龄分为6个区间（18-25, 26-35, 36-45, 46-55, 56-65, 65+）
- 收入分段：将月收入分为6个区间（<2k, 2k-4k, 4k-6k, 6k-8k, 8k-10k, >10k）
- 债务比率分段：将债务比率分为6个区间（<0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, >1.0）
- 总逾期次数：计算30-59天、60-89天和90天以上逾期的总和

#### 衍生特征
- 债务收入比：债务比率与月收入的乘积，反映实际债务负担
- 信用额度使用率与债务比率交互：反映信用使用与债务的综合情况
- 逾期严重程度：90天以上逾期占总逾期的比例
- 信用历史长度：基于年龄估算的信用历史
- 贷款数量与收入比：反映贷款负担
- 房产贷款占比：反映贷款结构
- 家庭负担：抚养人数与收入的关系
- 信用使用效率：信用额度的使用情况

## 系统架构
```
GiveMeSomeCredit/
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后的数据
├── models/                  # 模型目录
├── results/                 # 结果目录
│   ├── plots/               # 可视化图表
│   └── metrics/             # 评估指标
├── src/                     # 源代码
│   ├── config.py            # 配置文件
│   ├── data_exploration.py  # 数据探索
│   ├── model_training.py    # 模型训练
│   ├── predict.py           # 预测功能
│   ├── predict_sample.py    # 预测示例
│   ├── main.py              # 主程序
│   └── monitor.py           # 模型监控
├── tests/                   # 测试代码
├── requirements.txt         # 依赖包
└── README.md                # 项目说明
```

## 环境要求
- Python 3.8+
- 依赖包（见 requirements.txt）

## 安装说明
1. 克隆仓库：
```bash
git clone https://github.com/yourusername/GiveMeSomeCredit.git
cd GiveMeSomeCredit
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据探索
运行数据探索流程，包括数据加载、探索和预处理：
```bash
python src/main.py --mode explore
```

### 2. 模型训练
训练指定类型的模型：
```bash
python src/main.py --mode train --model_type lightgbm
```
训练所有模型：
```bash
python src/main.py --mode train --train_all
```

支持的模型类型：
- xgboost
- lightgbm
- random_forest
- logistic_regression
- catboost

### 3. 预测
#### 单样本预测
使用特定模型和示例数据进行预测：
```bash
python src/main.py --mode predict --model_type lightgbm
```
使用所有模型和示例数据进行预测：
```bash
python src/main.py --mode predict --use_all_models
```

#### 批量预测
使用特定模型和输入文件进行批量预测：
```bash
python src/main.py --mode predict --model_type lightgbm --input_file data/test.csv --output_file results/predictions.csv
```
使用所有模型和输入文件进行批量预测：
```bash
python src/main.py --mode predict --use_all_models --input_file data/test.csv --output_file results/predictions.csv
```

### 4. 预测示例
运行预测示例脚本，展示如何使用模型进行预测：
```bash
python src/predict_sample.py
```

## 输入数据格式
输入数据应包含以下特征：
```python
{
    'RevolvingUtilizationOfUnsecuredLines': float,  # 信用卡使用率
    'age': int,                                     # 年龄
    'NumberOfTime30-59DaysPastDueNotWorse': int,    # 30-59天逾期次数
    'DebtRatio': float,                             # 债务比率
    'MonthlyIncome': float,                         # 月收入
    'NumberOfOpenCreditLinesAndLoans': int,         # 贷款数量
    'NumberOfTimes90DaysLate': int,                 # 90天以上逾期次数
    'NumberRealEstateLoansOrLines': int,            # 房产贷款数量
    'NumberOfTime60-89DaysPastDueNotWorse': int,    # 60-89天逾期次数
    'NumberOfDependents': int                       # 抚养人数
}
```

## 输出结果
预测结果包含以下信息：
- probability: 违约概率
- prediction: 预测类别（0/1）
- risk_level: 风险等级（低风险/中低风险/中高风险/高风险）

## 模型评估
系统使用以下指标评估模型性能：
- AUC-ROC
- PR AUC
- 精确率
- 召回率
- F1分数
- 混淆矩阵

## 注意事项
1. 确保数据目录结构正确
2. 运行预测前需要先训练模型
3. 输入数据需要包含所有必需的特征
4. 建议使用LightGBM模型，因其在测试中表现最佳
5. 定期监控模型性能，及时更新模型

## 贡献
欢迎提交问题和改进建议！

