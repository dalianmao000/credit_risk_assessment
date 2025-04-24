# 信用评分数据集特征说明

## 目标变量
- **SeriousDlqin2yrs**（严重逾期预测）
  - 含义：借款人是否在未来2年内出现90天以上的逾期
  - 类型：是/否（Y/N）

## 特征变量

### 信用使用情况
- **RevolvingUtilizationOfUnsecuredLines**（信用额度使用率）
  - 含义：信用卡和个人信用额度总余额占比
  - 类型：百分比

### 个人基本信息
- **age**（年龄）
  - 含义：借款人年龄
  - 类型：整数
- **NumberOfDependents**（家庭受抚养人数）
  - 含义：家庭受抚养人数量（不包括本人）
  - 类型：整数

### 收入与债务
- **MonthlyIncome**（月收入）
  - 含义：借款人月收入
  - 类型：实数
- **DebtRatio**（债务收入比）
  - 含义：月债务支出（包括赡养费、生活费）占收入比例
  - 类型：百分比

### 信用历史
- **NumberOfTime30-59DaysPastDueNotWorse**（轻度逾期次数）
  - 含义：过去逾期30-59天的次数
  - 类型：整数
- **NumberOfTime60-89DaysPastDueNotWorse**（中度逾期次数）
  - 含义：过去逾期60-89天的次数
  - 类型：整数
- **NumberOfTimes90DaysLate**（严重逾期次数）
  - 含义：过去逾期90天或更长时间的次数
  - 类型：整数

### 信贷账户情况
- **NumberOfOpenCreditLinesAndLoans**（开放信贷数量）
  - 含义：开放的信用额度和贷款数量
  - 类型：整数
- **NumberRealEstateLoansOrLines**（不动产贷款数量）
  - 含义：不动产贷款和信用额度数量
  - 类型：整数 