"""
生成智析项目全部学习Notebook
运行: cd zhixi && python notebooks/generate_notebooks.py
"""
import json
import os

def md(text):
    """创建Markdown cell"""
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(text):
    """创建Code cell"""
    return {"cell_type": "code", "metadata": {}, "source": [text], "outputs": [], "execution_count": None}

def make_notebook(cells, filename):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": cells
    }
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Generated: {filename} ({len(cells)} cells)")


# ========================================================================
# Notebook 01: NumPy + Pandas + 可视化 + ML入门 (W1-W4)
# ========================================================================
def gen_notebook_01():
    cells = [
        md("""# Week 1-4: Python数据科学基础 + 机器学习入门

> **学习目标**: 从零掌握 NumPy、Pandas、Matplotlib、Scikit-learn，完成一个端到端的数据分析+预测项目
>
> **预计时间**: 4周 (每天1-2小时)
>
> **本周交付**: Titanic生存预测完整Pipeline

---

## 学习路线

| 周 | 主题 | 核心技能 |
|----|------|----------|
| W1 | NumPy + Pandas | 数组操作、DataFrame、数据清洗 |
| W2 | 数据可视化 | Matplotlib、Seaborn、EDA |
| W3 | 探索性分析(EDA) | 统计描述、特征工程、缺失值处理 |
| W4 | 机器学习入门 | Scikit-learn、分类模型、交叉验证 |

> **学习方法**: 每个cell都**动手运行**，然后修改参数观察结果变化。不要只看不练！"""),

        # ===== Part 1: NumPy =====
        md("""---
## Part 1: NumPy — 数值计算基础

### 什么是NumPy？
NumPy是Python科学计算的基石。它的核心是**ndarray**（N维数组），比Python原生列表快几十倍。

**为什么需要NumPy？**
- 向量化运算：不用写for循环，一行代码完成整个数组的计算
- 内存高效：连续内存存储，C语言底层实现
- 广播机制：不同形状的数组可以自动对齐运算"""),

        code("""import numpy as np
print(f"NumPy 版本: {np.__version__}")

# ========================================
# 1.1 创建数组 — 5种常用方法
# ========================================

# 方法1: 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"从列表创建: {arr1}，形状: {arr1.shape}，类型: {arr1.dtype}")

# 方法2: 全零/全一数组 (常用于初始化权重)
zeros = np.zeros((3, 4))     # 3行4列全零矩阵
ones = np.ones((2, 3))       # 2行3列全一矩阵
print(f"\\n全零矩阵(3x4):\\n{zeros}")
print(f"全一矩阵(2x3):\\n{ones}")

# 方法3: 等差数列 (类似range但返回数组)
arr_range = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
arr_linspace = np.linspace(0, 1, 5) # 0到1之间均匀5个点
print(f"\\narange: {arr_range}")
print(f"linspace: {arr_linspace}")

# 方法4: 随机数组 (机器学习中最常用!)
np.random.seed(42)  # 固定随机种子，确保结果可复现
rand_arr = np.random.rand(3, 3)          # 均匀分布 [0,1)
randn_arr = np.random.randn(3, 3)        # 正态分布 (均值0,方差1)
randint_arr = np.random.randint(1, 100, (3, 3))  # 随机整数
print(f"\\n均匀分布随机矩阵:\\n{rand_arr}")
print(f"正态分布随机矩阵:\\n{randn_arr}")"""),

        code("""# ========================================
# 1.2 数组运算 — 向量化操作(比for循环快100倍!)
# ========================================

a = np.array([10, 20, 30, 40, 50])
b = np.array([1, 2, 3, 4, 5])

# 基础运算 (逐元素)
print(f"a + b = {a + b}")       # 加法
print(f"a * b = {a * b}")       # 逐元素乘法
print(f"a ** 2 = {a ** 2}")     # 平方
print(f"np.sqrt(a) = {np.sqrt(a)}")  # 开方

# 统计操作 (数据分析中最常用!)
print(f"\\n--- 统计操作 ---")
print(f"均值: {a.mean()}")        # 平均值
print(f"中位数: {np.median(a)}")   # 中位数
print(f"标准差: {a.std():.2f}")    # 标准差 (衡量数据离散程度)
print(f"最大值: {a.max()}，索引: {a.argmax()}")
print(f"最小值: {a.min()}，索引: {a.argmin()}")
print(f"总和: {a.sum()}")

# 排序
sorted_arr = np.sort(np.array([3, 1, 4, 1, 5, 9, 2, 6]))
print(f"\\n排序: {sorted_arr}")"""),

        code("""# ========================================
# 1.3 矩阵操作 — 二维数组
# ========================================

matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(f"矩阵形状: {matrix.shape}")  # (3, 4) = 3行4列

# 索引和切片 (和Python列表类似但更强大)
print(f"\\n第2行: {matrix[1]}")           # [5, 6, 7, 8]
print(f"第3列: {matrix[:, 2]}")           # [3, 7, 11]
print(f"子矩阵(前2行,后2列):\\n{matrix[:2, 2:]}")  # [[3,4],[7,8]]

# 沿轴操作 (axis=0是列方向, axis=1是行方向)
print(f"\\n每列求和(axis=0): {matrix.sum(axis=0)}")  # [15, 18, 21, 24]
print(f"每行求和(axis=1): {matrix.sum(axis=1)}")  # [10, 26, 42]
print(f"每列均值(axis=0): {matrix.mean(axis=0)}")

# reshape — 改变数组形状 (不改变数据)
reshaped = matrix.reshape(4, 3)  # 3x4 → 4x3
print(f"\\nreshape(4,3):\\n{reshaped}")

flat = matrix.flatten()  # 展平为一维
print(f"flatten: {flat}")"""),

        code("""# ========================================
# 1.4 布尔索引和花式索引 (数据筛选利器!)
# ========================================

scores = np.array([85, 92, 78, 95, 88, 72, 96, 81])
names = np.array(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'])

# 布尔索引: 筛选出分数>85的学生
mask = scores > 85
print(f"分数>85: {scores[mask]}")
print(f"对应学生: {names[mask]}")

# 多条件组合 (& = and, | = or)
high_and_not_alice = (scores > 85) & (names != 'Alice')
print(f"\\n>85且不是Alice: {names[high_and_not_alice]}")

# np.where: 条件赋值 (类似Excel的IF函数)
grades = np.where(scores >= 90, 'A', 
         np.where(scores >= 80, 'B', 'C'))
print(f"\\n成绩等级: {grades}")
for n, s, g in zip(names, scores, grades):
    print(f"  {n}: {s}分 → {g}")"""),

        # ===== Part 2: Pandas =====
        md("""---
## Part 2: Pandas — 数据分析核心

### 什么是Pandas？
Pandas是Python数据分析的核心库，由Wes McKinney开发。它的核心数据结构是**DataFrame**（二维表格），可以理解为一个可以在Python中操作的Excel表格。

**核心概念**:
- **Series**: 一维数组，类似Excel的一列
- **DataFrame**: 二维表格，类似一个Excel工作表
- **Index**: 行索引，用于快速定位数据"""),

        code("""import pandas as pd
print(f"Pandas 版本: {pd.__version__}")

# ========================================
# 2.1 创建DataFrame — 3种方式
# ========================================

# 方式1: 从字典创建 (最直观)
df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 35, 28, 32],
    '城市': ['北京', '上海', '广州', '深圳', '杭州'],
    '薪资': [15000, 25000, 18000, 22000, 20000],
    '满意度': [4, 5, 3, 4, 5],
})
print("=== 从字典创建 ===")
print(df)
print(f"\\n形状: {df.shape}  (5行 x 5列)")
print(f"列名: {list(df.columns)}")"""),

        code("""# 方式2: 从CSV文件读取 (实际项目中最常用!)
# df_csv = pd.read_csv('data.csv')

# 这里我们用网络上的经典数据集: Titanic
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
try:
    titanic = pd.read_csv(url)
    print(f"Titanic数据集加载成功! 形状: {titanic.shape}")
except:
    print("网络加载失败，使用模拟数据")
    titanic = pd.DataFrame({
        'PassengerId': range(1, 11),
        'Survived': [0,1,1,1,0,0,0,0,1,1],
        'Pclass': [3,1,3,1,3,3,1,3,3,2],
        'Name': ['Braund','Cumyns','Heikkinen','Futrelle','Allen',
                 'Moran','McCarthy','Palsson','Johnson','Nasser'],
        'Sex': ['male','female','female','female','male',
                'male','male','female','female','female'],
        'Age': [22,38,26,35,35,27,54,2,27,14],
        'SibSp': [1,1,0,1,0,0,0,3,0,1],
        'Fare': [7.25,71.28,7.92,53.1,8.05,8.46,51.86,21.08,11.13,30.07],
    })

# 快速了解数据
print("\\n=== 前5行 ===")
print(titanic.head())
print("\\n=== 基本信息 ===")
titanic.info()"""),

        code("""# ========================================
# 2.2 数据探索 — 拿到数据第一步做什么？
# ========================================

# 第1步: 看形状 (多少行多少列)
print(f"数据形状: {titanic.shape}")  # (891, 12) 表示891行12列

# 第2步: 看统计描述 (快速了解数值列的分布)
print("\\n=== 统计描述 ===")
print(titanic.describe())

# 第3步: 检查缺失值 (几乎每个真实数据集都有缺失值!)
print("\\n=== 缺失值统计 ===")
print(titanic.isnull().sum())  # 每列有多少个缺失值
print(f"\\n缺失率:")
print((titanic.isnull().sum() / len(titanic) * 100).round(1))  # 百分比"""),

        code("""# ========================================
# 2.3 数据筛选 — Pandas最核心的操作
# ========================================

# 1. 选择列
print("=== 选择单列 ===")
print(titanic['Name'].head())

# 2. 选择多列
print("\\n=== 选择多列 ===")
subset = titanic[['Name', 'Age', 'Fare']].head()
print(subset)

# 3. 条件筛选 (类似SQL的WHERE)
print("\\n=== 女性乘客 ===")
females = titanic[titanic['Sex'] == 'female']
print(f"女性人数: {len(females)}")

# 4. 多条件筛选 (& = AND, | = OR)
print("\\n=== 头等舱且存活的女性 ===")
result = titanic[(titanic['Pclass'] == 1) & (titanic['Survived'] == 1) & (titanic['Sex'] == 'female')]
print(f"人数: {len(result)}")

# 5. 排序
print("\\n=== 票价最高的5人 ===")
top_fare = titanic.nlargest(5, 'Fare')[['Name', 'Fare', 'Pclass']]
print(top_fare)"""),

        code("""# ========================================
# 2.4 GroupBy 分组统计 — 数据分析的灵魂操作
# ========================================
# GroupBy = 分组 + 聚合，类似SQL的 GROUP BY

# 按性别分组统计
print("=== 按性别分组 ===")
print(titanic.groupby('Sex').agg({
    'Survived': ['count', 'mean'],  # 人数和存活率
    'Age': 'mean',                   # 平均年龄
    'Fare': 'mean',                  # 平均票价
}))

# 按船舱等级分组
print("\\n=== 按船舱等级分组 ===")
print(titanic.groupby('Pclass').agg({
    'Survived': 'mean',   # 各等级存活率
    'Fare': 'mean',       # 各等级平均票价
    'Age': 'mean',        # 各等级平均年龄
}).round(2))

# 交叉统计: 性别 × 船舱等级
print("\\n=== 存活率交叉表 (性别 × 船舱) ===")
print(pd.crosstab(titanic['Sex'], titanic['Pclass'], 
                   values=titanic['Survived'], aggfunc='mean').round(2))"""),

        code("""# ========================================
# 2.5 数据清洗 — 真实数据都是脏的
# ========================================

# 创建一份"脏数据"来练习清洗
dirty_df = pd.DataFrame({
    'name': ['  Alice ', 'BOB', 'charlie', None, 'Diana', 'alice'],
    'age': [25, 30, None, 28, -5, 200],     # 有缺失值和不合理值
    'salary': ['15000', '25000元', '18,000', '22000', None, '20000'],  # 格式不统一
    'join_date': ['2024-01-15', '2024/02/20', 'Jan 10, 2024', '2024-03-01', '2024-04-05', '2024-05-10'],
})
print("=== 原始脏数据 ===")
print(dirty_df)

# 清洗1: 处理字符串 — 去空格、统一大小写
dirty_df['name'] = dirty_df['name'].str.strip().str.lower()

# 清洗2: 删除重复行
dirty_df = dirty_df.drop_duplicates(subset='name', keep='first')

# 清洗3: 处理缺失值
dirty_df['age'] = dirty_df['age'].fillna(dirty_df['age'].median())  # 用中位数填充

# 清洗4: 处理不合理值 (年龄不可能是负数或>150)
dirty_df.loc[dirty_df['age'] < 0, 'age'] = np.nan
dirty_df.loc[dirty_df['age'] > 150, 'age'] = np.nan
dirty_df['age'] = dirty_df['age'].fillna(dirty_df['age'].median())

# 清洗5: 统一薪资格式 — 提取数字
dirty_df['salary'] = dirty_df['salary'].str.replace('[^0-9]', '', regex=True)
dirty_df['salary'] = pd.to_numeric(dirty_df['salary'], errors='coerce')

print("\\n=== 清洗后 ===")
print(dirty_df)"""),

        # ===== Part 3: 可视化 =====
        md("""---
## Part 3: Matplotlib + Seaborn — 数据可视化

### 为什么要可视化？
> "一张图胜过千言万语" — 在数据分析中，可视化能帮你发现:
> - 数据的分布特征 (正态？偏态？)
> - 变量间的关系 (正相关？负相关？)
> - 异常值和数据质量问题

### 工具选择:
- **Matplotlib**: 底层引擎，功能最全但语法繁琐
- **Seaborn**: 基于Matplotlib的高级封装，一行代码出漂亮图表"""),

        code("""import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')  # 白色网格背景

# ========================================
# 3.1 基础图表 — 5种最常用的图
# ========================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('5种最常用的数据可视化图表', fontsize=16, fontweight='bold')

# 1. 折线图 — 展示趋势变化
x = np.arange(1, 13)
y = np.random.randint(100, 500, 12)
axes[0, 0].plot(x, y, 'b-o', linewidth=2, markersize=6)
axes[0, 0].set_title('折线图: 月度销售趋势')
axes[0, 0].set_xlabel('月份')
axes[0, 0].set_ylabel('销售额')

# 2. 柱状图 — 对比分类数据
categories = ['产品A', '产品B', '产品C', '产品D']
values = [320, 450, 280, 520]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
axes[0, 1].bar(categories, values, color=colors)
axes[0, 1].set_title('柱状图: 产品销量对比')

# 3. 散点图 — 展示两个变量的关系
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
axes[0, 2].scatter(x, y, alpha=0.6, c=y, cmap='viridis')
axes[0, 2].set_title('散点图: X与Y的关系')

# 4. 直方图 — 展示数据分布
data = np.random.randn(1000)
axes[1, 0].hist(data, bins=30, color='#45B7D1', alpha=0.7, edgecolor='white')
axes[1, 0].set_title('直方图: 数据分布')
axes[1, 0].axvline(data.mean(), color='red', linestyle='--', label=f'均值={data.mean():.2f}')
axes[1, 0].legend()

# 5. 箱线图 — 展示数据分布+异常值
box_data = [np.random.randn(50) * s + m for s, m in zip([1,1.5,0.8,2], [0,1,-1,0.5])]
axes[1, 1].boxplot(box_data, labels=['组A', '组B', '组C', '组D'])
axes[1, 1].set_title('箱线图: 各组分布对比')

# 隐藏空白子图
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('../data/processed/01_basic_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存!")"""),

        code("""# ========================================
# 3.2 Seaborn 高级可视化 — 一行代码出漂亮图
# ========================================

# 用Titanic数据集演示
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Seaborn: 探索性数据分析常用图表', fontsize=16, fontweight='bold')

# 1. 分布图 (KDE + 直方图)
sns.histplot(data=titanic, x='Age', hue='Survived', kde=True, ax=axes[0,0], bins=30)
axes[0,0].set_title('年龄分布 (按是否存活)')

# 2. 计数图
sns.countplot(data=titanic, x='Pclass', hue='Survived', ax=axes[0,1])
axes[0,1].set_title('各船舱等级人数 (按存活)')

# 3. 小提琴图 (箱线图+分布)
sns.violinplot(data=titanic, x='Pclass', y='Age', hue='Survived', ax=axes[0,2], split=True)
axes[0,2].set_title('年龄分布小提琴图')

# 4. 热力图 — 相关性矩阵 (特征选择必备!)
numeric_cols = titanic.select_dtypes(include=[np.number]).columns
corr_matrix = titanic[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1,0])
axes[1,0].set_title('相关性热力图')

# 5. 散点图+回归线
sns.scatterplot(data=titanic, x='Age', y='Fare', hue='Survived', ax=axes[1,1], alpha=0.5)
axes[1,1].set_title('年龄 vs 票价')

# 6. 存活率对比
survival_rate = titanic.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack()
survival_rate.plot(kind='bar', ax=axes[1,2], stacked=True, color=['#FF6B6B','#4ECDC4','#45B7D1'])
axes[1,2].set_title('存活率: 性别 × 船舱等级')
axes[1,2].set_ylabel('存活率')
axes[1,2].legend(title='船舱等级')

plt.tight_layout()
plt.savefig('../data/processed/01_seaborn_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA图表已保存!")"""),

        # ===== Part 4: Scikit-learn =====
        md("""---
## Part 4: Scikit-learn — 机器学习入门

### 什么是机器学习？
简单来说，机器学习就是**让计算机从数据中自动发现规律，然后用这些规律预测新数据**。

### 分类 vs 回归
- **分类**: 预测类别 (如: 是否存活、垃圾邮件识别)
- **回归**: 预测数值 (如: 房价、温度)

### Scikit-learn工作流
```
原始数据 → 特征工程 → 划分训练集/测试集 → 训练模型 → 评估模型
```"""),

        code("""from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Scikit-learn 导入成功!")

# ========================================
# 4.1 准备数据 — Titanic生存预测
# ========================================

# 选择用于预测的特征 (特征选择是门学问!)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

# 处理数据
ml_data = titanic[features + [target]].copy()

# 填充Age缺失值 (用中位数)
ml_data['Age'] = ml_data['Age'].fillna(ml_data['Age'].median())

# 编码性别 (male=1, female=0)
ml_data['Sex'] = LabelEncoder().fit_transform(ml_data['Sex'])

print(f"数据形状: {ml_data.shape}")
print(f"缺失值:\\n{ml_data.isnull().sum()}")
print(f"\\n前5行:")
print(ml_data.head())"""),

        code("""# ========================================
# 4.2 划分训练集和测试集
# ========================================
# 训练集: 用来训练模型 (80%)
# 测试集: 用来评估模型 (20%) — 模拟"从未见过的数据"

X = ml_data[features]
y = ml_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20%作为测试集
    random_state=42,     # 固定随机种子
    stratify=y           # 保持正负样本比例一致
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"训练集存活率: {y_train.mean():.2%}")
print(f"测试集存活率: {y_test.mean():.2%}")

# 特征标准化 (让所有特征在同一尺度上)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 注意: 测试集只用transform!

print(f"\\n标准化后 - 训练集均值: {X_train_scaled.mean(axis=0).round(2)}")  # 接近0
print(f"标准化后 - 训练集标准差: {X_train_scaled.std(axis=0).round(2)}")    # 接近1"""),

        code("""# ========================================
# 4.3 训练3个模型并对比
# ========================================

models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '决策树': DecisionTreeClassifier(random_state=42, max_depth=5),
    '随机森林': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
}

results = {}

print("=" * 60)
print("模型对比实验")
print("=" * 60)

for name, model in models.items():
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)  # 5折交叉验证
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }
    
    print(f"\\n--- {name} ---")
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['未存活', '存活']))"""),

        code("""# ========================================
# 4.4 可视化对比结果
# ========================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 准确率对比
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0].bar(x - width/2, accuracies, width, label='测试集准确率', color='#4ECDC4')
axes[0].bar(x + width/2, cv_means, width, label='交叉验证均值', color='#FF6B6B',
            yerr=cv_stds, capsize=5)
axes[0].set_ylabel('准确率')
axes[0].set_title('模型准确率对比')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names)
axes[0].legend()
axes[0].set_ylim(0.7, 0.9)

# 最佳模型的混淆矩阵
best_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['未存活', '存活'], yticklabels=['未存活', '存活'])
axes[1].set_title('随机森林 - 混淆矩阵')
axes[1].set_xlabel('预测值')
axes[1].set_ylabel('真实值')

plt.tight_layout()
plt.savefig('../data/processed/01_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n恭喜! 你已经完成了第一个端到端的机器学习项目!")"""),

        md("""---
## 本周总结 & 下一步

### 你已经学会了:
1. **NumPy**: 数组创建、运算、索引、统计操作
2. **Pandas**: DataFrame操作、数据筛选、GroupBy、数据清洗
3. **可视化**: Matplotlib基础图表、Seaborn统计图表、热力图
4. **机器学习**: Scikit-learn工作流、3种分类算法、模型评估

### 核心知识点检查清单:
- [ ] 能用 `np.array` 创建数组并进行运算
- [ ] 能用 `pd.read_csv` 读取数据并用 `head/describe/info` 探索
- [ ] 能用布尔索引和GroupBy进行数据筛选和统计
- [ ] 能画折线图、柱状图、散点图、直方图、热力图
- [ ] 能用 `train_test_split` 划分数据并用 `StandardScaler` 标准化
- [ ] 能训练一个分类模型并用 `accuracy_score` 评估

### 下一步 (Week 5+):
- 学习回归模型 (LinearRegression, Ridge, Lasso)
- 学习Pipeline和超参数调优 (GridSearchCV)
- 进入NLP和文本分类领域
- 开始使用项目模块: `doc_parser.py`, `nlp_pipeline.py`"""),
    ]
    make_notebook(cells, "01_numpy_pandas_basics.ipynb")


# ========================================================================
# Notebook 02: 文档解析实验 (CV/OCR)
# ========================================================================
def gen_notebook_02():
    cells = [
        md("""# 文档解析实验: PDF文本/表格/图像提取

> **学习目标**: 理解PDF文档结构，掌握PyMuPDF和pdfplumber库，学会使用智析项目的 `doc_parser.py` 模块
>
> **对应项目模块**: `src/doc_parser.py` (CV层)
>
> **预计时间**: 2-3天
>
> **前置条件**: `pip install PyMuPDF pdfplumber Pillow tqdm`

---

## 背景知识

### PDF文档是什么？
PDF（Portable Document Format）不只是"一张图片"，它是一个**结构化的容器**，内部包含:
- **文本对象**: 带有字体、大小、位置信息的文字
- **图像对象**: 嵌入的图片（JPEG/PNG等）
- **矢量图形**: 线条、表格框线
- **页面结构**: 每页有独立的坐标系

### 为什么要解析PDF？
在智析项目中，PDF解析是**整个系统的入口**:
```
PDF文档 → [文档解析] → 纯文本+表格+图像 → [NLP分析] → [知识图谱] → [RAG问答]
```
如果解析不好，后续所有模块都会受影响。所以这一步非常重要！"""),

        md("""## Part 1: PyMuPDF基础 — 直接操作PDF

### 什么是PyMuPDF？
PyMuPDF（导入名为 `fitz`）是Python中最快最全面的PDF处理库。它可以:
- 提取文本（包括中文）
- 提取嵌入图像
- 获取页面尺寸和布局信息
- 创建和修改PDF"""),

        code("""import fitz  # PyMuPDF的导入名是fitz
import os

print(f"PyMuPDF 版本: {fitz.version}")

# ========================================
# 1.1 创建一个测试PDF (方便后续实验)
# ========================================
os.makedirs('../data/sample_docs', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)

doc = fitz.open()

# 第1页: 项目介绍
page = doc.new_page()
# insert_text(位置, 文本, 字号)  位置是(x, y)坐标，单位是"点"(1点=1/72英寸)
page.insert_text((72, 72), '智析 (ZhiXi) - 多模态文档智能分析平台', fontsize=20)
page.insert_text((72, 110), '项目概述:', fontsize=14)
page.insert_text((72, 135), '智析是一个能够自动解析、理解、检索和问答行业研究报告的智能平台。', fontsize=11)
page.insert_text((72, 155), '用户上传PDF研报，系统自动提取信息、建立知识索引，', fontsize=11)
page.insert_text((72, 175), '用户可以通过自然语言对话获取洞察。', fontsize=11)
page.insert_text((72, 210), '技术栈:', fontsize=14)
page.insert_text((72, 235), '1. 文档解析层 (CV): PyMuPDF + pdfplumber + PaddleOCR', fontsize=11)
page.insert_text((72, 255), '2. NLP分析层: HuggingFace Transformers + spaCy + KeyBERT', fontsize=11)
page.insert_text((72, 275), '3. 知识构建层 (数据挖掘): NetworkX + Scikit-learn', fontsize=11)
page.insert_text((72, 295), '4. RAG问答层 (LLM): LangChain + ChromaDB + OpenAI/Ollama', fontsize=11)
page.insert_text((72, 315), '5. Web界面: Streamlit', fontsize=11)

# 第2页: 系统架构
page2 = doc.new_page()
page2.insert_text((72, 72), '系统架构设计', fontsize=20)
page2.insert_text((72, 110), '采用四层架构，每层职责明确:', fontsize=12)
page2.insert_text((72, 140), '文档解析层 → NLP分析层 → 知识构建层 → RAG问答层', fontsize=14)
page2.insert_text((72, 180), '核心设计原则:', fontsize=14)
page2.insert_text((72, 210), '- 延迟加载: 模型按需初始化，节省内存', fontsize=11)
page2.insert_text((72, 230), '- 降级策略: 每个模块都有错误处理，保证系统不崩溃', fontsize=11)
page2.insert_text((72, 250), '- 模块化: 每个层可以独立开发和测试', fontsize=11)
page2.insert_text((72, 290), '关键数据指标:', fontsize=14)
page2.insert_text((72, 315), '- 支持PDF页数: 1-1000页', fontsize=11)
page2.insert_text((72, 335), '- 文本提取准确率: >95%', fontsize=11)
page2.insert_text((72, 355), '- 问答响应时间: <3秒', fontsize=11)

# 第3页: 更多细节
page3 = doc.new_page()
page3.insert_text((72, 72), 'RAG检索增强生成', fontsize=20)
page3.insert_text((72, 110), 'RAG (Retrieval-Augmented Generation) 是当前最流行的LLM应用架构。', fontsize=12)
page3.insert_text((72, 140), '核心流程:', fontsize=14)
page3.insert_text((72, 170), '1. 文档切块 (Chunking): 将长文档分成500字符的小块', fontsize=11)
page3.insert_text((72, 190), '2. 向量化 (Embedding): 将文本块转换为高维向量', fontsize=11)
page3.insert_text((72, 210), '3. 存储 (Indexing): 将向量存入ChromaDB向量数据库', fontsize=11)
page3.insert_text((72, 230), '4. 检索 (Retrieval): 用户提问时，找到最相关的文档块', fontsize=11)
page3.insert_text((72, 250), '5. 生成 (Generation): 将检索到的内容作为上下文，让LLM生成回答', fontsize=11)

test_pdf_path = '../data/sample_docs/test.pdf'
doc.save(test_pdf_path)
doc.close()
print(f"测试PDF已创建: {test_pdf_path}")
print(f"文件大小: {os.path.getsize(test_pdf_path)} bytes")"""),

        code("""# ========================================
# 1.2 打开PDF并提取文本 (最基础的操作)
# ========================================

doc = fitz.open(test_pdf_path)

print(f"文件: {doc.name}")
print(f"页数: {len(doc)}")
print(f"元数据: {doc.metadata}")
print()

# 逐页提取文本
for page_num, page in enumerate(doc):
    text = page.get_text("text")  # "text"模式提取纯文本
    print(f"{'='*50}")
    print(f"第 {page_num + 1} 页 (尺寸: {page.rect.width:.0f} x {page.rect.height:.0f} 点)")
    print(f"{'='*50}")
    print(text[:300])  # 只看前300字符
    print()

doc.close()"""),

        code("""# ========================================
# 1.3 不同的文本提取模式
# ========================================

doc = fitz.open(test_pdf_path)
page = doc[0]  # 第1页

# 模式1: "text" — 纯文本 (最常用)
text_mode = page.get_text("text")
print("=== 模式: text ===")
print(text_mode[:200])

# 模式2: "dict" — 结构化信息 (包含字体、颜色、位置)
dict_mode = page.get_text("dict")
print(f"\\n=== 模式: dict (包含 {len(dict_mode['blocks'])} 个文本块) ===")
for block in dict_mode['blocks'][:2]:
    if 'lines' in block:
        for line in block['lines'][:1]:
            for span in line['spans'][:1]:
                print(f"  文本: '{span['text'][:50]}'")
                print(f"  字体: {span['font']}, 大小: {span['size']:.1f}")
                print(f"  位置: ({span['bbox'][0]:.0f}, {span['bbox'][1]:.0f})")

# 模式3: "blocks" — 按文本块提取 (保留布局)
blocks = page.get_text("blocks")
print(f"\\n=== 模式: blocks ({len(blocks)} 个块) ===")
for block in blocks[:3]:
    print(f"  位置: ({block[0]:.0f},{block[1]:.0f})-({block[2]:.0f},{block[3]:.0f}) | 文本: {block[4][:60].strip()}")

doc.close()"""),

        code("""# ========================================
# 1.4 提取PDF中的嵌入图像
# ========================================

# 先创建一个带图片的PDF
doc_with_img = fitz.open()
page = doc_with_img.new_page()
page.insert_text((72, 72), '这是一份带图片的文档', fontsize=16)

# 创建一个简单的彩色图像并嵌入
from PIL import Image
import io

img = Image.new('RGB', (200, 100), color='#4ECDC4')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

page.insert_image(fitz.Rect(72, 120, 272, 220), stream=img_bytes.getvalue())

img_pdf_path = '../data/sample_docs/with_image.pdf'
os.makedirs(os.path.dirname(img_pdf_path), exist_ok=True)
doc_with_img.save(img_pdf_path)
doc_with_img.close()
print(f"带图片PDF已创建: {img_pdf_path}")

# 提取图像
doc = fitz.open(img_pdf_path)
page = doc[0]
image_list = page.get_images(full=True)
print(f"\\n发现 {len(image_list)} 张图像")

os.makedirs('../data/processed/test_images', exist_ok=True)
for img_idx, img_info in enumerate(image_list):
    xref = img_info[0]
    base_image = doc.extract_image(xref)
    img_bytes = base_image["image"]
    img_ext = base_image.get("ext", "png")
    
    save_path = f'../data/processed/test_images/img_{img_idx + 1}.{img_ext}'
    with open(save_path, 'wb') as f:
        f.write(img_bytes)
    
    print(f"  图像{img_idx+1}: {len(img_bytes)} bytes, 格式: {img_ext}")
    print(f"  保存至: {save_path}")

doc.close()"""),

        md("""## Part 2: pdfplumber — 表格提取

### 为什么需要pdfplumber？
PyMuPDF擅长提取文本和图像，但对**表格**的处理不够好。pdfplumber专门用于PDF表格提取，它能:
- 识别表格的行列结构
- 处理合并单元格
- 提取为结构化的Python列表"""),

        code("""import pdfplumber

# 创建一个带表格的PDF
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 50), 'AI行业研究报告 - 数据表格', fontsize=16)

# 用PyMuPDF画一个简单的表格
# (实际项目中PDF已经有表格了，这里是为了演示)
table_data = [
    ['指标', '2023年', '2024年', '增长率'],
    ['全球AI市场规模', '$1500亿', '$2000亿', '33%'],
    ['LLM应用数量', '5000+', '20000+', '300%'],
    ['AI工程师需求', '100K', '250K', '150%'],
    ['开源模型数量', '500+', '2000+', '300%'],
]

y_start = 90
row_height = 25
col_widths = [150, 80, 80, 80]
x_start = 72

for row_idx, row in enumerate(table_data):
    x = x_start
    y = y_start + row_idx * row_height
    for col_idx, cell in enumerate(row):
        # 画单元格边框
        rect = fitz.Rect(x, y, x + col_widths[col_idx], y + row_height)
        page.draw_rect(rect, color=(0, 0, 0), width=0.5)
        # 写入文本
        fontsize = 10 if row_idx == 0 else 9
        page.insert_text((x + 5, y + 17), cell, fontsize=fontsize)
        x += col_widths[col_idx]

table_pdf_path = '../data/sample_docs/with_table.pdf'
doc.save(table_pdf_path)
doc.close()
print(f"带表格PDF已创建")

# 用pdfplumber提取表格
with pdfplumber.open(table_pdf_path) as pdf:
    page = pdf.pages[0]
    tables = page.extract_tables()
    
    print(f"\\n发现 {len(tables)} 个表格")
    
    for t_idx, table in enumerate(tables):
        print(f"\\n=== 表格 {t_idx + 1} ({len(table)} 行) ===")
        for row in table:
            print(f"  {row}")
        
        # 转换为Pandas DataFrame
        if table:
            import pandas as pd
            headers = table[0]
            data = table[1:]
            df_table = pd.DataFrame(data, columns=headers)
            print(f"\\n--- DataFrame格式 ---")
            print(df_table)"""),

        md("""## Part 3: 使用智析项目的 doc_parser.py

现在你已经理解了PDF解析的底层原理，让我们来使用项目中封装好的模块。

### `doc_parser.py` 的设计思路:
1. **DocumentParser类**: 封装了PyMuPDF + pdfplumber的调用
2. **parse()方法**: 一次调用完成所有提取（文本+表格+图像）
3. **get_text_chunks()方法**: 将文本切分为适合RAG的块
4. **DocumentResult**: 统一的结果数据结构"""),

        code("""import sys
sys.path.insert(0, '..')

from src.doc_parser import DocumentParser, parse_pdf

# ========================================
# 3.1 一键解析PDF
# ========================================

# 使用封装好的函数
result = parse_pdf(test_pdf_path, output_dir='../data/processed', save_result=True)

print(f"文件名: {result.filename}")
print(f"页数: {result.total_pages}")
print(f"总字符数: {len(result.full_text)}")
print(f"\\n每页内容预览:")
for page in result.pages:
    print(f"  第{page.page_number}页: {len(page.text)}字符, {len(page.tables)}个表格, {len(page.images)}张图像")"""),

        code("""# ========================================
# 3.2 文本切块 — RAG的关键步骤
# ========================================
# 
# 为什么需要切块？
# 1. LLM有上下文长度限制 (GPT-4o: 128K tokens)
# 2. 嵌入向量对短文本效果更好
# 3. 检索时需要精确匹配到相关段落
#
# chunk_size: 每个块的字符数 (太小会丢失上下文，太大会降低检索精度)
# chunk_overlap: 相邻块的重叠字符 (防止关键信息被切断)

parser = DocumentParser(test_pdf_path, output_dir='../data/processed')

# 尝试不同的切块策略
for chunk_size in [200, 500, 1000]:
    chunks = parser.get_text_chunks(chunk_size=chunk_size, chunk_overlap=50)
    print(f"\\n=== chunk_size={chunk_size} ===")
    print(f"切分为 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):  # 只看前3个
        print(f"  块{i} (页{chunk['page']}): {chunk['text'][:80]}...")
    print(f"  ...")"""),

        code("""# ========================================
# 3.3 完整的端到端流程演示
# ========================================

print("=" * 60)
print("智析 - 文档解析完整流程演示")
print("=" * 60)

# Step 1: 解析
print("\\n[Step 1] 解析PDF...")
result = parse_pdf(test_pdf_path, save_result=False)

# Step 2: 查看统计
print(f"\\n[Step 2] 解析统计:")
print(f"  总页数: {result.total_pages}")
print(f"  总字符: {len(result.full_text)}")
tables = sum(len(p.tables) for p in result.pages)
images = sum(len(p.images) for p in result.pages)
print(f"  表格数: {tables}")
print(f"  图像数: {images}")

# Step 3: 切块
print(f"\\n[Step 3] 文本切块...")
chunks = parser.get_text_chunks(chunk_size=500, chunk_overlap=50)
print(f"  切分结果: {len(chunks)} 个块")
avg_len = sum(len(c['text']) for c in chunks) / len(chunks)
print(f"  平均块长度: {avg_len:.0f} 字符")

# Step 4: 保存结果
print(f"\\n[Step 4] 保存解析结果...")
result.save('../data/processed/parse_result.json')

print(f"\\n{'=' * 60}")
print("文档解析完成! 下一步: 运行 03_rag_experiment.ipynb 体验RAG问答")"""),

        md("""---
## 知识总结

### 你学到了什么:
1. **PDF结构**: 文本对象、图像对象、页面坐标系
2. **PyMuPDF (fitz)**: 文本提取（3种模式）、图像提取、页面信息
3. **pdfplumber**: 表格识别和结构化提取
4. **doc_parser.py**: 项目封装的解析模块，一键完成全流程
5. **文本切块**: chunk_size和chunk_overlap的含义和调优

### 核心概念:
| 概念 | 含义 | 在智析中的作用 |
|------|------|--------------|
| `page.get_text("text")` | 提取纯文本 | 获取文档内容 |
| `page.get_images()` | 获取图像列表 | 提取图表用于分析 |
| `page.extract_tables()` | 提取表格结构 | 获取数据表格 |
| `chunk_size` | 文本块大小 | 影响RAG检索精度 |
| `chunk_overlap` | 块间重叠 | 防止信息丢失 |

### 下一步:
- 用**真实论文/研报PDF**测试解析效果
- 运行 `04_nlp_analysis.ipynb` 学习NLP分析
- 运行 `03_rag_experiment.ipynb` 体验智能问答"""),
    ]
    make_notebook(cells, "02_ocr_experiment.ipynb")


# ========================================================================
# Notebook 03: RAG问答实验
# ========================================================================
def gen_notebook_03():
    cells = [
        md("""# RAG问答实验: 从原理到实践的检索增强生成

> **学习目标**: 理解RAG架构原理，掌握向量数据库和Embedding，学会使用智析项目的 `rag_engine.py`
>
> **对应项目模块**: `src/rag_engine.py` (LLM应用层)
>
> **预计时间**: 3-4天
>
> **前置条件**: 
> - 已完成 Notebook 02 (文档解析)
> - `pip install langchain langchain-openai chromadb python-dotenv`
> - 配置好 `.env` 文件 (或安装Ollama)

---

## 背景知识: 什么是RAG？

### LLM的局限性
大语言模型（如GPT-4）虽然很强大，但有三个核心问题:
1. **知识截止**: 训练数据有截止日期，不知道最新信息
2. **幻觉**: 会编造看起来合理但实际错误的信息
3. **无法访问私有数据**: 不知道你的文档里写了什么

### RAG如何解决这些问题？
RAG (Retrieval-Augmented Generation) = **检索 + 生成**

```
用户提问 → 检索相关文档 → 将文档作为上下文 → 让LLM基于文档回答
```

**核心思想**: 不让LLM"凭空回答"，而是先找到相关信息，再让LLM基于这些信息生成答案。
这样就能:
- 回答最新信息 (只要文档是最新的)
- 减少幻觉 (有文档作为依据)
- 访问私有数据 (你自己的文档)"""),

        md("""### RAG的5个核心步骤

| 步骤 | 名称 | 做什么 | 智析中的技术 |
|------|------|--------|-------------|
| 1 | Chunking | 将长文档切成小块 | `doc_parser.get_text_chunks()` |
| 2 | Embedding | 将文本块转换为向量 | OpenAI Embedding |
| 3 | Indexing | 将向量存入数据库 | ChromaDB |
| 4 | Retrieval | 用户提问时检索最相关的块 | 向量相似度搜索 |
| 5 | Generation | 将检索结果构建Prompt，LLM生成答案 | OpenAI/Ollama |

### 什么是Embedding（向量嵌入）？
Embedding是将文本转换为一个高维向量（如1536维的数字数组）。
- 语义相似的文本 → 向量距离近
- 语义不同的文本 → 向量距离远

例如:
- "人工智能"和"机器学习" → 向量很接近
- "人工智能"和"香蕉" → 向量很远

这就是为什么RAG能找到"语义相关"的文档，而不只是"关键词匹配"。"""),

        md("""## Part 1: 理解Embedding — 向量化的魔力

> 注意: 下面的实验需要OpenAI API Key。如果你没有，可以跳过Part 1直接到Part 2。"""),

        code("""import os
import sys
sys.path.insert(0, '..')

from dotenv import load_dotenv
load_dotenv()  # 从.env文件加载API Key

# 检查是否有API Key
has_api_key = bool(os.getenv('OPENAI_API_KEY'))
print(f"OpenAI API Key: {'已配置' if has_api_key else '未配置'}")

if has_api_key:
    # ========================================
    # 1.1 体验Embedding
    # ========================================
    from openai import OpenAI
    client = OpenAI()
    
    # 将文本转换为向量
    texts = [
        "人工智能正在改变世界",
        "机器学习是AI的核心技术",
        "今天天气真好，适合出去散步",
        "深度学习模型需要大量数据训练",
    ]
    
    embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    response = client.embeddings.create(input=texts, model=embedding_model)
    
    embeddings = [item.embedding for item in response.data]
    dim = len(embeddings[0])
    print(f"\\nEmbedding维度: {dim}")  # 1536维
    print(f"第1个文本的向量前10个值: {embeddings[0][:10]}")
    
    # ========================================
    # 1.2 计算文本相似度
    # ========================================
    import numpy as np
    
    def cosine_similarity(a, b):
        \"\"\"余弦相似度: 1=完全相同, 0=无关, -1=完全相反\"\"\"
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\\n=== 文本相似度矩阵 ===")
    print(f"{'':>30}", end="")
    for t in texts:
        print(f"{t[:12]:>15}", end="")
    print()
    
    for i, t1 in enumerate(texts):
        print(f"{t1[:12]:>30}", end="")
        for j, t2 in enumerate(texts):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"{sim:>15.3f}", end="")
        print()
    
    print("\\n观察: AI相关的文本之间相似度更高!")
else:
    print("\\n跳过Embedding实验 (需要OpenAI API Key)")
    print("你可以:")
    print("  1. 在.env文件中配置OPENAI_API_KEY")
    print("  2. 或者安装Ollama使用本地模型")"""),

        md("""## Part 2: ChromaDB — 向量数据库

### 什么是向量数据库？
传统数据库用SQL查询，向量数据库用"向量距离"查询。你给它一个向量，它能快速找到最相似的向量。

### 为什么用ChromaDB？
- **纯Python**: 不需要安装数据库服务器
- **嵌入式**: 直接在Python代码中运行
- **轻量**: 适合开发和小型项目
- **持久化**: 可以保存到磁盘，下次启动不用重建"""),

        code("""import chromadb

# ========================================
# 2.1 创建ChromaDB客户端和集合
# ========================================

# 使用内存模式 (不保存到磁盘，适合实验)
chroma_client = chromadb.Client()

# 创建一个集合 (collection) — 类似数据库中的"表"
collection = chroma_client.create_collection(
    name="test_docs",
    metadata={"description": "智析项目测试集合"}
)

print(f"集合已创建: {collection.name}")
print(f"当前文档数: {collection.count()}")

# ========================================
# 2.2 添加文档 (不需要Embedding模型也能测试!)
# ========================================
# ChromaDB有内置的Embedding函数 (默认用all-MiniLM-L6-v2)

# 模拟从PDF解析出的文档块
documents = [
    "智析是一个多模态文档智能分析平台，能解析PDF中的文字和图表。",
    "系统采用四层架构: 文档解析层、NLP分析层、知识构建层、RAG问答层。",
    "RAG代表检索增强生成，是当前最流行的LLM应用架构。",
    "ChromaDB是一个嵌入式向量数据库，纯Python实现，无需额外部署。",
    "LangChain是一个LLM应用开发框架，支持RAG、Agent等模式。",
    "PaddleOCR是百度开源的OCR工具，中文识别率非常高。",
    "知识图谱用NetworkX构建，能展示实体之间的关系。",
    "Streamlit可以在30行代码内创建一个Web应用。",
]

metadatas = [
    {"page": 1, "source": "项目介绍"},
    {"page": 1, "source": "项目介绍"},
    {"page": 2, "source": "RAG说明"},
    {"page": 2, "source": "技术说明"},
    {"page": 3, "source": "技术说明"},
    {"page": 1, "source": "技术说明"},
    {"page": 3, "source": "知识图谱"},
    {"page": 3, "source": "Web界面"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)

print(f"\\n已添加 {collection.count()} 个文档块")
print(f"文档示例: {documents[0][:60]}...")"""),

        code("""# ========================================
# 2.3 语义搜索 — RAG的核心
# ========================================

# 提问: 找到最相关的文档块
queries = [
    "这个项目用了什么技术？",
    "什么是RAG？",
    "怎么识别图片中的文字？",
]

for query in queries:
    print(f"\\n{'='*50}")
    print(f"查询: '{query}'")
    print(f"{'='*50}")
    
    results = collection.query(
        query_texts=[query],
        n_results=3,  # 返回最相关的3个文档
    )
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
    )):
        print(f"\\n  [{i+1}] 相关度距离: {dist:.4f} (越小越相关)")
        print(f"      来源: {meta['source']}, 页码: {meta['page']}")
        print(f"      内容: {doc}")"""),

        code("""# ========================================
# 2.4 带过滤条件的搜索
# ========================================

# 只搜索特定来源的文档
results = collection.query(
    query_texts=["项目架构是什么？"],
    n_results=5,
    where={"source": "项目介绍"},  # 只从"项目介绍"中搜索
)

print("=== 过滤搜索: 只搜索'项目介绍' ===")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"  [{i+1}] [{meta['source']}] {doc[:80]}")

# 查看所有文档
print(f"\\n=== 集合中的所有文档 ({collection.count()}个) ===")
all_docs = collection.get()
for doc_id, doc, meta in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
    print(f"  {doc_id}: [{meta['source']}] {doc[:60]}...")"""),

        md("""## Part 3: 使用智析的 RAG 引擎

现在我们已经理解了底层原理，来使用项目中封装好的 `rag_engine.py`。

### `RAGEngine` 类的设计:
- **延迟初始化**: 模型在第一次使用时才加载，节省内存
- **双模式**: 支持OpenAI API和本地Ollama
- **完整链路**: `ingest_documents()` → `ask()` 两步完成"""),

        code("""# ========================================
# 3.1 完整RAG流程演示
# ========================================

from src.doc_parser import DocumentParser
from src.rag_engine import RAGEngine, RAGAnswer

# Step 1: 解析PDF并切块
print("[Step 1] 解析PDF...")
parser = DocumentParser(test_pdf_path, output_dir='../data/processed')
chunks = parser.get_text_chunks(chunk_size=500, chunk_overlap=50)
print(f"  切分为 {len(chunks)} 个块\\n")

# Step 2: 初始化RAG引擎
print("[Step 2] 初始化RAG引擎...")

# 选择模式 (根据你的环境选择)
USE_OLLAMA = not has_api_key  # 没有API Key就用Ollama

if USE_OLLAMA:
    print("  使用 Ollama 本地模式")
    engine = RAGEngine(
        use_ollama=True,
        ollama_model="qwen2.5:7b",
    )
else:
    print("  使用 OpenAI API 模式")
    engine = RAGEngine(
        use_ollama=False,
        openai_model="gpt-4o-mini",  # 最便宜的模型
    )

# Step 3: 导入文档
print("\\n[Step 3] 导入文档到向量数据库...")
engine.ingest_documents(chunks)

# Step 4: 提问!
print("\\n[Step 4] 开始问答!")
questions = [
    "这个项目叫什么名字？它的主要功能是什么？",
    "系统采用了什么架构？有几层？",
    "RAG是什么？它的工作流程是怎样的？",
]

for q in questions:
    print(f"\\n{'='*60}")
    print(f"Q: {q}")
    answer = engine.ask(q, top_k=3)
    print(f"A: {answer.answer}")
    print(f"\\n使用的模型: {answer.model_used}")
    print(f"参考来源数: {len(answer.sources)}")
    for src in answer.sources:
        print(f"  - 第{src['page']}页: {src['content'][:80]}...")"""),

        code("""# ========================================
# 3.2 纯检索模式 (不生成回答，只看检索结果)
# ========================================

print("=== 检索测试 (不消耗LLM tokens) ===\\n")

search_queries = [
    "文本切块",
    "向量数据库",
    "知识图谱",
]

for query in search_queries:
    results = engine.search(query, top_k=2)
    print(f"查询: '{query}'")
    for i, r in enumerate(results):
        print(f"  [{i+1}] 第{r['page']}页: {r['content'][:100]}...")
    print()"""),

        md("""## Part 4: 理解RAG Prompt工程

### 为什么Prompt很重要？
RAG的效果很大程度上取决于你如何构建Prompt。好的Prompt应该:
1. 明确告诉LLM"只基于文档回答"
2. 提供清晰的文档上下文
3. 允许LLM说"我不知道"

### 看看智析项目中的Prompt模板:"""),

        code("""# 查看RAGEngine使用的Prompt模板
from src.rag_engine import RAGEngine

engine = RAGEngine()

# 模拟一个问答场景
question = "ChromaDB有什么特点？"
context = \"\"\"[文档片段 1]
ChromaDB是一个嵌入式向量数据库，纯Python实现，无需额外部署。
它支持持久化存储，可以将向量数据保存到磁盘。

[文档片段 2]
ChromaDB适合开发阶段和小型项目使用，
对于大规模生产环境，建议使用FAISS或Milvus。\"\"\"

# 查看生成的完整Prompt
prompt = engine._build_prompt(question, context)
print("=== RAG Prompt 模板 ===")
print(prompt)
print("\\n" + "="*60)
print("观察: Prompt中包含了:")
print("  1. 系统指令 (只基于文档回答)")
print("  2. 检索到的文档内容")
print("  3. 用户问题")
print("  4. 明确的格式要求")"""),

        md("""---
## 知识总结

### 你学到了什么:
1. **RAG原理**: 检索+生成，解决LLM的知识截止和幻觉问题
2. **Embedding**: 将文本转换为向量，实现语义搜索
3. **ChromaDB**: 嵌入式向量数据库，添加/查询/过滤
4. **RAGEngine**: 项目封装的RAG引擎，两步完成全流程
5. **Prompt工程**: RAG问答的Prompt模板设计

### RAG参数调优指南:
| 参数 | 作用 | 调优建议 |
|------|------|----------|
| `chunk_size` | 文本块大小 | 200-1000, 默认500 |
| `chunk_overlap` | 块间重叠 | 通常为chunk_size的10% |
| `top_k` | 检索数量 | 3-5, 太多会引入噪声 |
| `temperature` | 生成随机性 | 0.1(精确)到0.7(创造性) |

### 常见问题排查:
| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 回答不准确 | chunk_size太小 | 增大chunk_size |
| 回答"不知道" | 检索不到相关内容 | 增大top_k或调整切块策略 |
| 回答太慢 | 模型太大 | 换用gpt-4o-mini或量化模型 |
| API报错 | Key无效或余额不足 | 检查.env配置 |

### 下一步:
- 用**真实研报/论文**测试完整流程
- 运行 `04_nlp_analysis.ipynb` 学习NLP分析
- 运行 `05_knowledge_graph.ipynb` 构建知识图谱
- 运行 `streamlit run src/app.py` 体验Web界面"""),
    ]
    make_notebook(cells, "03_rag_experiment.ipynb")


# ========================================================================
# Notebook 04-06 (精简版)
# ========================================================================
def gen_notebook_04():
    cells = [
        md("""# NLP分析实验: 实体识别/关键词提取/摘要生成

> **对应模块**: `src/nlp_pipeline.py` (NLP分析层)
> **前置条件**: `pip install transformers keybert wordcloud torch`"""),

        md("""## Part 1: 什么是NLP？

**NLP (自然语言处理)** 是让计算机理解人类语言的AI分支。

在智析项目中，NLP用于:
1. **命名实体识别 (NER)**: 找出文本中的人名、组织、地点
2. **关键词提取**: 用KeyBERT基于语义找到核心词汇
3. **自动摘要**: 用BART模型生成文档摘要
4. **词云**: 可视化展示文本中的高频词汇"""),

        code("""import sys
sys.path.insert(0, '..')
from src.nlp_pipeline import NLPPipeline, Entity, NLPResult, analyze_text

# 测试文本 (英文效果最好，因为默认模型是英文的)
sample_text = \"\"\"
Artificial intelligence (AI) is transforming industries worldwide. 
Companies like Google, Microsoft, and OpenAI are leading the development 
of large language models. In 2024, the global AI market was valued at 
over $200 billion. Researchers at Stanford University and MIT have 
published groundbreaking papers on transformer architectures and 
reinforcement learning from human feedback (RLHF). The technology 
has applications in healthcare, finance, education, and autonomous 
vehicles. Sundar Pichai, CEO of Google, announced a $2 billion 
investment in AI research.
\"\"\"

print("=== 测试文本 ===")
print(sample_text[:200] + "...")"""),

        code("""# ========================================
# 1.1 关键词提取 (KeyBERT) — 最先运行，不需要大模型
# ========================================

nlp = NLPPipeline()

print("[1] 关键词提取...")
keywords = nlp.extract_keywords(sample_text, top_k=10)

print("\\n提取到的关键词 (按相关度排序):")
for kw, score in keywords:
    bar = '█' * int(score * 30)
    print(f"  {kw:>30s}  {score:.4f}  {bar}")"""),

        code("""# ========================================
# 1.2 命名实体识别 (NER)
# ========================================
# 注意: 首次运行需要下载模型 (~400MB)

print("[2] 命名实体识别...")
entities = nlp.extract_entities(sample_text)

print(f"\\n识别到 {len(entities)} 个实体:")

# 按类型分组
from collections import defaultdict
by_type = defaultdict(list)
for e in entities:
    by_type[e.label].append(e.text)

type_names = {'PER': '人名', 'ORG': '组织', 'LOC': '地点', 'MISC': '其他'}
for label, names in by_type.items():
    label_cn = type_names.get(label, label)
    unique_names = list(set(names))
    print(f"\\n  {label_cn} ({label}): {', '.join(unique_names)}")"""),

        code("""# ========================================
# 1.3 自动摘要生成 (BART)
# ========================================
# 注意: 首次运行需要下载模型 (~1.5GB)
# 如果内存不够可以跳过这步

print("[3] 自动摘要生成...")
try:
    summary = nlp.generate_summary(sample_text)
    print(f"\\n原文长度: {len(sample_text)} 字符")
    print(f"摘要长度: {len(summary)} 字符")
    print(f"\\n摘要内容:\\n{summary}")
except Exception as e:
    print(f"摘要生成跳过: {e}")
    print("(可能原因: 内存不足或模型下载失败)")"""),

        code("""# ========================================
# 1.4 词云可视化
# ========================================
import os
os.makedirs('../data/processed', exist_ok=True)

print("[4] 词云生成...")
path = nlp.generate_wordcloud(sample_text, '../data/processed/wordcloud.png')
if path:
    from PIL import Image
    import matplotlib.pyplot as plt
    
    img = Image.open(path)
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('关键词词云')
    plt.tight_layout()
    plt.show()
    print(f"词云已保存: {path}")"""),

        code("""# ========================================
# 1.5 一键分析 (analyze方法)
# ========================================

print("=== 一键NLP分析 ===")
result = analyze_text(sample_text)

print(f"词数: {result.word_count}")
print(f"实体数: {len(result.entities)}")
print(f"关键词数: {len(result.keywords)}")
print(f"摘要长度: {len(result.summary)} 字符")

# 转为字典 (方便保存为JSON)
result_dict = result.to_dict()
print(f"\\n结果字典键: {list(result_dict.keys())}")"""),

        md("""---
## 总结

### NLPPipeline 提供的方法:
| 方法 | 功能 | 模型 | 首次运行 |
|------|------|------|---------|
| `extract_keywords()` | 关键词提取 | KeyBERT | 下载~400MB |
| `extract_entities()` | 实体识别 | BERT-NER | 下载~400MB |
| `generate_summary()` | 摘要生成 | BART | 下载~1.5GB |
| `generate_wordcloud()` | 词云图 | 无需模型 | 即时可用 |
| `analyze()` | 一键全部分析 | 全部 | 全部下载 |

### 下一步: 运行 `05_knowledge_graph.ipynb` 构建知识图谱"""),
    ]
    make_notebook(cells, "04_nlp_analysis.ipynb")


def gen_notebook_05():
    cells = [
        md("""# 知识图谱实验: NetworkX图结构与实体关系

> **对应模块**: `src/knowledge_graph.py` (数据挖掘层)
> **前置条件**: `pip install networkx matplotlib scikit-learn`"""),

        md("""## 什么是知识图谱？

知识图谱是一种**用图结构表示实体和关系**的数据模型:
- **节点 (Node)**: 代表实体（人、组织、概念等）
- **边 (Edge)**: 代表实体间的关系
- 例如: `Google --owns--> DeepMind --created--> AlphaGo`

在智析项目中，知识图谱从NLP识别的实体中自动构建，
用于展示文档中实体之间的关联，帮助用户理解文档结构。"""),

        code("""import sys
sys.path.insert(0, '..')
from src.knowledge_graph import KnowledgeGraphBuilder, cluster_documents
import os
os.makedirs('../data/processed', exist_ok=True)

# ========================================
# 1.1 手动构建知识图谱
# ========================================

kg = KnowledgeGraphBuilder(graph_type='directed')

# 添加实体节点
entities = [
    ("Google", "ORG"), ("DeepMind", "ORG"), ("AlphaGo", "PRODUCT"),
    ("Demis Hassabis", "PER"), ("London", "LOC"),
    ("Neural Network", "TECH"), ("Reinforcement Learning", "TECH"),
    ("OpenAI", "ORG"), ("GPT-4", "PRODUCT"), ("Sam Altman", "PER"),
]
kg.add_entities(entities)
print(f"添加了 {kg.graph.number_of_nodes()} 个实体节点")

# 添加关系边
relations = [
    ("Google", "owns", "DeepMind"),
    ("DeepMind", "created", "AlphaGo"),
    ("Demis Hassabis", "leads", "DeepMind"),
    ("DeepMind", "located_in", "London"),
    ("AlphaGo", "uses", "Neural Network"),
    ("AlphaGo", "uses", "Reinforcement Learning"),
    ("OpenAI", "created", "GPT-4"),
    ("Sam Altman", "leads", "OpenAI"),
    ("GPT-4", "uses", "Neural Network"),
]
for src, rel, tgt in relations:
    kg.add_relation(src, rel, tgt)
print(f"添加了 {kg.graph.number_of_edges()} 条关系")"""),

        code("""# ========================================
# 1.2 图谱统计信息
# ========================================

stats = kg.get_stats()
print(f"=== 图谱统计 ===")
print(f"节点数: {stats.node_count}")
print(f"边数: {stats.edge_count}")
print(f"\\n实体类型分布:")
for etype, count in stats.entity_types.items():
    print(f"  {etype}: {count}个")

print(f"\\n度最高的节点 (最'重要'的实体):")
for node_info in stats.top_nodes:
    print(f"  {node_info['node']:>25s}  度={node_info['degree']}  类型={node_info['type']}")"""),

        code("""# ========================================
# 1.3 路径查找 — 发现实体间的隐含关系
# ========================================

print("=== 路径查找 ===\\n")

# 查找 Google 到 Neural Network 的路径
paths = kg.find_paths("Google", "Neural Network")
print(f"Google → Neural Network 的路径:")
for path in paths:
    print(f"  {' → '.join(path)}")

# 查找 Google 到 GPT-4 的路径 (可能没有直接路径)
paths = kg.find_paths("Google", "GPT-4")
if paths:
    print(f"\\nGoogle → GPT-4 的路径:")
    for path in paths:
        print(f"  {' → '.join(path)}")
else:
    print(f"\\nGoogle → GPT-4: 无直接路径 (它们属于不同的子图)")"""),

        code("""# ========================================
# 1.4 知识图谱可视化
# ========================================

print("生成知识图谱可视化...")
path = kg.visualize('../data/processed/knowledge_graph.png')

if path:
    from PIL import Image
    import matplotlib.pyplot as plt
    
    img = Image.open(path)
    plt.figure(figsize=(14, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('AI领域知识图谱')
    plt.tight_layout()
    plt.show()
    print(f"图谱已保存: {path}")"""),

        code("""# ========================================
# 1.5 从文本自动构建共现关系
# ========================================

text = \"\"\"
Google acquired DeepMind in 2014. Demis Hassabis leads DeepMind in London.
DeepMind created AlphaGo which uses neural networks and reinforcement learning.
OpenAI, founded by Sam Altman, created GPT-4 using neural networks.
Google and OpenAI are competitors in the AI field.
\"\"\"

kg2 = KnowledgeGraphBuilder()
entities_auto = [
    ("Google", "ORG"), ("DeepMind", "ORG"), ("Demis Hassabis", "PER"),
    ("London", "LOC"), ("AlphaGo", "PRODUCT"), ("neural networks", "TECH"),
    ("reinforcement learning", "TECH"), ("OpenAI", "ORG"),
    ("Sam Altman", "PER"), ("GPT-4", "PRODUCT"),
]
kg2.add_entities(entities_auto)
kg2.add_relations_from_text(text, entities_auto)

stats2 = kg2.get_stats()
print(f"自动构建: {stats2.node_count}节点, {stats2.edge_count}条关系")
print(f"\\n自动发现的关系:")
for edge in kg2.graph.edges(data=True):
    print(f"  {edge[0]} --{edge[2].get('relation', '?')}--> {edge[1]}")"""),

        code("""# ========================================
# 1.6 主题聚类 (cluster_documents)
# ========================================

texts = [
    "机器学习是人工智能的核心分支，包括监督学习和无监督学习。",
    "深度学习使用神经网络进行特征学习和模式识别。",
    "自然语言处理让计算机能够理解和生成人类语言。",
    "计算机视觉使机器能够从图像和视频中提取信息。",
    "强化学习通过奖励信号让智能体学习最优策略。",
    "Python是数据科学和机器学习最常用的编程语言。",
    "Pandas和NumPy是Python数据分析的基础库。",
    "Matplotlib和Seaborn用于数据可视化和探索性分析。",
]

result = cluster_documents(texts, n_clusters=3)

print(f"=== 主题聚类结果 ({result['n_clusters']}个簇) ===\\n")
for cluster_id in range(result['n_clusters']):
    texts_in_cluster = [texts[i] for i, l in enumerate(result['labels']) if l == cluster_id]
    print(f"簇 {cluster_id}:")
    print(f"  关键词: {', '.join(result['top_words'][cluster_id][:5])}")
    for t in texts_in_cluster:
        print(f"  - {t[:50]}...")
    print()"""),

        md("""---
## 总结

### KnowledgeGraphBuilder 核心方法:
| 方法 | 功能 |
|------|------|
| `add_entities()` | 批量添加实体节点 |
| `add_relation()` | 添加关系边 |
| `add_relations_from_text()` | 从文本自动提取共现关系 |
| `build_from_nlp_result()` | 从NLP结果构建图谱 |
| `get_stats()` | 获取统计信息 |
| `find_paths()` | 查找实体间路径 |
| `get_subgraph()` | 获取子图 |
| `visualize()` | 可视化图谱 |
| `save()`/`load()` | 保存和加载 |

### 下一步: 运行 `06_streamlit_app.ipynb` 学习Web应用部署"""),
    ]
    make_notebook(cells, "05_knowledge_graph.ipynb")


def gen_notebook_06():
    cells = [
        md("""# Streamlit Web应用部署指南

> **对应模块**: `src/app.py` (Web界面)
> **前置条件**: `pip install streamlit python-dotenv`

---

## 什么是Streamlit？

Streamlit是一个Python Web框架，专为数据科学和ML项目设计。
- **零前端知识**: 不需要HTML/CSS/JavaScript
- **热重载**: 代码修改后自动刷新
- **丰富组件**: 上传文件、滑块、聊天框、图表，一行代码搞定

## 启动智析Web应用"""),

        code("""# 在终端中运行以下命令 (不是在Notebook中):
# cd zhixi
# streamlit run src/app.py

# 如果是在Notebook中测试，可以用以下方式:
print("=" * 60)
print("智析 Web应用启动指南")
print("=" * 60)

print(\"\"\"
步骤1: 安装依赖
  cd zhixi
  pip install -r requirements.txt

步骤2: 配置环境变量
  复制 .env.example 为 .env
  填入你的 OpenAI API Key (或配置Ollama)

步骤3: 启动应用
  streamlit run src/app.py

步骤4: 打开浏览器
  默认地址: http://localhost:8501

步骤5: 使用流程
  1. 在侧边栏选择LLM模式 (OpenAI/Ollama)
  2. 在"文档解析"标签页上传PDF
  3. 在"NLP分析"标签页执行分析
  4. 在"知识图谱"标签页构建图谱
  5. 在"智能问答"标签页提问
\"\"\")

print("应用截图说明:")
print("  - 顶部: 项目标题 '📊 智析'")
print("  - 侧边栏: LLM配置 + RAG参数")
print("  - 主区域: 4个标签页 (文档解析/NLP分析/知识图谱/智能问答)")"""),

        md("""## app.py 代码结构解析

### 核心函数:
| 函数 | 功能 |
|------|------|
| `init_session_state()` | 初始化Streamlit会话状态 |
| `render_sidebar()` | 渲染侧边栏 (LLM配置+RAG参数) |
| `render_header()` | 渲染页面标题 |
| `render_upload_section()` | 文档上传和解析 |
| `render_nlp_section()` | NLP分析 |
| `render_kg_section()` | 知识图谱构建 |
| `render_rag_section()` | RAG问答 |
| `_parse_document()` | 调用doc_parser解析PDF |
| `_init_rag()` | 初始化RAGEngine并导入文档 |
| `_ask_question()` | 调用RAGEngine回答问题 |

### 关键Streamlit组件:
```python
st.file_uploader()     # 文件上传
st.chat_input()        # 聊天输入框
st.chat_message()      # 聊天气泡
st.tabs()              # 标签页
st.spinner()           # 加载动画
st.metric()            # 指标卡片
st.expander()          # 可折叠区域
st.session_state       # 会话状态管理
```"""),

        code("""# ========================================
# 模拟Web应用的数据流
# ========================================

import sys, os
sys.path.insert(0, '..')
os.makedirs('../data/processed', exist_ok=True)

print("模拟Web应用的完整数据流:\\n")

# Step 1: 用户上传PDF → doc_parser
print("[Tab 1: 文档解析]")
from src.doc_parser import DocumentParser

test_pdf = '../data/sample_docs/test.pdf'
if os.path.exists(test_pdf):
    parser = DocumentParser(test_pdf, output_dir='../data/processed')
    result = parser.parse()
    chunks = parser.get_text_chunks(chunk_size=500, chunk_overlap=50)
    print(f"  页数: {result.total_pages}, 字符: {len(result.full_text)}, 块数: {len(chunks)}")
else:
    print("  请先运行 02_ocr_experiment.ipynb 创建测试PDF")
    chunks = []

# Step 2: NLP分析 → nlp_pipeline  
print("\\n[Tab 2: NLP分析]")
print("  (需要安装transformers/keybert，首次运行需下载模型)")
print("  功能: 关键词提取、实体识别、摘要生成、词云")

# Step 3: 知识图谱 → knowledge_graph
print("\\n[Tab 3: 知识图谱]")
from src.knowledge_graph import KnowledgeGraphBuilder
kg = KnowledgeGraphBuilder()
kg.add_entities([("智析", "PRODUCT"), ("RAG", "TECH"), ("ChromaDB", "TECH")])
kg.add_relation("智析", "uses", "RAG")
kg.add_relation("智析", "uses", "ChromaDB")
stats = kg.get_stats()
print(f"  节点: {stats.node_count}, 关系: {stats.edge_count}")

# Step 4: RAG问答 → rag_engine
print("\\n[Tab 4: RAG问答]")
print("  流程: 初始化引擎 → 导入文档块 → 用户提问 → 检索+生成")
if chunks:
    print(f"  已准备好 {len(chunks)} 个文档块，等待初始化RAG引擎...")

print("\\n" + "="*60)
print("Web应用数据流模拟完成!")
print("运行 'streamlit run src/app.py' 启动真实应用")"""),

        md("""---
## 部署到互联网 (可选)

### 方案1: Streamlit Community Cloud (免费)
1. 将项目推送到GitHub
2. 访问 share.streamlit.io
3. 连接GitHub仓库，选择 `src/app.py`
4. 自动生成公开链接

### 方案2: HuggingFace Spaces (免费)
1. 注册 huggingface.co 账号
2. 创建新Space，选择Streamlit框架
3. 上传项目文件
4. 自动生成公开链接

### 简历展示技巧
- 提供**在线Demo链接** (面试官不用安装任何东西)
- GitHub README中放**截图和GIF**
- 准备**2分钟演示视频**
- 准备**30秒电梯演讲**: 
  > "我做了一个多模态文档智能分析平台，能解析论文/研报的文字和图表，
  > 构建知识图谱，通过RAG增强的LLM对话回答深度问题。"

---

## 项目总结: 你学到了什么

### 技术栈全景:
```
Python基础 → NumPy/Pandas → Matplotlib → Scikit-learn
                                              ↓
                                    PyMuPDF/pdfplumber (CV)
                                              ↓
                                    HuggingFace/KeyBERT (NLP)
                                              ↓
                                    NetworkX (知识图谱)
                                              ↓
                                    LangChain/ChromaDB (RAG)
                                              ↓
                                    Streamlit (Web应用)
```

### 四大AI领域的融合:
- **CV**: PDF解析、OCR、图表识别
- **NLP**: 实体识别、关键词、摘要
- **数据挖掘**: 知识图谱、主题聚类
- **LLM应用**: RAG问答、Prompt工程

恭喜你完成了整个学习项目！"""),
    ]
    make_notebook(cells, "06_streamlit_app.ipynb")


# ========================================================================
# 运行所有生成
# ========================================================================
if __name__ == "__main__":
    print("生成智析项目学习Notebook...\n")
    gen_notebook_01()
    gen_notebook_02()
    gen_notebook_03()
    gen_notebook_04()
    gen_notebook_05()
    gen_notebook_06()
    print("\n全部生成完成!")
