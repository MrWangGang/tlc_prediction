# 🧠 深度学习结合纳什均衡构建的风险竞争模型:网约车收益最大化预测与调度策略
因数据集涉及到隐私,如需要请联系wanggang userbean@outlook.com
## 📝 1. 项目概述


本项目专注于城市交通领域的**微观行为分析**与**宏观趋势预测**。我们运用复杂的**深度学习**（时序预测）技术和严谨的**运筹学/博弈论**方法，致力于：

1.  **高精度预测**城市不同区域未来的出租车客流需求（**时空态势感知**）。
2.  通过模拟司机间的竞争，迭代求解市场激励下的**纳什均衡**（**策略博弈均衡**），从而发现最优的资源分配和司机调度策略。

这套研究工具提供了从数据资产化、需求感知到行为优化的完整架构，旨在提升城市交通运营的**效率**和司机的**经济效益**。

## 💡 2. 核心功能与技术栈

| 核心功能 | 描述 | 关键技术 | 脚本文件 |
| :--- | :--- | :--- | :--- |
| **数据资产化** | 批量处理原始 Parquet 文件，聚合区域客流，并生成完整、连续的多变量时间序列数据。 | Pandas, Time Series Grouper | `convert_gr.py` |
| **时空需求预测** | 训练基于 **LSTM 结合注意力机制 (Attention)** 的深度学习模型，预测未来各个交通区域的客流量。 | PyTorch, LSTM-Attention, Feature Engineering | `train_prediction.py` |
| **策略博弈均衡** | **纳什均衡的核心应用**：模拟司机竞争，通过迭代求解稳定策略，实现调度优化和资源均衡。 | NetworkX, 博弈论 (Game Theory), 收益函数 | `view_game.py` |

## ⚙️ 3. 模块详解

### 3.1. 数据资产化模块 (`convert_gr.py`)

该脚本负责数据预处理，将原始、离散的行程数据转化为模型可用的、连续的多变量时间序列数据。

#### 工作流程：
1.  **文件读取与过滤：** 查找并处理指定目录下的 Parquet 文件，确保数据质量和时间范围一致性。
2.  **时间序列聚合：** 以 `PULocationID` 为键，按照预设的频率 (`FREQUENCY`) 对 `tpep_pickup_datetime` 进行分组，计算总乘客数。
3.  **重塑与填充（关键）：**
    * 数据重塑为“宽”格式：时间戳为索引，每个区域 ID 作为一个独立的客流量特征列。
    * **时间序列完整性保证：** 重新索引并用 **0** 填充缺失的时间点，确保时间序列的连续性，这是时序模型有效工作的前提。

### 3.2. 时空需求预测模块 (`train_prediction.py`)

该脚本负责模型的训练、评估和报告生成。

#### 核心评估指标（Evaluation Metrics）：
模型的性能通过以下关键指标进行衡量（其中 $N$ 为数据点总数，$y_i$ 为真实值，$\hat{y}_i$ 为预测值，$\bar{y}$ 和 $\bar{\hat{y}}$ 分别为真实值和预测值的均值）：


1.  **平均绝对误差 (Mean Absolute Error, MAE)** - **主要优化目标**：
    $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

2.  **均方根误差 (Root Mean Square Error, RMSE)**：
    $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

3.  **决定系数 ($R^2$, Coefficient of Determination)**：
    $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

4.  **一致性相关系数 (Concordance Correlation Coefficient, CCC)**：
    $$\text{CCC} = \frac{2 \cdot \text{Cov}(y, \hat{y})}{\sigma_y^2 + \sigma_{\hat{y}}^2 + (\mu_y - \mu_{\hat{y}})^2}$$

### 3.3. 策略博弈均衡模块 (`view_game.py`)

该模块是项目的核心创新点，通过模拟司机的理性决策，求解资源在交通网络上的稳定分配策略。

#### 司机收益函数 (Payoff Function)：
司机的策略选择基于最大化其预期收益。收益 $P(s, a)$ 是载客奖励、移动成本和空驶惩罚的加权组合，定义如下：

$$P(s, a) = (\text{TripDist}_{s,a} \cdot R) - (\alpha \cdot \text{MoveCost}_{s, a}) - \mathbb{I}(\text{NoDemand}_{a}) \cdot \lambda$$


其中各项参数定义如下：

| 参数 | 符号 | 类型 | 描述 | 对应代码参数/计算来源 |
| :--- | :--- | :--- | :--- | :--- |
| **当前状态** | $s$ | **动态变量** | 司机在博弈中的当前停靠区域。 | 模拟迭代中的实时位置 |
| **选择动作** | $a$ | **动态变量** | 司机选择前往的区域（决策）。 | 模拟迭代中的实时决策 |
| **奖励系数** | $R$ | **超参数** | 单位行程距离奖励。 | `PASSENGER_REWARD` |
| **成本系数** | $\alpha$ | **超参数** | 移动成本系数，调节司机对巡游距离的敏感度。 | `ALPHA` |
| **空驶惩罚** | $\lambda$ | **超参数** | 空驶或无需求时的惩罚系数。 | `NO_PASSENGER_PENALTY` |
| **载客收益** | $\text{TripDist}_{s,a}$ | **计算结果** | 假设载客成功，乘客的行程距离。 | 由 NetworkX 图结构决定 |
| **移动成本** | $\text{MoveCost}_{s, a}$ | **计算结果** | 从区域 $s$ 到 $a$ 的最短路径距离（成本）。 | 由 NetworkX 图结构决定 |
| **需求指示** | $\mathbb{I}(\text{NoDemand}_{a})$ | **计算结果** | 指示函数，当区域 $a$ 没有可载乘客时为 1，否则为 0。 | 实时需求和竞争状态 |

#### 纳什均衡求解：
采用迭代机制，司机总是选择能带来最高**预期收益**的区域。**纳什均衡**代表了一个稳定的策略分布：在该分布下，**没有任何司机有动机单方面改变其当前策略**（即停靠位置）。

### 3.4. 模拟市场场景配置

在 `view_game.py` 中，我们通过组合两组参数来模拟四种不同的市场环境，以测试调度策略在不同资源竞争和激励机制下的鲁棒性。

#### 1. 资源配置 (GROUP_PARAMS)
| 参数组 | `NUM_TAXIS` (出租车数量) | `NUM_PASSENGERS` (乘客数量) | 描述 |
| :--- | :--- | :--- | :--- |
| **LowTaxis** | 30 | 100 | 模拟竞争较低、资源相对稀缺的市场。 |
| **MidTaxis** | 55 | 100 | 模拟竞争中等、供需相对均衡的市场。 |
| **HighTaxis** | 80 | 100 | 模拟竞争激烈、出租车供给充足的市场。 |

#### 2. 激励环境 (MARKET_PARAMS)
| 参数组 | $\alpha$ (移动成本系数) | $R$ (奖励系数) | $\lambda$ (空驶惩罚) | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| **LowCost\_HighReward** | 0.1 | 20.0 | 5 | 低成本、高奖励，鼓励司机积极巡游并接远途客。 |
| **HighCost\_LowReward** | 0.8 | 5.0 | 10 | 高成本、低奖励，激励司机谨慎选择策略、降低空驶。 |

#### 3. 最终模拟场景
三种资源配置（Low/Mid/High）与两种激励环境（Low/High）将组合形成 **六种** 场景，独立运行，并生成对应的博弈均衡策略和收益对比图。

#### 最终的收益效果图
<img width="1000" height="600" alt="rmse_curve" src="https://github.com/user-attachments/assets/bc3a6912-f7a3-47ff-8a21-a348dfe99471" />
<img width="1000" height="600" alt="r2_curve" src="https://github.com/user-attachments/assets/136024c4-abf2-4c24-b445-7bc3a2ac8fff" />
<img width="1500" height="800" alt="prediction_plot" src="https://github.com/user-attachments/assets/3f6b1871-a5d6-462b-9b9f-99e245d5690b" />

<img width="800" height="600" alt="两种策略收益对比图" src="https://github.com/user-attachments/assets/96ed241a-6791-4367-9b27-deae372287a5" />

<img width="2000" height="1200" alt="纳什均衡出租车分布图" src="https://github.com/user-attachments/assets/0318ddc8-34e6-4f0c-8552-302aaf296f0b" />


