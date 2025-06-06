要将一组包含正负值的数值数据映射到 [0,1] 区间，同时保持原始数据的相对大小关系（单调性）不变，可以使用**线性变换**方法。以下是完整实现步骤和公式：

---

### **核心公式：线性归一化**
\[
x_{\text{norm}} = \frac{x - \min(X)}{\max(X) - \min(X)}
\]

其中：
- \( x \)：原始数据中的任意值
- \( \min(X) \)：原始数据集中的最小值
- \( \max(X) \)：原始数据集中的最大值

---

### **分步实现**
#### **1. 计算极值**
```python
data = [-5, -2, 0, 3, 8]  # 示例数据（含负值）

min_val = min(data)  # min_val = -5
max_val = max(data)  # max_val = 8
range_val = max_val - min_val  # range_val = 13
```

#### **2. 应用归一化公式**
```python
normalized_data = [(x - min_val) / range_val for x in data]
```

#### **3. 验证结果**
| 原始数据 | 归一化结果 | 验证关系 |
|---------|-----------|---------|
| -5      | 0.0       | 最小 → 0 |
| -2      | 0.2308    | -2 > -5 → 0.2308 > 0.0 |
| 0       | 0.3846    | 0 > -2 → 0.3846 > 0.2308 |
| 3       | 0.6154    | 3 > 0 → 0.6154 > 0.3846 |
| 8       | 1.0       | 最大 → 1 |

---

### **边界情况处理**
#### **情况1：所有值相同**
若数据全等（如 `[7, 7, 7]`），则统一映射到 **0.5**：
```python
if min_val == max_val:
    normalized_data = [0.5] * len(data)
```

#### **情况2：空数据集**
直接返回空列表：
```python
if len(data) == 0:
    normalized_data = []
```

---

### **完整Python函数**
```python
def normalize_to_01(data):
    if len(data) == 0:
        return []
    
    min_val = min(data)
    max_val = max(data)
    
    if min_val == max_val:
        return [0.5] * len(data)
    
    range_val = max_val - min_val
    return [(x - min_val) / range_val for x in data]

# 测试
data = [-5, -2, 0, 3, 8]
print(normalize_to_01(data))  # 输出: [0.0, 0.2307..., 0.3846..., 0.6153..., 1.0]
```

---

### **数学原理**
1. **平移变换**：  
   \( x - \min(X) \) 将数据平移到非负区间（最小值变为0）。
   
2. **缩放变换**：  
   \( \div (\max(X) - \min(X)) \) 将数据压缩到 [0,1] 区间。

3. **单调性保持**：  
   由于变换函数是**严格单调递增**的线性函数，原始数据的大小关系完全保留。

---

### **其他方法对比**
| **方法**         | 是否保持大小关系 | 是否支持负值 | 映射区间   | 适用场景               |
|------------------|----------------|------------|-----------|-----------------------|
| **线性归一化**   | ✔️             | ✔️          | [0,1]     | 通用场景（推荐）       |
| Sigmoid函数      | ✔️             | ✔️          | (0,1)     | 需要压缩极端值         |
| 小数定标标准化   | ✔️             | ✔️          | [-1,1]    | 需保留负号信息         |
| Z-Score标准化    | ✘              | ✔️          | (-∞,∞)    | 需符合正态分布         |

---

### **进阶场景处理**
#### **需求1：保留负号信息**
若需区分正负（如将负值映射到 [0,0.5)，正值到 [0.5,1]），使用：
```python
scaled_data = [0.5 + 0.5 * (x / max(abs(min_val), abs(max_val))) for x in data]
```

#### **需求2：处理极端异常值**
若存在离群点（如 `[-5, -2, 0, 3, 100]`），使用**分位数裁剪**：
```python
from numpy import percentile
q1 = percentile(data, 5)  # 5%分位数
q2 = percentile(data, 95) # 95%分位数
clipped_data = [max(min(x, q2), q1) for x in data]  # 裁剪到[5%,95%]
normalized_data = normalize_to_01(clipped_data)
```

---

### **总结**
- **标准方法**：使用线性归一化公式 \( x_{\text{norm}} = \frac{x - \min}{\max - \min} \) 可完美满足需求。
- **关键优势**：严格保持数据相对大小，支持正负值，计算结果在 [0,1] 区间。
- **注意事项**：处理全等数据时返回 0.5，避免除零错误。