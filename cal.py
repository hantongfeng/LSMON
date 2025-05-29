from collections import deque
import numpy as np
import re
# 定义文件名
file_name = 'results_arryt.txt'

# 使用deque读取文件的最后五行
last_five_lines = deque(maxlen=3)
with open(file_name, 'r') as file: 
    for line in file:
        last_five_lines.append(line.strip())

# 存储提取的百分比值
percentages = []

# 从最后五行中提取百分比值
for line in last_five_lines:
    # 使用正则表达式匹配百分比值
    match = re.search(r'(\d+\.\d+)%', line)
    if match:
        # 将百分比值转换为浮点数并添加到列表中
        percentage = float(match.group(1))
        percentages.append(percentage)

# 计算平均值
if percentages:
    mean = np.mean(percentages)
    std=np.std(percentages)
    print(f'以上十行的平均值: {mean}')
    print(f'以上十行的标准差: {std}')
else:
    print("没有有效的数据进行计算")


with open(file_name, 'a') as f:
            f.write("平均值为"+str(mean) + '\n')
            f.write("标准差为"+str(std) + '\n')