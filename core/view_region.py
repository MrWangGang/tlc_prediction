import networkx as nx
import matplotlib.pyplot as plt

# --- 1. 图的构建 (Graph Construction) ---
# 将262个区域想象成一个14x19的矩阵
rows, cols = 14, 19
total_regions = 262

# 使用 networkx 的 grid_2d_graph() 函数自动创建网格图
# 每个节点都是一个坐标元组 (row, col)
G = nx.grid_2d_graph(rows, cols)

# 因为矩阵有266个格子，而我们只需要262个区域
# 找出需要移除的多余节点
nodes_to_remove = [(i, j) for i in range(rows) for j in range(cols) if (i * cols + j + 1) > total_regions]

# 移除这些多余的节点
G.remove_nodes_from(nodes_to_remove)

# 为每个节点添加 'region_id' 属性，方便后续操作
node_labels = {node: node[0] * cols + node[1] + 1 for node in G.nodes()}
nx.set_node_attributes(G, node_labels, 'region_id')

# 打印图的基本信息
print(f"图中的节点数量: {G.number_of_nodes()}")
print(f"图中的边数量: {G.number_of_edges()}")
print("--------------------")

# --- 2. 图的可视化 (Graph Visualization) ---

plt.figure(figsize=(20, 12)) # 增大图的大小以容纳更多标签

# 定义节点位置 (布局)
# 使用节点的坐标作为布局，使图呈现网格状
pos = {(r, c): (c, -r) for r, c in G.nodes()}

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100, alpha=0.8)

# 绘制边
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)

# 绘制所有节点的标签，并减小字体大小
labels = {node: G.nodes[node]['region_id'] for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=5, font_color='black')

# 将标题修改为纯英文
plt.title(f"Grid-like Graph of {total_regions} Regions", size=15)
plt.axis('off') # 不显示坐标轴
plt.tight_layout() # 自动调整布局，防止标签重叠
plt.show() # 显示绘制好的图形