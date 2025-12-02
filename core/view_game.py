import matplotlib
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib.font_manager as fm

# --- 修复中文乱码的配置 ---
matplotlib.use("agg")
try:
    font_path = fm.findfont(fm.FontProperties(family='SimHei', style='normal', weight='normal'))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
except:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# --- 全局参数设定 ---
ROWS, COLS = 14, 19
TOTAL_REGIONS = 262

# --- 1. 图的构建 (Graph Construction) ---
def build_graph():
    G_coords = nx.grid_2d_graph(ROWS, COLS)
    nodes_to_remove = [(i, j) for i in range(ROWS) for j in range(COLS)
                       if (i * COLS + j + 1) > TOTAL_REGIONS]
    G_coords.remove_nodes_from(nodes_to_remove)
    node_labels = {node: node[0] * COLS + node[1] + 1 for node in G_coords.nodes()}
    G = nx.relabel_nodes(G_coords, node_labels)
    return G

# --- 2. 初始化乘客 ---
def get_passenger_predictions_with_dest(num_regions, num_passengers):
    passengers = {r: 0 for r in range(1, num_regions + 1)}
    destinations = {r: [] for r in range(1, num_regions + 1)}
    all_region_ids = list(range(1, num_regions + 1))
    passenger_assignment = np.random.choice(all_region_ids, num_passengers, replace=True)
    for r in passenger_assignment:
        passengers[r] += 1
    for r in all_region_ids:
        dests = []
        for _ in range(passengers[r]):
            while True:
                dest = np.random.choice(all_region_ids)
                if dest != r:
                    dests.append(dest)
                    break
        destinations[r] = dests
    return passengers, destinations

# --- 3. 收益计算 ---
def compute_payoff(G, current_pos, target_region, passenger_predictions, destinations, alpha, passenger_reward, no_passenger_penalty):
    try:
        move_cost = nx.shortest_path_length(G, source=current_pos, target=target_region)
    except nx.NetworkXNoPath:
        return -np.inf
    demand = passenger_predictions.get(target_region, 0)
    if demand <= 0:
        return -alpha * move_cost - no_passenger_penalty
    if destinations.get(target_region) and len(destinations[target_region]) > 0:
        dest = destinations[target_region][0]
        try:
            trip_dist = nx.shortest_path_length(G, source=target_region, target=dest)
        except nx.NetworkXNoPath:
            trip_dist = 0
    else:
        trip_dist = 0
    payoff = trip_dist * passenger_reward - alpha * move_cost
    return payoff

# --- 4. 纳什均衡迭代 (采用竞争规则和自动停止) ---
def find_and_save_nash_equilibrium(G, num_taxis, num_passengers, params, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    all_region_ids = list(G.nodes())
    taxi_positions = np.random.choice(all_region_ids, num_taxis).tolist()
    passenger_predictions, destinations = get_passenger_predictions_with_dest(G.number_of_nodes(), num_passengers)
    current_positions = taxi_positions.copy()
    moved_history = []

    for it in range(1, 50 + 1):
        intended_moves = []
        for taxi_idx in range(num_taxis):
            current_pos = current_positions[taxi_idx]
            best_payoff = -np.inf
            best_region = current_pos
            for region in all_region_ids:
                payoff = compute_payoff(G, current_pos, region, passenger_predictions, destinations,
                                        alpha=params['ALPHA'],
                                        passenger_reward=params['PASSENGER_REWARD'],
                                        no_passenger_penalty=params['NO_PASSENGER_PENALTY'])
                if payoff > best_payoff:
                    best_payoff, best_region = payoff, region
            intended_moves.append({'taxi_idx': taxi_idx, 'current_pos': current_pos, 'intended_pos': best_region, 'payoff': best_payoff})
        intended_moves.sort(key=lambda x: x['payoff'], reverse=True)
        new_positions = current_positions.copy()
        moved = 0
        rem_pass_round = {k: v for k, v in passenger_predictions.items()}
        for move in intended_moves:
            taxi_idx = move['taxi_idx']
            current_pos = move['current_pos']
            intended_pos = move['intended_pos']
            if rem_pass_round.get(intended_pos, 0) > 0 and move['payoff'] > 0:
                new_positions[taxi_idx] = intended_pos
                rem_pass_round[intended_pos] -= 1
                moved += 1
            else:
                new_positions[taxi_idx] = current_pos
        current_positions = new_positions
        avg_payoff = np.mean([m['payoff'] for m in intended_moves])
        print(f"Iteration {it}: avg payoff = {avg_payoff:.2f}, moved = {moved}")
        moved_history.append(moved)
        if moved == 0:
            print("达到完美均衡，停止。")
            break
        if len(moved_history) >= 3 and moved_history[-1] == moved_history[-2] and moved_history[-2] == moved_history[-3]:
            print("达到动态稳定，停止。")
            break
    with open(os.path.join(base_dir, "nash_equilibrium.json"), "w") as f:
        json.dump({"positions": current_positions}, f)
    return current_positions, taxi_positions, passenger_predictions, destinations

# --- 5. 策略收益 ---
def evaluate_and_save_strategies(G, current_taxi_positions, passenger_predictions, destinations, params, base_dir):
    rec_payoffs, rand_payoffs = [], []
    all_nodes_ids = list(G.nodes())

    rem_pass_rec = {k: v for k, v in passenger_predictions.items()}
    rem_dest_rec = {k: list(v) for k, v in destinations.items()}
    for i, pos in enumerate(current_taxi_positions):
        best_payoff = -np.inf
        best_region = pos
        for r in all_nodes_ids:
            payoff = compute_payoff(G, pos, r, rem_pass_rec, rem_dest_rec,
                                    alpha=params['ALPHA'],
                                    passenger_reward=params['PASSENGER_REWARD'],
                                    no_passenger_penalty=params['NO_PASSENGER_PENALTY'])
            if payoff > best_payoff:
                best_payoff, best_region = payoff, r
        if rem_pass_rec.get(best_region, 0) > 0:
            dest = rem_dest_rec[best_region][0]
            try:
                trip_dist = nx.shortest_path_length(G, source=best_region, target=dest)
                move_cost = nx.shortest_path_length(G, source=pos, target=best_region)
            except nx.NetworkXNoPath:
                trip_dist = 0
                move_cost = 0
            payoff = trip_dist * params['PASSENGER_REWARD'] - params['ALPHA'] * move_cost
            rem_pass_rec[best_region] -= 1
            rem_dest_rec[best_region].pop(0)
        else:
            payoff = -1
        rec_payoffs.append(payoff)

    rem_pass_rand = {k: v for k, v in passenger_predictions.items()}
    rem_dest_rand = {k: list(v) for k, v in destinations.items()}
    for i, pos in enumerate(current_taxi_positions):
        rand_region = np.random.choice(all_nodes_ids)
        if rem_pass_rand.get(rand_region, 0) > 0:
            dest = rem_dest_rand[rand_region][0]
            try:
                trip_dist = nx.shortest_path_length(G, source=rand_region, target=dest)
                move_cost = nx.shortest_path_length(G, source=pos, target=rand_region)
            except nx.NetworkXNoPath:
                trip_dist = 0
                move_cost = 0
            payoff = trip_dist * params['PASSENGER_REWARD'] - params['ALPHA'] * move_cost
            rem_pass_rand[rand_region] -= 1
            rem_dest_rand[rand_region].pop(0)
        else:
            payoff = -1
        rand_payoffs.append(payoff)

    plt.figure(figsize=(8, 6))
    plt.boxplot([rec_payoffs, rand_payoffs], tick_labels=["Recommended", "Random"])
    plt.ylabel("Payoff")
    plt.title("Comparison of Strategies")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(base_dir, "两种策略收益对比图.png"))
    plt.close()

# --- 6. 保存网格图 ---
def save_graph_visualization(G, data_dict, title, file_path):
    plt.figure(figsize=(20, 12))
    pos = {node_id: ((node_id - 1) % 19, -((node_id - 1) // 19)) for node_id in G.nodes()}
    node_colors = ['yellow' if data_dict.get(node_id, 0) > 0 else 'white' for node_id in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)
    labels = {node_id: data_dict.get(node_id, 0) for node_id in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    plt.title(title, size=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

# --- 7. 主程序 ---
def main():
    G = build_graph()

    # 定义5组不同的超参数
    group_params = {
        "Group 1 (均衡参数)": {"ALPHA": 1.0, "PASSENGER_REWARD": 3, "NO_PASSENGER_PENALTY": 5},
        "Group 2 (高成本参数)": {"ALPHA": 2.0, "PASSENGER_REWARD": 3, "NO_PASSENGER_PENALTY": 5},
        "Group 3 (司机友好参数)": {"ALPHA": 0.5, "PASSENGER_REWARD": 10, "NO_PASSENGER_PENALTY": 5},
        "Group 4 (低回报参数)": {"ALPHA": 1.0, "PASSENGER_REWARD": 2, "NO_PASSENGER_PENALTY": 5},
        "Group 5 (极高惩罚参数)": {"ALPHA": 1.0, "PASSENGER_REWARD": 3, "NO_PASSENGER_PENALTY": 10},
    }

    # 每组有3个市场，通过改变供需关系来模拟
    market_types = {
        "供给过剩": {"NUM_PASSENGERS": 200, "NUM_TAXIS": 600},
        "供需平衡": {"NUM_PASSENGERS": 200, "NUM_TAXIS": 300},
        "需求过剩": {"NUM_PASSENGERS": 300, "NUM_TAXIS": 200},
    }

    base_output_dir = "./view"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    for group_name, group_params in group_params.items():
        for market_name, market_params in market_types.items():
            scenario_name = f"{group_name} - {market_name}"
            print(f"\n--- 模拟场景: {scenario_name} ---")

            # 合并参数
            params = {**group_params, **market_params}

            print(f"参数: 出租车={params['NUM_TAXIS']}, 乘客={params['NUM_PASSENGERS']}, ALPHA={params['ALPHA']}, REWARD={params['PASSENGER_REWARD']}, PENALTY={params['NO_PASSENGER_PENALTY']}")

            scenario_dir = os.path.join(base_output_dir, scenario_name)

            nash_positions, initial_taxi_positions, passenger_predictions, destinations = find_and_save_nash_equilibrium(G, params['NUM_TAXIS'], params['NUM_PASSENGERS'], params, base_dir=scenario_dir)

            initial_taxi_counts = {r: initial_taxi_positions.count(r) for r in range(1, TOTAL_REGIONS + 1)}
            save_graph_visualization(G, initial_taxi_counts, f"Initial Taxi Distribution ({scenario_name})", os.path.join(scenario_dir, "初始出租车分布图.png"))

            save_graph_visualization(G, passenger_predictions, f"Initial Passenger Distribution ({scenario_name})", os.path.join(scenario_dir, "初始乘客分布图.png"))

            taxi_counts = {r: nash_positions.count(r) for r in range(1, TOTAL_REGIONS + 1)}
            save_graph_visualization(G, taxi_counts, f"Nash Equilibrium Taxi Distribution ({scenario_name})", os.path.join(scenario_dir, "纳什均衡出租车分布图.png"))

            evaluate_and_save_strategies(G, nash_positions, passenger_predictions, destinations, params, base_dir=scenario_dir)

    print("\n所有模拟已完成。")

if __name__ == "__main__":
    main()