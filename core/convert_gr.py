import pandas as pd
import os
import glob
from tqdm import tqdm

# 定义文件路径和数据处理参数
input_dir = './datasets/source/'
output_file = './datasets/tripdata.csv'
# --- 新增参数：你可以更改此值来控制时间粒度 ---
FREQUENCY = '60min' # 可选值: '15min', '60min', 'H' 等

try:
    # 1. 查找所有源 Parquet 文件
    parquet_files = glob.glob(os.path.join(input_dir, '*.parquet'))

    if not parquet_files:
        print(f"错误：在目录 '{input_dir}' 中找不到任何 .parquet 文件。")
        exit()

    print(f"找到 {len(parquet_files)} 个文件准备处理。")

    # 2. 读取、过滤并合并所有 Parquet 文件
    df_list = []
    total_rows_read = 0
    total_rows_filtered_out = 0

    # 使用 tqdm 包装文件列表，自动显示进度条
    for f in tqdm(parquet_files, desc="正在处理文件"):
        # 从文件名中提取年和月，格式为 YYYY-MM
        filename = os.path.basename(f)
        try:
            date_str = filename.split('_')[2].split('.')[0]
            # 转换为日期字符串，方便与数据进行比较
            expected_date = pd.to_datetime(date_str, format='%Y-%m')
        except (IndexError, ValueError):
            print(f"警告：无法从文件名 '{filename}' 中提取日期，跳过日期过滤。")
            df = pd.read_parquet(f)
            total_rows_read += len(df)
            df_list.append(df)
            continue

        df = pd.read_parquet(f)

        # 记录初始行数
        initial_rows = len(df)
        total_rows_read += initial_rows

        # 过滤数据，只保留日期与文件名匹配的行
        filtered_df = df[df['tpep_pickup_datetime'].dt.to_period('M') == expected_date.to_period('M')]

        # 记录被过滤掉的行数
        filtered_out_rows = initial_rows - len(filtered_df)
        total_rows_filtered_out += filtered_out_rows

        df_list.append(filtered_df)

    combined_df = pd.concat(df_list, ignore_index=True)

    if combined_df.empty:
        print("错误：合并后的数据为空。请检查源文件内容。")
        exit()

    # 输出统计信息
    print("\n--- 处理统计 ---")
    print(f"共读取 {total_rows_read} 行原始数据。")
    print(f"共过滤掉 {total_rows_filtered_out} 行不匹配日期的数据。")
    print(f"剩余 {len(combined_df)} 行数据用于聚合。")
    print("----------------")

    # 3. 按上车地点ID和指定频率聚合
    minutely_passenger_counts = combined_df.groupby(
        ['PULocationID', pd.Grouper(key='tpep_pickup_datetime', freq=FREQUENCY)]
    )['passenger_count'].sum().reset_index()

    # 4. 使用 pivot_table 将数据重塑为“宽”格式
    wide_df = minutely_passenger_counts.pivot_table(
        index='tpep_pickup_datetime',
        columns='PULocationID',
        values='passenger_count',
        fill_value=0
    ).reset_index()

    # --- 新增步骤：基于 FREQUENCY 填充完整的时间序列 ---
    # 获取所有不重复的日期
    all_dates = pd.to_datetime(wide_df['tpep_pickup_datetime'].dt.date.unique())

    # 为每个日期生成一个完整的24小时时间序列
    all_time_indices = []
    for date in all_dates:
        start_of_day = date.replace(hour=0, minute=0, second=0)
        end_of_day = date.replace(hour=23, minute=59, second=59)
        daily_range = pd.date_range(start=start_of_day, end=end_of_day, freq=FREQUENCY)
        all_time_indices.append(daily_range)

    # 合并所有日期的完整时间序列
    full_time_index = pd.Index([])
    if all_time_indices:
        full_time_index = pd.concat([pd.Series(s) for s in all_time_indices])

    # 将 wide_df 的索引设置为时间，以便于重采样
    wide_df.set_index('tpep_pickup_datetime', inplace=True)

    # 重新索引，并用 0 填充缺失的时间点
    completed_df = wide_df.reindex(full_time_index, fill_value=0).reset_index()

    # 5. 重命名时间列，使其更清晰
    completed_df.rename(columns={'index': '时间'}, inplace=True)

    # 6. 将处理后的数据保存到新的 CSV 文件中，不含表头
    completed_df.to_csv(output_file, header=False, index=False)

    print(f"\n数据已成功处理，并保存到文件: {output_file}")
    print("该文件已填充所有小时，不包含表头。")

except FileNotFoundError:
    print(f"错误：找不到目录 '{input_dir}'。请确保文件路径正确。")
except KeyError as e:
    print(f"错误：找不到列 {e}。请检查你的文件表头。")
except Exception as e:
    print(f"发生错误: {e}")