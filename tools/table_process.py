import os
import json
import pandas as pd
from pathlib import Path
from io import StringIO
import tempfile
import warnings
import re

# 导入 dots.ocr 的解析器
try:
    from dots_ocr.parser import DotsOCRParser
except ImportError:
    raise ImportError("请确保已安装 dots_ocr 包，或者将此脚本放在 dots.ocr 项目根目录下运行。")

# 忽略 pandas 的 FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def clean_table_footer(df):
    """
    自动检测并去除表格底部的注释/脚注行。
    逻辑：
    1. 检查最后一行。
    2. 如果最后一行大部分是空的 (NaN)，或者包含特定关键词 (Note, Source)，则删除。
    """
    if df.empty or len(df) < 2:
        return df

    # 获取最后一行
    last_row = df.iloc[-1]
    
    # 计算非空单元格的数量
    non_empty_count = last_row.count()
    total_cols = len(df.columns)
    
    # 获取最后一行第一个非空值的文本（如果有）
    last_row_text = ""
    for val in last_row.dropna():
        last_row_text = str(val)
        break # 只需要看第一个有效内容通常就够了

    # --- 判定规则 ---
    should_drop = False

    # 规则 1: 关键词检测 (不区分大小写)
    keywords = ["note:", "source:", "注：", "数据来源：", "说明：", "remarks:"]
    if any(k in last_row_text.lower() for k in keywords):
        should_drop = True

    # 规则 2: 结构检测 (如果一行只有 1 个格子有字，且字数较多，通常是注释)
    # 比如表格有 5 列，最后一行只有 1 列有值，且长度 > 10，大概率是注释而不是数据
    elif non_empty_count == 1 and total_cols > 1 and len(last_row_text) > 10:
        should_drop = True
    
    # 规则 3: 极度稀疏 (非空占比 < 20%)
    elif non_empty_count / total_cols < 0.2:
        should_drop = True

    if should_drop:
        print(f"        -> 自动移除疑似注释行: '{last_row_text[:20]}...'")
        return df.iloc[:-1] # 返回去掉最后一行的 df
    
    return df

def batch_process_pdfs(
    input_folder: str,
    output_folder: str,
    server_ip: str = "127.0.0.1",
    server_port: int = 8000,
    prompt_mode: str = "prompt_layout_all_en"
):
    """
    批量处理 PDF：提取表格，合并同名表头表格，去除底部注释
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 初始化 Parser
    print(f"正在初始化 DotsOCRParser (连接至 {server_ip}:{server_port})...")
    parser = DotsOCRParser(
        ip=server_ip,
        port=server_port,
        dpi=200,
        min_pixels=256*28*28,
        max_pixels=11289600
    )

    pdf_files = list(input_path.rglob("*.pdf"))
    if not pdf_files:
        print(f"在 {input_folder} 中未找到 PDF 文件。")
        return

    print(f"共找到 {len(pdf_files)} 个 PDF 文件，准备处理...")

    for idx, pdf_file in enumerate(pdf_files):
        print(f"\n[{idx+1}/{len(pdf_files)}] 正在处理: {pdf_file.name}")
        
        # 用于收集该文档的所有表格 DataFrame
        # 格式: [{'df': DataFrame, 'page': int}, ...]
        all_tables = []

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 解析整个 PDF
                results = parser.parse_pdf(
                    input_path=str(pdf_file),
                    filename=pdf_file.stem,
                    prompt_mode=prompt_mode,
                    save_dir=temp_dir
                )

                if not results:
                    print("    ! 解析未返回结果")
                    continue

                # 提取所有页面的表格
                for page_res in results:
                    page_num = page_res.get('page_no', 0)
                    json_path = page_res.get('layout_info_path')
                    
                    if not json_path or not os.path.exists(json_path):
                        continue

                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if not isinstance(data, list): continue

                        for item in data:
                            if not isinstance(item, dict): continue
                            
                            # 识别表格
                            if item.get('category') == 'Table' or item.get('type') == 'Table':
                                html_content = item.get('text', '')
                                if not html_content: continue

                                dfs = pd.read_html(StringIO(html_content))
                                if dfs:
                                    raw_df = dfs[0]
                                    # 先进行一次基础清洗
                                    clean_df = clean_table_footer(raw_df)
                                    if not clean_df.empty:
                                        all_tables.append({
                                            'df': clean_df,
                                            'page': page_num
                                        })
                    except Exception as e:
                        print(f"    ! 读取页面 {page_num} JSON 失败: {e}")

                # --- 合并逻辑 ---
                if not all_tables:
                    print("    - 未检测到有效表格")
                    continue

                merged_tables = [] # 存储最终合并后的表格列表
                
                if all_tables:
                    # 初始化第一个表格
                    current_merged = all_tables[0]['df']
                    current_page_start = all_tables[0]['page']
                    
                    # 从第二个表格开始遍历
                    for i in range(1, len(all_tables)):
                        next_table = all_tables[i]
                        next_df = next_table['df']
                        
                        # 核心判定：如果列名完全一致，则视为同一表格的延续
                        # 注意：这里假设 pandas 读取 html 产生的列名在跨页时是一致的
                        if list(current_merged.columns) == list(next_df.columns):
                            print(f"    - [合并] 检测到跨页/连续表格 (页 {next_table['page']})，正在合并...")
                            current_merged = pd.concat([current_merged, next_df], ignore_index=True)
                        else:
                            # 列名不同，保存当前合并好的表格，开始新表格
                            merged_tables.append(current_merged)
                            current_merged = next_df
                            current_page_start = next_table['page']
                    
                    # 添加最后一个表格
                    merged_tables.append(current_merged)

                # --- 保存 CSV ---
                for t_idx, table_df in enumerate(merged_tables):
                    # 再次清洗合并后的表格底部（防止合并后中间的注释变成了底部，或者底部的注释还没去干净）
                    final_df = clean_table_footer(table_df)
                    
                    csv_filename = f"{pdf_file.stem}_table_{t_idx + 1}.csv"
                    csv_path = output_path / csv_filename
                    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print(f"    ✓ 已保存: {csv_filename} (行数: {len(final_df)})")

            except Exception as e:
                print(f"    ! 处理文件异常: {e}")

    print(f"\n全部处理完成！结果已保存至: {output_folder}")

if __name__ == "__main__":
    # 默认配置
    INPUT_DIR = "./documents"
    OUTPUT_DIR = "./output_tables"
    PORT = 8000
    MODE = "prompt_layout_all_en"

    import argparse
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument("--input", type=str, default=INPUT_DIR)
    parser_args.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser_args.add_argument("--port", type=int, default=PORT)
    parser_args.add_argument("--mode", type=str, default=MODE)
    
    args = parser_args.parse_args()

    batch_process_pdfs(
        input_folder=args.input,
        output_folder=args.output,
        server_port=args.port,
        prompt_mode=args.mode
    )
