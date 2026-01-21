import os
import glob
import argparse
import pandas as pd
from io import StringIO
import json
import warnings

# Import dots_ocr components
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import image_extensions

class TableExtractor:
    def __init__(self,
                 vllm_ip='localhost',
                 vllm_port=8000,
                 vllm_model='rednote-hilab/dots.ocr',
                 num_thread=16,
                 output_dir='./output',
                 dpi=200):
        self.output_dir = output_dir
        self.parser = DotsOCRParser(
            ip=vllm_ip,
            port=vllm_port,
            model_name=vllm_model,
            num_thread=num_thread,
            dpi=dpi,
            output_dir=output_dir,
            protocol='http', # Assuming http as per user instruction "already deployed"
        )

    def process_folder(self, folder_path):
        """
        Process all PDFs in the folder, extract tables, and save them as CSVs.
        """
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist.")
            return

        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {folder_path}.")

        for pdf_file in pdf_files:
            self.process_pdf(pdf_file)

    def process_pdf(self, pdf_path):
        """
        Process a single PDF file: extract tables, merge them, and save to CSV.
        """
        print(f"Processing {pdf_path}...")

        # 1. Parse PDF using dots_ocr
        # Use 'prompt_layout_all_en' to get both layout and text content
        try:
            results = self.parser.parse_file(
                pdf_path,
                prompt_mode="prompt_layout_all_en",
                output_dir=self.output_dir
            )
        except Exception as e:
            print(f"Error parsing PDF {pdf_path}: {e}")
            return

        # 2. Extract and merge tables
        all_tables = []

        # 'results' contains a list of dicts, one per page
        for page_result in results:
            layout_info_path = page_result.get('layout_info_path')
            if not layout_info_path or not os.path.exists(layout_info_path):
                continue

            with open(layout_info_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)

            # layout_data is a list of detected elements
            # Sort by reading order if not already (assuming dots_ocr output is sorted)
            # The prompt instructions say "All layout elements must be sorted according to human reading order."

            for element in layout_data:
                if element.get('category') == 'Table':
                    html_content = element.get('text', '')
                    if html_content:
                        table_df = self._html_to_dataframe(html_content)
                        if table_df is not None and not table_df.empty:
                            all_tables.append(table_df)

        if not all_tables:
            print(f"No tables found in {pdf_path}.")
            return

        # 3. Merge tables
        merged_df = self._merge_tables(all_tables)

        # 4. Save to CSV
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
        merged_df.to_csv(csv_path, index=False)
        print(f"Saved merged table to {csv_path}")

    def _html_to_dataframe(self, html_content):
        """
        Convert HTML table string to Pandas DataFrame.
        """
        try:
            # Wrap in <table> if not present, though usually dots_ocr returns <table>...</table>
            if not html_content.strip().startswith('<table'):
                 html_content = f"<table>{html_content}</table>"

            # Use StringIO to avoid FutureWarning
            dfs = pd.read_html(StringIO(html_content))
            if dfs:
                return dfs[0]
        except Exception as e:
            print(f"Error converting HTML to DataFrame: {e}")
        return None

    def _merge_tables(self, tables):
        """
        Merge a list of DataFrames into one, handling headers.
        Assumption: Cross-page tables have repeated headers or no header on continuation.
        """
        if not tables:
            return pd.DataFrame()

        if len(tables) == 1:
            return tables[0]

        merged_df = tables[0]

        for i in range(1, len(tables)):
            current_df = tables[i]

            # Check if columns match
            if list(merged_df.columns) == list(current_df.columns):
                # If columns match exactly, just append
                # But sometimes the header is repeated as the first row in data if read_html didn't detect it as header
                # Or read_html detected it as header, so columns match.
                merged_df = pd.concat([merged_df, current_df], ignore_index=True)
            else:
                # If columns don't match, it might be that the second table has no header
                # and pandas assigned default int columns, OR it has a header that is slightly different (OCR error)

                # Heuristic: If number of columns is the same, assume it's continuation
                if len(merged_df.columns) == len(current_df.columns):
                    # Check if the first row of current_df looks like the header of merged_df
                    # If so, drop it.

                    # If current_df columns are integers (0, 1, 2...), it means no header was found.
                    # We can assign merged_df columns to it.
                    if pd.api.types.is_integer_dtype(current_df.columns):
                         current_df.columns = merged_df.columns
                         merged_df = pd.concat([merged_df, current_df], ignore_index=True)
                    else:
                        # Columns are strings but different.
                        # This happens when read_html treats the first row of a headerless continuation table as a header.
                        # We need to preserve this "header" as a data row.

                        # Convert headers to a dataframe row
                        header_row = pd.DataFrame([current_df.columns], columns=merged_df.columns)
                        # Fix current_df columns
                        current_df.columns = merged_df.columns
                        # Concatenate: merged + header_as_row + current_df data
                        merged_df = pd.concat([merged_df, header_row, current_df], ignore_index=True)
                else:
                    # Column count mismatch.
                    # This is tricky. Might be a different table or OCR error.
                    # We will append it anyway, but with outer join (pandas default for concat)
                    # OR we can warn and skip.
                    # Given the task "merge... into one csv", we probably should concat.
                    print(f"Warning: Table {i} has different column count ({len(current_df.columns)}) vs previous ({len(merged_df.columns)}). Concatenating anyway.")
                    merged_df = pd.concat([merged_df, current_df], ignore_index=True)

        return merged_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and merge tables from PDFs using dots_ocr.")
    parser.add_argument("folder_path", help="Path to the folder containing PDF files.")
    parser.add_argument("--output_dir", default="./output_csvs", help="Output directory for CSV files.")
    parser.add_argument("--vllm_ip", default="localhost", help="IP address of the vLLM service.")
    parser.add_argument("--vllm_port", default=8000, type=int, help="Port of the vLLM service.")
    parser.add_argument("--vllm_model", default="rednote-hilab/dots.ocr", help="Model name served by vLLM.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    extractor = TableExtractor(
        vllm_ip=args.vllm_ip,
        vllm_port=args.vllm_port,
        vllm_model=args.vllm_model,
        output_dir=args.output_dir
    )

    extractor.process_folder(args.folder_path)
