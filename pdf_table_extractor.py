import os
import glob
import argparse
import pandas as pd
from io import StringIO
import json
import warnings
import requests

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

        # Verify and fetch the correct model name from vLLM
        resolved_model = self._resolve_model_name(vllm_ip, vllm_port, vllm_model)
        if resolved_model != vllm_model:
            print(f"Switching model from '{vllm_model}' to '{resolved_model}'")

        self.parser = DotsOCRParser(
            ip=vllm_ip,
            port=vllm_port,
            model_name=resolved_model,
            num_thread=num_thread,
            dpi=dpi,
            output_dir=output_dir,
            protocol='http', # Assuming http as per user instruction "already deployed"
        )

    def _resolve_model_name(self, ip, port, requested_model):
        """
        Query vLLM for available models. If requested_model is not found,
        try to pick a reasonable default.
        """
        url = f"http://{ip}:{port}/v1/models"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                available_models = [m['id'] for m in data.get('data', [])]
                print(f"Available models on vLLM: {available_models}")

                if requested_model in available_models:
                    return requested_model

                # If only one model is available, use it
                if len(available_models) == 1:
                    print(f"Model '{requested_model}' not found. Using the only available model: '{available_models[0]}'")
                    return available_models[0]

                # If multiple models, look for one that looks like 'dots' or 'ocr'
                for m in available_models:
                    if 'dots' in m.lower() and 'ocr' in m.lower():
                        print(f"Model '{requested_model}' not found. Found similar model: '{m}'")
                        return m

                # Fallback to the first one if we can't match
                if available_models:
                    print(f"Model '{requested_model}' not found. Falling back to: '{available_models[0]}'")
                    return available_models[0]

        except Exception as e:
            print(f"Warning: Could not connect to vLLM to list models ({e}). Using provided model name: '{requested_model}'")

        return requested_model

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

            if isinstance(layout_data, list):
                for element in layout_data:
                    if isinstance(element, dict) and element.get('category') == 'Table':
                        html_content = element.get('text', '')
                        if html_content:
                            table_df = self._html_to_dataframe(html_content)
                            if table_df is not None and not table_df.empty:
                                all_tables.append(table_df)
            else:
                print(f"Warning: Unexpected layout data format in {layout_info_path}. Expected list, got {type(layout_data)}")

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
                df = dfs[0]
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
                return df
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
            return self._clean_dataframe(tables[0])

        merged_df = tables[0]

        for i in range(1, len(tables)):
            current_df = tables[i]

            # Check if columns match
            if list(merged_df.columns) == list(current_df.columns):
                # If columns match exactly, just append
                merged_df = pd.concat([merged_df, current_df], ignore_index=True)
            else:
                # If columns don't match, it might be that the second table has no header
                # and pandas assigned default int columns, OR it has a header that is slightly different (OCR error)

                # Heuristic: If number of columns is the same, assume it's continuation
                if len(merged_df.columns) == len(current_df.columns):

                    # If current_df columns are integers (0, 1, 2...), it means no header was found.
                    # We can assign merged_df columns to it.
                    if pd.api.types.is_integer_dtype(current_df.columns):
                         current_df.columns = merged_df.columns
                         merged_df = pd.concat([merged_df, current_df], ignore_index=True)
                    else:
                        # Columns are strings but different.
                        # It could be a REPEATED header (OCR error/slight diff) or DATA.

                        # Simple heuristic: compare similarity of column names to merged_df columns.
                        # If highly similar, assume it's a header and skip inserting it as data.

                        is_header = False
                        # Check intersection of words or exact match ratio
                        col_str1 = " ".join([str(c) for c in merged_df.columns])
                        col_str2 = " ".join([str(c) for c in current_df.columns])

                        # If more than 50% of the words are common, it's likely a header
                        # A better check: check if key columns like "NO", "NAME" are present
                        header_keywords = ["NO", "NAME", "FORMULA", "CAS", "TMIN", "TMAX", "A", "B", "C", "D"]
                        matches = sum(1 for k in header_keywords if k in str(current_df.columns))
                        if matches >= 2:
                            is_header = True

                        if is_header:
                            # It is a header, just update columns and concat data
                            current_df.columns = merged_df.columns
                            merged_df = pd.concat([merged_df, current_df], ignore_index=True)
                        else:
                            # Not a header (likely data read as header because of no thead)
                            # Convert headers to a dataframe row
                            header_row = pd.DataFrame([current_df.columns], columns=merged_df.columns)
                            # Fix current_df columns
                            current_df.columns = merged_df.columns
                            # Concatenate: merged + header_as_row + current_df data
                            merged_df = pd.concat([merged_df, header_row, current_df], ignore_index=True)
                else:
                    # Column count mismatch.
                    print(f"Warning: Table {i} has different column count ({len(current_df.columns)}) vs previous ({len(merged_df.columns)}). Concatenating anyway.")
                    merged_df = pd.concat([merged_df, current_df], ignore_index=True)

        return self._clean_dataframe(merged_df)

    def _clean_dataframe(self, df):
        """
        Remove rows that look like repeated headers, footer text, or artifacts.
        """
        def is_valid_row(row):
            # Convert row to string to check content
            row_str = " ".join([str(x) for x in row.values])

            # Check for header artifacts (e.g. tuple strings from MultiIndex)
            if "('NO', 'NO')" in row_str or "('FORMULA', 'FORMULA')" in row_str:
                return False

            # Check for repeated headers
            # If the row contains multiple header keywords
            header_keywords = ["NO", "NAME", "FORMULA", "CAS No", "TMIN", "TMAX"]
            matches = sum(1 for k in header_keywords if k in row_str)
            if matches >= 3:
                return False

            # Check for footer/legend text
            if "code: 1 - data" in row_str or "kgas - thermal conductivity" in row_str:
                return False

            return True

        return df[df.apply(is_valid_row, axis=1)]

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
