"""
Convert lexicon JSON files to CSV format
"""
import os
import json
import csv
import argparse
import glob
import sys
from typing import List, Dict, Any


def export_to_csv(lexicon: List[Dict[str, Any]], output_file: str) -> None:
    """
    Export lexicon data to CSV format
    """
    all_keys = set()
    for entry in lexicon:
        all_keys.update(entry.keys())

    headers = sorted(list(all_keys))
    priority_fields = ['word', 'definitions', 'part_of_speech', 'examples', 'section', 'source']

    for field in reversed(priority_fields):
        if field in headers:
            headers.remove(field)
            headers = [field] + headers

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for entry in lexicon:
            row = {}
            for k, v in entry.items():
                if isinstance(v, list):
                    if v and isinstance(v[0], dict):
                        nested_items = []
                        for item in v:
                            nested_str = "|".join(f"{nk}:{nv}" for nk, nv in item.items())
                            nested_items.append(nested_str)
                        row[k] = ";".join(nested_items)
                    else:
                        row[k] = ";".join(str(item) for item in v)
                else:
                    row[k] = v
            writer.writerow(row)

    print(f"Exported {len(lexicon)} entries to CSV: {output_file}")


def cli_main():
    parser = argparse.ArgumentParser(description='Convert lexicon JSON files to CSV')
    parser.add_argument('input', type=str, nargs='?',
                        help='Input JSON file (or directory with --dir flag). If omitted, all JSON files in the data/lexicon directory will be processed.')
    parser.add_argument('--dir', '-d', action='store_true',
                        help='Treat input as directory containing JSON files')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for CSV files (defaults to same as input)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose debug information')

    args = parser.parse_args()

    if args.input is None:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'lexicon')
        args.input = default_dir
        args.dir = True
        print(f"No input specified. Processing all JSON files in {default_dir}")
    else:
        print(f"Converting {args.input} to CSV format")

    if not args.dir and not os.path.exists(args.input):
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'lexicon')
        possible_path = os.path.join(default_dir, args.input)
        if os.path.exists(possible_path):
            args.input = possible_path
            print(f"Found input file at: {args.input}")
        else:
            if not args.input.endswith('.json'):
                possible_path = os.path.join(default_dir, f"{args.input}.json")
                if os.path.exists(possible_path):
                    args.input = possible_path
                    print(f"Found input file at: {args.input}")

    if os.path.isdir(args.input):
        json_files = glob.glob(os.path.join(args.input, '*.json'))
        output_dir = args.output_dir or args.input
    else:
        json_files = [args.input]
        output_dir = args.output_dir or os.path.dirname(args.input)

    os.makedirs(output_dir, exist_ok=True)

    for json_file in json_files:
        try:
            print(f"Processing {json_file}...")

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            base_name = os.path.splitext(os.path.basename(json_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}.csv")

            export_to_csv(data, output_file)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")


# ---------------------------------------------------------------------
# UI Mode (Streamlit)
# ---------------------------------------------------------------------
def ui_main():
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="Lexicon JSON → CSV Converter", layout="wide")
    st.title("📑 Lexicon Converter")

    uploaded_json = st.file_uploader("Upload Lexicon JSON", type=["json"], accept_multiple_files=True)
    output_dir = st.text_input("Output directory", value=os.path.join("scrapers", "data", "lexicon"))

    if uploaded_json:
        os.makedirs(output_dir, exist_ok=True)

        for file in uploaded_json:
            try:
                data = json.load(file)
                base_name = os.path.splitext(file.name)[0]
                output_file = os.path.join(output_dir, f"{base_name}.csv")

                export_to_csv(data, output_file)
                st.success(f"✅ Converted {file.name} → {output_file}")

                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error converting {file.name}: {e}")


if __name__ == "__main__":
    if "streamlit" in sys.argv[0]:  # UI mode
        ui_main()
    else:  # CLI mode
        cli_main()
