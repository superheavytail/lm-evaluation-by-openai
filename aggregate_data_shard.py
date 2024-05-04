# This code is utility for multi-gpu support
# It aggregates shards from step 1 (which executed with multi-gpu, to perform step 2.
import jsonlines
import json
from pathlib import Path
from itertools import count
from fire import Fire


OUTPUT_DIR = './generated'


def main(sharded_file_name: str):
    files = Path(OUTPUT_DIR)
    files = files.glob(f'{sharded_file_name}*')
    files = sorted(list(files), key=lambda x: x.stem)

    files_list = []
    for f in files:
        files_list.append([e for e in jsonlines.open(f).iter()])

    files_generator_list = [iter(e) for e in files_list]

    aggregated_list = []
    for i in count():
        try:
            aggregated_list.append(next(files_generator_list[i % len(files)]))
        except StopIteration:
            break

    aggregated_output_file = (Path(OUTPUT_DIR) / sharded_file_name).with_suffix('.jsonl')
    with open(aggregated_output_file, "w", encoding="utf-8") as f:
        for e in aggregated_list:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")


if __name__ == '__main__':
    Fire(main)
