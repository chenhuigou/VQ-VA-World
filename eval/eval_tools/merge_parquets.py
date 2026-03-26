# Copyright 2025 Zeyu Wang & Zilong Chen.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import pandas as pd
import re


def merge_parquet(output_dir, output_path=None, delete_shards=False):
    """Merge shard_*.parquet files into a single results.parquet."""
    shard_paths = []
    for fname in os.listdir(output_dir):
        if re.match(r"shard_\d+\.parquet$", fname):
            shard_paths.append(os.path.join(output_dir, fname))

    if not shard_paths:
        raise ValueError(f"No shard_*.parquet files found in {output_dir}")

    dfs = []
    for shard_path in sorted(shard_paths, key=lambda x: int(re.search(r"(\d+)", x).group(1))):
        print(f"Reading: {shard_path}")
        dfs.append(pd.read_parquet(shard_path))

    df_out = pd.concat(dfs, ignore_index=True)
    if output_path is None:
        output_path = os.path.join(output_dir, "results.parquet")
    df_out.to_parquet(output_path, index=False)
    print(f"Results saved to: {output_path}")

    if delete_shards:
        for path in shard_paths:
            try:
                os.remove(path)
                print(f"Deleted shard: {path}")
            except Exception as e:
                print(f"Failed to delete {path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Merge shard parquet files into one results.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing shard parquet files")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for merged parquet (default: output_dir/results.parquet)")
    parser.add_argument("--delete_shards", action="store_true", help="Delete shards after successful merge")
    args = parser.parse_args()

    merge_parquet(args.output_dir, args.output_path, args.delete_shards)


if __name__ == "__main__":
    main()
