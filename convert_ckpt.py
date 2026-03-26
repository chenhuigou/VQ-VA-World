#!/usr/bin/env python3
"""
Convert cdt-hf checkpoint to LightFusionWorld format.

ONLY converts VLM cross_attn keys:
  cross_attn_layers -> mm_attn_layers
  .cross_attn.     -> .mm_attn.       (only within cross_attn_layers)

Does NOT touch vgen_model.blocks.*.cross_attn (those stay as-is).

Usage:
  python convert_ckpt_v2.py --input /path/to/model.safetensors --output /path/to/output/model.safetensors
"""
import argparse
import os

def convert_key(key):
    # Only convert VLM cross_attn keys, not vgen_model ones
    if 'cross_attn_layers' in key:
        new_key = key.replace('cross_attn_layers', 'mm_attn_layers').replace('.cross_attn.', '.mm_attn.')
        return new_key
    return key

def main():
    parser = argparse.ArgumentParser(description='Convert cdt-hf ckpt to LightFusionWorld format')
    parser.add_argument('--input', required=True, help='Input safetensors file path')
    parser.add_argument('--output', required=True, help='Output safetensors file path')
    parser.add_argument('--dry-run', action='store_true', help='Only show key mapping without saving')
    args = parser.parse_args()

    assert args.input.endswith('.safetensors'), "Only .safetensors format supported"
    
    from safetensors.torch import load_file, save_file

    print(f"Loading checkpoint: {args.input}")
    state_dict = load_file(args.input, device="cpu")
    print(f"Total keys: {len(state_dict)}")

    new_state_dict = {}
    changed_keys = []
    unchanged_keys = []

    for key, value in state_dict.items():
        new_key = convert_key(key)
        if new_key != key:
            changed_keys.append((key, new_key))
        else:
            unchanged_keys.append(key)
        new_state_dict[new_key] = value

    print(f"\n--- Conversion Summary ---")
    print(f"Changed keys: {len(changed_keys)}")
    print(f"Unchanged keys: {len(unchanged_keys)}")
    
    # Verify no vgen keys were changed
    vgen_changed = [old for old, new in changed_keys if 'vgen_model' in old]
    print(f"vgen_model keys changed (should be 0): {len(vgen_changed)}")
    if vgen_changed:
        print("ERROR: vgen_model keys should not be changed!")
        return
    
    if changed_keys:
        print(f"\nChanged key examples (first 10):")
        for old, new in changed_keys[:10]:
            print(f"  {old}")
            print(f"  -> {new}")
    
    if args.dry_run:
        print("\n[DRY RUN] No file saved.")
        return

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_file(new_state_dict, args.output)
    print(f"\nSaved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024 / 1024:.2f} GB")

if __name__ == '__main__':
    main()
