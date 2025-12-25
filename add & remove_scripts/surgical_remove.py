import pydiffvg
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Surgically remove specific SVG groups/layers")
    parser.add_argument("svg_path", help="Input SVG path")
    parser.add_argument("output_path", help="Output SVG path")
    parser.add_argument("--remove_groups", type=int, nargs="+", help="IDs of groups to remove")
    parser.add_argument("--remove_shapes", type=int, nargs="+", help="IDs of shapes to remove")
    args = parser.parse_args()

    print(f"Loading: {args.svg_path}")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(args.svg_path)
    
    print(f"Original: {len(shape_groups)} groups, {len(shapes)} shapes")
    
    remove_group_ids = set(args.remove_groups) if args.remove_groups else set()
    remove_shape_ids = set(args.remove_shapes) if args.remove_shapes else set()
    
    print(f"Removing Groups: {remove_group_ids}")
    print(f"Removing Shapes: {remove_shape_ids}")
    
    kept_groups = []
    
    # We need to rebuild the shapes list and re-index
    # Strategy:
    # 1. Determine which shapes are kept.
    #    A shape is kept if:
    #    - Its group is NOT in remove_group_ids
    #    - It is NOT in remove_shape_ids
    
    # Map old_shape_id -> (is_kept, new_shape_id)
    old_to_new_map = {}
    final_shapes = []
    
    # But shape indices are determined by their order in the 'shapes' list.
    # So we iterate over original 'shapes' list.
    
    # Wait, we need to know which group a shape belongs to, to check group removal.
    shape_to_group_idx = {}
    for g_idx, group in enumerate(shape_groups):
        for sid in group.shape_ids:
            shape_to_group_idx[int(sid)] = g_idx
            
    current_new_idx = 0
    for old_idx, shape in enumerate(shapes):
        g_idx = shape_to_group_idx.get(old_idx)
        
        # Decide if we keep this shape
        keep = True
        
        if old_idx in remove_shape_ids:
            keep = False
        
        if g_idx is not None and g_idx in remove_group_ids:
            keep = False
            
        if keep:
            final_shapes.append(shape)
            old_to_new_map[old_idx] = current_new_idx
            current_new_idx += 1
        else:
            old_to_new_map[old_idx] = None # Removed
            
    # Now rebuild groups
    final_groups = []
    for g_idx, group in enumerate(shape_groups):
        if g_idx in remove_group_ids:
            continue
            
        # Filter shape_ids in this group
        new_ids = []
        for old_sid in group.shape_ids:
            old_sid = int(old_sid)
            if old_to_new_map.get(old_sid) is not None:
                new_ids.append(old_to_new_map[old_sid])
                
        if len(new_ids) > 0:
            group.shape_ids = torch.tensor(new_ids)
            final_groups.append(group)
        else:
            print(f"  - Group {g_idx} became empty, dropping.")
            
    # 4. Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    pydiffvg.save_svg(args.output_path, canvas_width, canvas_height, final_shapes, final_groups)
    print(f"Saved to: {args.output_path}")
    print(f"Final: {len(final_groups)} groups, {len(final_shapes)} shapes")

if __name__ == "__main__":
    main()
