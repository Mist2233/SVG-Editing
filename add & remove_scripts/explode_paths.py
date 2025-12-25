import argparse
from svgpathtools import svg2paths, wsvg

def main():
    parser = argparse.ArgumentParser(description="Explode compound SVG paths into separate paths")
    parser.add_argument("input_path", help="Input SVG path")
    parser.add_argument("output_path", help="Output SVG path")
    args = parser.parse_args()

    print(f"Loading: {args.input_path}")
    paths, attributes = svg2paths(args.input_path)
    
    new_paths = []
    new_attributes = []
    
    count_exploded = 0
    
    for path, attr in zip(paths, attributes):
        # Check for sub-paths
        subpaths = path.continuous_subpaths()
        
        if len(subpaths) > 1:
            count_exploded += 1
            for sub in subpaths:
                new_paths.append(sub)
                new_attributes.append(attr.copy()) # Copy attributes for each sub-path
        else:
            new_paths.append(path)
            new_attributes.append(attr)
            
    print(f"Exploded {count_exploded} compound paths.")
    print(f"Total paths: {len(paths)} -> {len(new_paths)}")
    
    wsvg(new_paths, attributes=new_attributes, filename=args.output_path)
    print(f"Saved to: {args.output_path}")

if __name__ == "__main__":
    main()
