from svgpathtools import svg2paths
import numpy as np

import math

def inspect(svg_path):
    paths, attributes = svg2paths(svg_path)
    
    print(f"{'ID':<5} | {'Closed':<8} | {'Compact':<8} | {'Length':<10} | {'bbox (w, h)'}")
    print("-" * 60)
    
    for i, path in enumerate(paths):
        try:
            is_closed = path.isclosed()
            length = path.length()
            num_segments = len(path)
            
            # Bounding box
            xmin, xmax, ymin, ymax = path.bbox()
            w = xmax - xmin
            h = ymax - ymin
            
            area = abs(path.area())
            
            # Compactness
            compactness = (4 * math.pi * area) / (length ** 2) if length > 0 else 0
            
            print(f"{i:<5} | {str(is_closed):<8} | {compactness:<8.2f} | {length:<10.2f} | ({w:.2f}, {h:.2f}) | {area:.2f}")
        except Exception as e:
            print(f"{i:<5} | Error: {e}")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        inspect("edit_pair/add & remove/2/fish_exploded.svg")
