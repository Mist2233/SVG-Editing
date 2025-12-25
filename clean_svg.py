import argparse
import os
from svgpathtools import svg2paths, Path, Line, QuadraticBezier, CubicBezier, Arc
import svgwrite
import xml.etree.ElementTree as ET
import re

def get_svg_dimensions(svg_path):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        width = root.get('width')
        height = root.get('height')
        viewbox = root.get('viewBox')
        return width, height, viewbox
    except:
        return None, None, None

def parse_css_classes(svg_path):
    """
    简单解析 SVG 中的 <style> 标签，提取 .classname { ... }
    返回字典: {'a': {'fill': '#...'}, 'b': {...}}
    """
    css_map = {}
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 寻找 <style>...</style>
        style_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
        if style_match:
            style_content = style_match.group(1)
            # 匹配 .classname { key:val; key:val }
            # 简化版正则，假设格式比较标准
            classes = re.findall(r'\.([a-zA-Z0-9_-]+)\s*\{(.*?)\}', style_content)
            
            for cls_name, props in classes:
                props_dict = {}
                # 分割属性
                items = props.split(';')
                for item in items:
                    if ':' in item:
                        k, v = item.split(':', 1)
                        props_dict[k.strip()] = v.strip()
                css_map[cls_name] = props_dict
    except Exception as e:
        print(f"  [Warning] Failed to parse CSS: {e}")
        
    return css_map

def clean_svg_file(input_path, output_path):
    print(f"--> Cleaning: {input_path}")
    
    width, height, viewbox = get_svg_dimensions(input_path)
    css_classes = parse_css_classes(input_path)
    if css_classes:
        print(f"  Found {len(css_classes)} CSS classes.")
    
    try:
        paths, attributes = svg2paths(input_path)
    except Exception as e:
        print(f"  [Error] Failed to load SVG: {e}")
        return

    new_paths = []
    new_attributes = []
    
    count_split = 0
    
    for path, attr in zip(paths, attributes):
        clean_attr = {}
        valid_keys = ['fill', 'stroke', 'stroke-width', 'stroke-opacity', 'fill-opacity', 'opacity', 'stroke-linecap', 'stroke-linejoin']
        
        # 0. [新增] 应用 CSS Class 样式
        if 'class' in attr:
            cls_name = attr['class']
            if cls_name in css_classes:
                # 把 CSS 里的属性合并进来
                css_props = css_classes[cls_name]
                for k, v in css_props.items():
                    if k in valid_keys:
                        clean_attr[k] = v
        
        # 1. 解析 style (行内样式优先于 CSS class)
        if 'style' in attr:
            style_str = attr['style']
            styles = style_str.split(';')
            for s in styles:
                if ':' in s:
                    k, v = s.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    if k in valid_keys:
                        clean_attr[k] = v
        
        # 2. 解析直接属性 (直接属性优先于 style? SVG 规范其实是 style > 直接 > class)
        # 这里我们假设 svgpathtools 返回的 attr 包含了直接属性
        # 为了保险，我们让 style 覆盖直接属性，所以把这一步放在 style 解析之前或之后需要权衡
        # 通常直接属性已经在 attr 里了，我们再遍历一遍
        for k, v in attr.items():
            if k in valid_keys:
                clean_attr[k] = v
        
        # 3. 智能推断缺省值
        has_fill = 'fill' in clean_attr and clean_attr['fill'] != 'none'
        has_stroke = 'stroke' in clean_attr and clean_attr['stroke'] != 'none'
        
        if not has_fill and not has_stroke:
            # 只有当确实什么都没有时，才给默认值。
            # 如果解析了 CSS 还是没有，那可能是真的黑色，或者透明
            # 这里的 black 兜底其实有风险，改为 'none' 更安全，或者如果不确定就 black
            # 鉴于 qianzi.svg 之前全黑，说明它确实依赖 CSS。现在有了 CSS 解析，应该不会走到这一步。
            # 如果还走到这一步，说明真的没颜色。
            clean_attr['fill'] = 'black' 
            
        # 4. 处理路径 (不拆分复合路径)
        # 保持复合路径完整性以正确处理 fill-rule (挖空效果)
        # 先拆分成连续子路径，分别转换，再拼回去
        subpaths = path.continuous_subpaths()
        cleaned_subpaths_d = []
        
        for subpath in subpaths:
            clean_subpath = Path()
            for segment in subpath:
                if isinstance(segment, Line):
                    clean_subpath.append(CubicBezier(segment.start, segment.start, segment.end, segment.end))
                elif isinstance(segment, QuadraticBezier):
                    p0, p1, p2 = segment.start, segment.control, segment.end
                    c1 = p0 + (2/3) * (p1 - p0)
                    c2 = p2 + (2/3) * (p1 - p2)
                    clean_subpath.append(CubicBezier(p0, c1, c2, p2))
                else:
                    clean_subpath.append(segment)
            cleaned_subpaths_d.append(clean_subpath.d())
        
        # 将处理后的子路径重新组合成一个复合路径字符串
        full_d = " ".join(cleaned_subpaths_d)
        
        new_paths.append(full_d)
        new_attributes.append(clean_attr.copy())

    print(f"  Result: {len(paths)} paths -> {len(new_paths)} paths (Kept compound paths merged)")

    dwg = svgwrite.Drawing(output_path, profile='tiny')
    
    if viewbox:
        clean_viewbox = ' '.join(viewbox.replace(',', ' ').split())
        dwg['viewBox'] = clean_viewbox
        
    if width:
        dwg['width'] = width
    if height:
        dwg['height'] = height
        
    for d_str, attr in zip(new_paths, new_attributes):
        # d_str 已经是字符串了
        dwg.add(dwg.path(d=d_str, **attr))
        
    dwg.save()
    print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Input SVG files to clean")
    args = parser.parse_args()
    
    for f in args.files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
            
        base, ext = os.path.splitext(f)
        output_path = f"{base}_fixed{ext}"
        
        clean_svg_file(f, output_path)
