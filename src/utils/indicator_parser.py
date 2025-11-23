import re
import ast
from typing import List
from src.indicators.gpu_indicators import get_all_gpu_indicators, get_gpu_indicator_name


def parse_indices(indices_field: str) -> List[int]:
    if not indices_field:
        return []
    try:
        x = ast.literal_eval(indices_field)
        if isinstance(x, (list, tuple)):
            return [int(i) for i in x]
    except Exception:
        pass
    # If indices are given as names (e.g., "SMA(50), RSI(14)"), map names to GPU indicator indices
    parts = re.split(r"[,|]", indices_field)
    parts = [p.strip() for p in parts if p.strip()]
    indicators = []
    # Build name lookup (lowercase base name -> index)
    name_lookup = {}
    for i in get_all_gpu_indicators():
        name = get_gpu_indicator_name(i).split('(')[0].strip().lower()
        name_lookup[name] = i
    for p in parts:
        # extract base name (before '(' ) and try to map
        base = p.split('(')[0].strip().lower()
        if base in name_lookup:
            indicators.append(name_lookup[base])
            continue
        # fallback: extract digits and treat them as explicit indices
        nums = re.findall(r"\d+", p)
        if nums:
            indicators.extend([int(n) for n in nums])
    return indicators


def parse_indicator_params(params_field: str) -> List[List[float]]:
    if not params_field:
        return []
    parts = [p.strip() for p in params_field.split('|') if p.strip()]
    out = []
    for p in parts:
        # There may be two parentheses groups: e.g. SMA(50)(4.6,183.2,0.0)
        # Prefer explicit param from the first parentheses if it contains a single number (e.g., period)
        # Then parse the trailing parentheses which often contain ranges or param triples.
        explicit_val = None
        first_paren = re.search(r"^[^\(]+\(([0-9\s]+)\)", p)
        if first_paren:
            try:
                explicit_val = float(first_paren.group(1).strip())
            except Exception:
                explicit_val = None
        m = re.search(r"\(([0-9\-\.,\s]+)\)\s*$", p)
        if not m:
            if '(' in p and ')' in p:
                inside = p[p.rfind('(')+1:p.rfind(')')]
            else:
                inside = ''
        else:
            inside = m.group(1)
        if not inside:
            vals = [0.0, 0.0, 0.0]
        else:
            inside_norm = inside.replace(',', '.')
            vals = []
            for t in inside_norm.split(','):
                t = t.strip()
                if not t:
                    continue
                try:
                    vals.append(float(t))
                except Exception:
                    vals.append(0.0)
            while len(vals) < 3:
                vals.append(0.0)
            vals = vals[:3]
        # If an explicit numeric param was present in the indicator name (e.g. SMA(50)), prefer that
        if explicit_val is not None and explicit_val > 0:
            vals[0] = explicit_val
        out.append(vals)
    return out
