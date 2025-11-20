import re
import ast
from typing import List


def parse_indices(indices_field: str) -> List[int]:
    if not indices_field:
        return []
    try:
        x = ast.literal_eval(indices_field)
        if isinstance(x, (list, tuple)):
            return [int(i) for i in x]
    except Exception:
        pass
    nums = re.findall(r"\d+", indices_field)
    return [int(n) for n in nums]


def parse_indicator_params(params_field: str) -> List[List[float]]:
    if not params_field:
        return []
    parts = [p.strip() for p in params_field.split('|') if p.strip()]
    out = []
    for p in parts:
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
        out.append(vals)
    return out
