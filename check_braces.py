with open('src/gpu_kernels/backtest_with_precomputed.cl', 'r') as f:
    lines = f.readlines()

opens = 0
closes = 0
for i, line in enumerate(lines[1634:2390], start=1635):
    opens += line.count('{')
    closes += line.count('}')
    if '{' in line or '}' in line:
        print(f"Line {i}: open={opens} close={closes} balance={opens-closes} | {line[:80].rstrip()}")

print(f"\nFinal balance: {opens-closes}")
