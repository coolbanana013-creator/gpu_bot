import sys
sys.path.insert(0, 'src')
try:
    import live_trading.indicator_calculator as ic
    print('Imported successfully')
except SyntaxError as e:
    print('SyntaxError:', e)
except Exception as e:
    print('Import error:', e)
