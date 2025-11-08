"""Simple script to auto-answer main.py prompts via file input."""
# Create input file with all answers
with open('main_inputs.txt', 'w') as f:
    f.write("1\n")       # Mode 1
    f.write("\n")        # Trading pair (default)
    f.write("\n")        # Balance (default)
    f.write("1000\n")    # Population
    f.write("2\n")       # Generations
    f.write("1\n")       # Cycles (1 cycle to fit available data)
    f.write("5\n")       # Days (reduced to fit 10k bar limit)
    f.write("\n")        # Timeframe (default)
    f.write("\n")        # Min leverage (default)
    f.write("\n")        # Max leverage (default)
    f.write("\n")        # Min indicators (default)
    f.write("\n")        # Max indicators (default)
    f.write("\n")        # Min risk (default)
    f.write("\n")        # Max risk (default)
    f.write("\n")        # Random seed (default)
    f.write("\n")        # Seed value (default)

print("Created main_inputs.txt")
print("Parameters: 1000 bots, 2 generations, 1 cycle, 5 days")
