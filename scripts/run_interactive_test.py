"""Run main.py in interactive mode for debugging diversity issues."""
import sys
from io import StringIO

print("="*70)
print("INTERACTIVE MODE ENABLED")
print("The program will pause after each generation for inspection")
print("Press Enter at each pause to continue to the next generation")
print("="*70)
print()

# Mock stdin for the prompts, but include 'y' for interactive mode
# Mode 1, then all defaults, then 'y' for interactive mode
mock_inputs = ["1"] + [""] * 20 + ["y"]

# Replace stdin
original_stdin = sys.stdin
sys.stdin = StringIO("\n".join(mock_inputs))

try:
    # Import and run main
    import main
    main.main()
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\n\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore stdin
    sys.stdin = original_stdin
