"""Auto-run main.py with defaults for testing."""
import sys
from io import StringIO

# Mock stdin with all default responses (just pressing Enter for each prompt)
mock_inputs = ["1"] + [""] * 20  # Mode 1, then empty strings for all prompts

# Replace stdin
original_stdin = sys.stdin
sys.stdin = StringIO("\n".join(mock_inputs))

try:
    # Import and run main
    import main
    main.main()
except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nError: {e}")
finally:
    # Restore stdin
    sys.stdin = original_stdin
