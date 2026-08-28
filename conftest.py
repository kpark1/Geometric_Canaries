import sys
from pathlib import Path

# rollback/*.py runs as a script (sys.path[0] == rollback/); mirror that for tests
# so flat imports like `from validation import ...` resolve.
sys.path.insert(0, str(Path(__file__).parent / "rollback"))
