import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.llm import chat
print(chat('Name three common solvents used in chemical engineering. Be brief.'))
