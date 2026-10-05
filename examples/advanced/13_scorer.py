import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Use the OlmoforgeScorer integration wrapper.

Run:  python examples/advanced/13_scorer.py
"""
from integrations.langchain import OlmoforgeScorer

scorer = OlmoforgeScorer(skill=0.8)
ranked = scorer.score("which next step?", ["A", "B", "C", "D"])
print("ranked:", ranked)
print("choice:", scorer.choose("which next step?", ["A", "B", "C", "D"]))
