"""Generate a placeholder daily stats JSON file.

IMPORTANT: every value here is randomly generated. They are NOT measurements
of the real models or real users. The output carries ``"synthetic": true`` so
consumers can tell.

The output path defaults to ``data/daily_stats.json`` and can be overridden
with the ``DAILY_STATS_FILE`` environment variable.
"""
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("daily_stats")

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "daily_stats.json")
DATA_FILE = os.environ.get("DAILY_STATS_FILE", DEFAULT_FILE)

LEARNING_STYLES = ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"]
MODELS = ["CatBoost", "Neural Network", "Decision Tree", "Random Forest"]


def generate_stats(path: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Create random placeholder stats, write them to ``path`` and return them.

    Raises OSError if the file cannot be written.
    """
    rng = random.Random(seed)
    path = path or DATA_FILE
    now = datetime.now()

    stats: Dict[str, Any] = {
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.isoformat(),
        "active_users": rng.randint(50, 500),
        "top_learning_style": rng.choice(LEARNING_STYLES),
        "model_accuracy": round(rng.uniform(0.85, 0.98), 4),
        "best_performing_model": rng.choice(MODELS),
        "predictions_made": rng.randint(100, 1000),
        "synthetic": True,
        "note": "Randomly generated placeholder values; not real measurements.",
    }

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.warning("Generating SYNTHETIC placeholder stats (random values).")
    try:
        stats = generate_stats()
    except OSError as exc:
        logger.error("Error generating stats: %s", exc)
        return 1
    logger.info("Wrote stats for %s to %s", stats["date"], DATA_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
