import sys
from pathlib import Path

import pytest

_MODELS = Path(__file__).resolve().parents[1]
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))

from db.sql_identifiers import safe_identifier, ALLOWED_TABLES


def test_known_table_passes():
    assert safe_identifier("FighterStats", ALLOWED_TABLES) == "FighterStats"


def test_injection_payload_rejected():
    with pytest.raises(ValueError):
        safe_identifier("users; DROP TABLE users;--", ALLOWED_TABLES)


def test_unknown_identifier_rejected():
    with pytest.raises(ValueError):
        safe_identifier("not_a_table", ALLOWED_TABLES)
