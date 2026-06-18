"""Tests for the model_loader SHA-256 integrity gate (anti-RCE).

The point of the gate is that pickle.load is never reached for an artifact whose
hash doesn't match a trusted expectation. Run from DelphiAIApp/Models:

    python -m pytest ml/tests/test_model_integrity.py -v
"""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.model_loader import MLPredictor


def _write(tmp_path, data: bytes) -> Path:
    p = tmp_path / "model_latest.pkl"
    p.write_bytes(data)
    return p


def test_mismatched_env_digest_refuses_to_unpickle(tmp_path, monkeypatch):
    # Arbitrary (non-pickle) bytes: if the gate worked, we never try to unpickle.
    model = _write(tmp_path, b"not a real model")
    monkeypatch.setenv("DELPHI_MODEL_SHA256", "deadbeef" * 8)  # wrong digest

    predictor = MLPredictor(model_path=str(model))

    assert predictor.is_available() is False
    assert "integrity check failed" in (predictor.get_load_error() or "")


def test_matching_env_digest_passes_gate(tmp_path, monkeypatch):
    # Correct digest -> gate passes, so loading proceeds and fails LATER at
    # unpickling (the bytes aren't a valid pickle). The distinction proves the
    # gate let it through rather than blocking on integrity.
    raw = b"not a real model"
    model = _write(tmp_path, raw)
    monkeypatch.setenv("DELPHI_MODEL_SHA256", hashlib.sha256(raw).hexdigest())

    predictor = MLPredictor(model_path=str(model))

    assert predictor.is_available() is False
    assert "integrity check failed" not in (predictor.get_load_error() or "")


def test_sidecar_digest_is_used_when_no_env_pin(tmp_path, monkeypatch):
    raw = b"not a real model"
    model = _write(tmp_path, raw)
    (tmp_path / "model_latest.pkl.sha256").write_text("deadbeef" * 8 + "\n")
    monkeypatch.delenv("DELPHI_MODEL_SHA256", raising=False)

    predictor = MLPredictor(model_path=str(model))

    assert predictor.is_available() is False
    assert "integrity check failed" in (predictor.get_load_error() or "")
