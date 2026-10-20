import json
import logging

from heart_ml.logging_config import JsonFormatter, configure_logging


def test_json_formatter_includes_safe_context_and_exception():
    try:
        raise ValueError("bad input")
    except ValueError:
        record = logging.LogRecord("test", logging.ERROR, __file__, 12, "failed", (),
                                   __import__("sys").exc_info())
    record.event = "request_failed"
    record.request_id = "request-1"
    record.unused_context = "not serialized"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "request_failed"
    assert payload["request_id"] == "request-1"
    assert "ValueError: bad input" in payload["exception"]
    assert "unused_context" not in payload


def test_configure_logging_uses_environment_level(monkeypatch):
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    capture = []
    monkeypatch.setattr("heart_ml.logging_config.logging.captureWarnings", capture.append)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    try:
        configure_logging()
        assert root.level == logging.DEBUG
        assert root.handlers[0].formatter.__class__ is JsonFormatter
        assert capture == [True]
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)