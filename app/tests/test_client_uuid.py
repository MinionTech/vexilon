"""
tests/test_client_uuid.py — Unit tests for pseudonymous client UUID handling

Tests verify _parse_client_uuid only accepts well-formed UUID v4 strings,
rejecting anything that could be used to inject arbitrary content into
session state or logs via an untrusted window_message payload.
"""

import uuid

from main import _parse_client_uuid


class TestParseClientUuid:
    """Tests for the _parse_client_uuid validator."""

    def test_accepts_valid_uuid4(self):
        """A well-formed UUID v4 string should be accepted and normalized."""
        valid = str(uuid.uuid4())
        result = _parse_client_uuid(valid)
        assert result == valid

    def test_accepts_uppercase_uuid4(self):
        """UUID string casing should not affect validity."""
        valid = str(uuid.uuid4()).upper()
        result = _parse_client_uuid(valid)
        assert result is not None

    def test_rejects_non_uuid_string(self):
        """Arbitrary strings should be rejected."""
        assert _parse_client_uuid("not-a-uuid") is None

    def test_rejects_wrong_uuid_version(self):
        """UUID v1 (e.g. time-based) should be rejected since v4 is required."""
        v1 = str(uuid.uuid1())
        assert _parse_client_uuid(v1) is None

    def test_rejects_none(self):
        """None input should be rejected."""
        assert _parse_client_uuid(None) is None

    def test_rejects_non_string_types(self):
        """Non-string payloads (e.g. dict, list, int) should be rejected."""
        assert _parse_client_uuid(12345) is None
        assert _parse_client_uuid({"uuid": str(uuid.uuid4())}) is None
        assert _parse_client_uuid([str(uuid.uuid4())]) is None

    def test_rejects_empty_string(self):
        """Empty string should be rejected."""
        assert _parse_client_uuid("") is None

    def test_rejects_oversized_string(self):
        """Long garbage strings should be rejected, not partially parsed."""
        assert _parse_client_uuid("a" * 10000) is None
