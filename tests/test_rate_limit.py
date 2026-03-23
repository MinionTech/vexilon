"""
tests/test_rate_limit.py — Unit tests for rate limiting

Tests verify the RateLimiter class behavior.
"""

import pytest
import time

from app import RateLimiter


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_allows_first_request(self):
        """First request should be allowed."""
        limiter = RateLimiter(max_per_minute=10, max_per_hour=100)
        allowed, msg = limiter.is_allowed("user1")
        assert allowed is True
        assert msg == ""

    def test_blocks_over_minute_limit(self):
        """Should block when minute limit exceeded."""
        limiter = RateLimiter(max_per_minute=2, max_per_hour=100)

        # First two requests should be allowed
        allowed1, _ = limiter.is_allowed("user1")
        allowed2, _ = limiter.is_allowed("user1")
        assert allowed1 is True
        assert allowed2 is True

        # Third request should be blocked
        allowed3, msg = limiter.is_allowed("user1")
        assert allowed3 is False
        assert "per minute" in msg

    def test_separate_users_independent(self):
        """Different users should have independent limits."""
        limiter = RateLimiter(max_per_minute=1, max_per_hour=100)

        # User 1 hits limit
        allowed1, _ = limiter.is_allowed("user1")
        assert allowed1 is True

        allowed1_again, _ = limiter.is_allowed("user1")
        assert allowed1_again is False

        # User 2 should still be allowed
        allowed2, _ = limiter.is_allowed("user2")
        assert allowed2 is True

    def test_cleans_old_requests(self):
        """Old requests should be cleaned after 1 hour."""
        limiter = RateLimiter(max_per_minute=10, max_per_hour=3)

        # Add some requests
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")

        # Manually add old request (simulating time passage)
        limiter.requests["user1"].append(time.time() - 7200)  # 2 hours ago

        # Should still have the old entry in the list but filtered for recent
        allowed, _ = limiter.is_allowed("user1")
        assert (
            allowed is True
        )  # 2 recent requests are within limits; old request is cleaned

    def test_hour_limit_enforced(self):
        """Should block when hourly limit exceeded."""
        limiter = RateLimiter(max_per_minute=100, max_per_hour=2)

        # Two requests allowed per hour
        allowed1, _ = limiter.is_allowed("user1")
        allowed2, _ = limiter.is_allowed("user1")
        assert allowed1 is True
        assert allowed2 is True

        # Third request should be blocked by hourly limit
        allowed3, msg = limiter.is_allowed("user1")
        assert allowed3 is False
        assert "per hour" in msg

    def test_returns_error_message(self):
        """Should return descriptive error message when blocked."""
        limiter = RateLimiter(max_per_minute=1, max_per_hour=100)

        limiter.is_allowed("user1")
        allowed, msg = limiter.is_allowed("user1")

        assert allowed is False
        assert "Rate limit exceeded" in msg
        assert "1" in msg
        assert "per minute" in msg
