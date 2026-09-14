"""
Playwright configuration for e2e tests in Docker environment.
"""

import os

import pytest


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Configure browser launch args for Docker/CI environment."""
    return {
        **browser_type_launch_args,
        "args": [
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-gpu",
            "--disable-software-rasterizer",
        ],
    }


@pytest.fixture(scope="module")
def app_url():
    """Base URL for the running Chainlit app."""
    return os.getenv("APP_URL", "http://localhost:7860")
