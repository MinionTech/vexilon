"""
Playwright configuration for e2e tests in Docker environment.
"""

import os

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import wait_for_custom_js


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


@pytest.fixture
def loaded_page(page: Page, app_url: str) -> Page:
    """Navigate to the app and wait for custom JS to attach."""
    page.goto(app_url, wait_until="domcontentloaded")
    wait_for_custom_js(page)
    return page
