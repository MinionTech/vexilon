import os

AGNAV_APP_NAME = os.getenv("AGNAV_APP_NAME", "BCGEU Navigator")
AGNAV_APP_DESCRIPTION = os.getenv("AGNAV_APP_DESCRIPTION", "BCGEU Agreement Navigator")
AGNAV_WELCOME_TITLE = os.getenv("AGNAV_WELCOME_TITLE", "BCGEU Navigator")


def get_brand() -> dict[str, str]:
    return {
        "name": AGNAV_APP_NAME,
        "description": AGNAV_APP_DESCRIPTION,
        "welcome_title": AGNAV_WELCOME_TITLE,
    }