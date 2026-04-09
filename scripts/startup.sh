#!/bin/bash
set -e

# The app now intelligently handles the index:
# 1. Attempts to load pre-computed index (instant)
# 2. Rebuilds from sources ONLY if cache is missing or corrupt
exec python app.py
