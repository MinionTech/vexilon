#!/bin/bash
# Ask the committed golden questions on a Hugging Face Space.
# Usage: ./.github/scripts/golden_questions.sh <space_id>
# A non-zero exit blocks PROD promotion. This script does not deploy or restart.
set -euo pipefail

if [ "$#" -ne 1 ] || [ -z "${1}" ]; then
    echo "Usage: ./.github/scripts/golden_questions.sh <space_id>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/golden_questions.py" "${1}"
