#!/bin/bash
# Oracle: install the reference solution as the agent's submission.
set -euo pipefail

for ext in py cpp; do
    if [ -f "/solution/reference.$ext" ]; then
        cp "/solution/reference.$ext" "/app/solution.$ext"
        echo "copied reference.$ext -> /app/solution.$ext"
        exit 0
    fi
done

echo "No reference solution available." >&2
exit 1
