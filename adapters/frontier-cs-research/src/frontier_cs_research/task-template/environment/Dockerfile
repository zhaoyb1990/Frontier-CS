FROM {base_image}

# The verifier runs the problem's Python evaluator (evaluator.py via evaluate.sh).
# Ensure python3 exists — some base images (e.g. gcc:*) don't ship it. This is a
# no-op on images that already have python3 (Triton/CUDA images, python:*).
RUN if ! command -v python3 >/dev/null 2>&1; then \
        if command -v apt-get >/dev/null 2>&1; then \
            apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*; \
        elif command -v microdnf >/dev/null 2>&1; then microdnf install -y python3 python3-pip; \
        elif command -v dnf >/dev/null 2>&1; then dnf install -y python3 python3-pip; \
        elif command -v yum >/dev/null 2>&1; then yum install -y python3 python3-pip; \
        elif command -v apk >/dev/null 2>&1; then apk add --no-cache python3 py3-pip; \
        fi; \
    fi

{agent_layer}
WORKDIR /app
