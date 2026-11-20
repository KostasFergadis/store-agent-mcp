FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

WORKDIR /app
# Install exact versions from uv.lock first, so code changes don't redo this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home app
USER app

EXPOSE 8088
CMD ["uvicorn", "store_agent.approvals_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8088"]
