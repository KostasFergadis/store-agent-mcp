import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    agent_id: str
    approver_api_key: str
    embedder: str
    high_risk_threshold: float


def get_settings() -> Settings:
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "postgresql://store:store@localhost:5544/store"),
        agent_id=os.environ.get("AGENT_ID", "claude-agent"),
        approver_api_key=os.environ.get("APPROVER_API_KEY", "dev-approver-key"),
        embedder=os.environ.get("EMBEDDER", "fastembed"),
        high_risk_threshold=float(os.environ.get("HIGH_RISK_THRESHOLD", "0.30")),
    )
