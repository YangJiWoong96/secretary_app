from dataclasses import dataclass, asdict
from typing import Optional
import logging


@dataclass
class PlannerLog:
    p_rag_calib: float
    p_web_calib: float
    prior_rag: float
    prior_web: float
    tau: float
    delta: float
    low_conf: float
    fast_margin: float
    raw_decision: dict
    amb: dict
    aux_router: Optional[str]
    final_decision: dict
    reason: str
    time_budget_ms: int


def log_planner(pl: PlannerLog):
    logging.getLogger("router").info("[planner] %s", asdict(pl))
