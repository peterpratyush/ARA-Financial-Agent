import operator
from typing import Annotated, List, TypedDict

class AgentState(TypedDict):
    tickers: List[str]
    ticker: str
    current_plan: List[str]
    extracted_facts: Annotated[List[dict], operator.add]
    executed_steps: Annotated[List[str], operator.add]
    final_report: str