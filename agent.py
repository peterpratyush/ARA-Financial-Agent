import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from state import AgentState
from tools import TOOL_REGISTRY
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

load_dotenv()

class AutonomousResearchAgent:
    def __init__(self):
        self.memory = MemorySaver()
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
        self.build_graph()

    def setup_node(self, state: AgentState) -> dict:
        tickers = state.get("tickers") or []
        print(f"DEBUG: Setup node identified tickers: {tickers}")
        return {"tickers": tickers, "current_plan": []}
    
    def review_node(self, state: AgentState) -> dict:
        print("\n--- Reviewing Report ---")
        # In a real app, this is where you'd pause for user input
        # For now, let's simulate the check:
        user_wants_more = input("Are you satisfied? (yes/no): ").lower()
        
        if user_wants_more == "no":
            follow_up = input("What else do you need to know? ")
            return {"current_plan": ["web_search"], "query": follow_up} # Loop back
        return {}

    def planner_node(self, state: AgentState) -> dict:
        # 1. Check for a new user query first
        new_query = state.get("query")
        
        # 2. Check for tickers to research
        researched = {f.get('ticker') for f in state.get("extracted_facts", [])}
        target_ticker = next((t for t in state.get("tickers", []) if t not in researched), None)
        
        # 3. If we have a plan, don't interfere
        if state.get("current_plan"): return {}

        # 4. Determine what to plan for
        if new_query:
            # Plan for a general follow-up query
            prompt = f"Create a research plan to answer this: '{new_query}'. Available tools: {list(TOOL_REGISTRY.keys())}. Return ONLY comma-separated list."
            ticker_context = "General Query"
        elif target_ticker:
            # Plan for the next ticker
            prompt = f"""
            Create a research plan for the company with ticker {target_ticker}.
            CRITICAL: Focus ONLY on company profile, financial performance, and market news.
            Available tools: {list(TOOL_REGISTRY.keys())}.
            Return ONLY a comma-separated list.
            """
            ticker_context = target_ticker
        else:
            return {} # Nothing to do

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.replace("*", "").replace("`", "").replace("{", "").replace("}", "")
            steps = [s.strip() for s in content.split(",") if s.strip() in TOOL_REGISTRY]
            
            # If it's a follow-up query, clear the 'query' field so we don't re-plan it
            return {"current_plan": steps, "ticker": ticker_context, "query": None}
        
        except Exception as e:
            print(f"DEBUG: Planner LLM call failed: {e}")
            return {}
    def execution_node(self, state: AgentState) -> dict:
        ticker = state.get("ticker")
        plan = state.get("current_plan", [])
        if not ticker or not plan: return {}

        tool_name = plan[0]
        tool = TOOL_REGISTRY.get(tool_name)
        
        # Smart mapping: Only pass the ticker if the tool expects it.
        # If the tool expects numbers (like DCF), we skip it or provide defaults.
        tool_input = {}
        for arg in tool.args:
            if arg == "ticker":
                tool_input[arg] = ticker
            elif arg in ["free_cash_flow", "growth_rate", "discount_rate"]:
                # Provide a realistic default or skip if missing
                tool_input[arg] = 0.05 
            elif arg == "years":
                tool_input[arg] = 5
            else:
                tool_input[arg] = f"Financial analysis for {ticker}"

        try:
            observation = tool.invoke(tool_input)
        except Exception as e:
            observation = f"Error: {str(e)}"

        new_fact = {"ticker": ticker, "tool": tool_name, "data": str(observation)}
        
        return {
            "extracted_facts": [new_fact],
            "executed_steps": [tool_name],
            "current_plan": plan[1:]
        }

    def synthesis_node(self, state: AgentState) -> dict:
        # Filter out errors from the facts before summarizing
        clean_facts = [f for f in state.get("extracted_facts", []) if "Error" not in f['data']]
        prompt = f"Summarize these successful research findings: {clean_facts}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"final_report": response.content}

    def route_after_execution(self, state: AgentState) -> str:
        if state.get("current_plan"): return "execute"
        researched = {f.get('ticker') for f in state.get("extracted_facts", [])}
        if any(t not in researched for t in state.get("tickers", [])): return "plan"
        return "synthesize"
    def route_after_synthesis(self, state: AgentState) -> str:
        # If the user added a new follow-up query, route back to planning
        if state.get("query"):
            return "plan"
        return END

    def build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("setup", self.setup_node)
        builder.add_node("plan", self.planner_node)
        builder.add_node("execute", self.execution_node)
        builder.add_node("synthesize", self.synthesis_node)
        
        builder.set_entry_point("setup")
        builder.add_edge("setup", "plan")
        builder.add_edge("plan", "execute")
        builder.add_conditional_edges("execute", self.route_after_execution)
        builder.add_edge("synthesize", END)
        self.graph = builder.compile(checkpointer=self.memory)

if __name__ == "__main__":
    agent = AutonomousResearchAgent()
    # Consistent thread_id keeps the memory of the conversation alive
    config = {"configurable": {"thread_id": "research_session_01"}}
    
    print("--- Financial Research Agent Started ---")
    
    # 1. Initial Research
    user_input = input("Enter tickers to research: ").upper()
    ticker_list = [t.strip() for t in user_input.split(",")]
    
    # Run the initial research
    print("--- Starting Initial Research ---")
    for event in agent.graph.stream({"tickers": ticker_list}, config=config):
        pass # Stream events silently to keep the chat clean
            
    # 2. Start the follow-up loop
    while True:
        # Get the latest report state
        state = agent.graph.get_state(config)
        print("\n📊 CURRENT REPORT:\n" + state.values.get("final_report", "No report yet."))
        
        # Ask for follow-up
        follow_up = input("\nDo you have a follow-up question? (or type 'quit' to exit): ")
        if follow_up.lower() == 'quit':
            break
            
        # Inject the follow-up query into the state and re-run the graph
        # We pass the follow_up into the 'query' field of the state
        print(f"--- Researching: {follow_up} ---")
        for event in agent.graph.stream({"query": follow_up}, config=config):
            pass 
            
    print("Session ended.")