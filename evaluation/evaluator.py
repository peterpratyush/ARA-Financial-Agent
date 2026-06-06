import json
import sys
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from experimental import ExperimentalResearchAgent # Ensure this imports your agent class


# Load environment variables
load_dotenv()

# Initialize the Judge using Groq
# 70b-versatile is the best model for "reasoning/judging" tasks
judge = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

def judge_report(query, report, expected_facts):
    """Uses Groq to evaluate the quality of your Agent's report."""
    prompt = f"""
    You are a Senior Financial Analyst Evaluator. Rate this report based on:
    1. Factual Accuracy: Are the numbers correct?
    2. Reliability Hierarchy: Did the agent prioritize SEC filings over news?
    3. Completeness: Did it answer: {query}?
    
    REPORT:
    {report}
    
    EXPECTED KEY FACTS:
    {expected_facts}
    
    Score 1-5 (where 5 is perfect) and provide a 1-sentence reasoning.
    Return output strictly in this JSON format:
    {{"score": 0, "reasoning": "your reasoning here"}}
    """
    response = judge.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)

def run_evaluation():
    agent = ExperimentalResearchAgent()
    
    # Load your 8 progressive benchmarks
    with open("evaluation/benchmarks.json", "r") as f:
        benchmarks = json.load(f)

    for case in benchmarks:
        print(f"\n--- Running Challenge {case['id']}: {case['difficulty']} ---")
        
        # 1. Define a configuration with a unique thread_id for this challenge
        # This tells LangGraph: "Save this conversation state under thread_id 'eval_{case_id}'"
        config = {"configurable": {"thread_id": f"eval_{case['id']}"}}
        
        # 2. Pass the config to the invoke method
        result = agent.graph.invoke({"tickers": [case['query']]}, config=config)
        
        report = result.get("final_report", "No report generated.")
        
        # 3. Evaluate...
        evaluation = judge_report(case['query'], report, case['expected_key_facts'])
        
        # ... rest of your code
        # Log result for your Trace Gallery
        print(f"Result Score: {evaluation['score']}/5")
        print(f"Feedback: {evaluation['reasoning']}")

if __name__ == "__main__":
    run_evaluation()