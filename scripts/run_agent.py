#!/usr/bin/env python
"""Run a single agent for GitHub Actions."""

import asyncio
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.data_loader_agent import create_data_loader_agent
from src.agents.feature_engineer_agent import create_feature_engineer_agent
from src.agents.model_trainer_agent import create_model_trainer_agent
from src.agents.submission_agent import create_submission_agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part

AGENTS = {
    "data_loader": create_data_loader_agent,
    "feature_engineer": create_feature_engineer_agent,
    "model_trainer": create_model_trainer_agent,
    "submission": create_submission_agent,
}

MESSAGES = {
    "data_loader": "Load the March Madness competition data and give me a summary.",
    "feature_engineer": "Compute Elo ratings for all teams based on the loaded data.",
    "model_trainer": "Train a prediction model using Elo differences and seed differences.",
    "submission": "Generate the submission file with predictions for all possible matchups.",
}

async def run_agent(agent_name):
    """Run a single agent."""
    
    if agent_name not in AGENTS:
        print(f"❌ Unknown agent: {agent_name}")
        print(f"Available agents: {list(AGENTS.keys())}")
        return 1
    
    print(f"🚀 Running {agent_name}...")
    
    # Create agent
    agent = AGENTS[agent_name]()
    
    # Run agent
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="march_madness",
        session_service=session_service
    )
    
    session = await session_service.create_session(
        app_name="march_madness",
        user_id="github_action",
        session_id="run_1"
    )
    
    async for event in runner.run_async(
        user_id="github_action",
        session_id=session.id,
        new_message=Content(role="user", parts=[Part(text=MESSAGES[agent_name])])
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    print(part.text)
    
    print(f"✅ {agent_name} completed successfully")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a single agent')
    parser.add_argument('agent', choices=AGENTS.keys(), help='Agent to run')
    
    args = parser.parse_args()
    
    exit_code = asyncio.run(run_agent(args.agent))
    sys.exit(exit_code)