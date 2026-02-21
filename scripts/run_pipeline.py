#!/usr/bin/env python
"""Run the complete March Madness prediction pipeline."""

import asyncio
import argparse
import sys
import json
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
from src.tools.data_drift import detect_data_drift
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part

async def run_pipeline(train=True, detect_drift=False, generate_submission=True):
    """Run the agent pipeline."""
    
    print("🚀 Starting March Madness Prediction Pipeline...")
    print("="*60)
    
    # Create agents
    agents = []
    if train:
        agents.extend([
            create_data_loader_agent(),
            create_feature_engineer_agent(),
            create_model_trainer_agent(),
        ])
    if generate_submission:
        agents.append(create_submission_agent())
    
    # Create pipeline
    pipeline = SequentialAgent(
        name="MarchMadnessPipeline",
        sub_agents=agents,
        description="March Madness prediction pipeline"
    )
    
    # Run pipeline
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline,
        app_name="march_madness",
        session_service=session_service
    )
    
    session = await session_service.create_session(
        app_name="march_madness",
        user_id="cli_user",
        session_id="run_1"
    )
    
    async for event in runner.run_async(
        user_id="cli_user",
        session_id=session.id,
        new_message=Content(
            role="user", 
            parts=[Part(text="Run the March Madness prediction pipeline. Load data, compute features, train model, and generate submission.")]
        )
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"\n🤖 [{event.author}]")
            print("-"*40)
            # Extract text from parts
            text_parts = [part.text for part in event.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                print("\n".join(text_parts))
            else:
                print("[No text response]")
            print("="*60)
    
    # Run drift detection if requested
    if detect_drift:
        print("\n🔍 Running data drift detection...")
        print("-"*40)
        drift_result = detect_data_drift()
        print(json.dumps(drift_result, indent=2, ensure_ascii=False))
        print("="*60)
    
    print("\n✅ Pipeline complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run March Madness prediction pipeline')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--detect-drift', action='store_true', help='Run drift detection')
    parser.add_argument('--no-submission', action='store_true', help='Skip submission generation')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    
    args = parser.parse_args()
    
    # If --all is specified, run everything
    if args.all:
        args.train = True
        args.detect_drift = True
        args.no_submission = False
    
    asyncio.run(run_pipeline(
        train=args.train,
        detect_drift=args.detect_drift,
        generate_submission=not args.no_submission
    ))