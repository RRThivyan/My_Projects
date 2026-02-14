"""
Main Entry Point for BFSI Call Center AI Assistant
Run this file to start the complete system
"""

import sys
import config
from orchestrator import BFSIOrchestrator, ask
from gradio_ui import GradioUI

def main(mode: str = "ui"):
    """
    Main function to run the BFSI Assistant
    
    Args:
        mode: 'ui' for Gradio interface, 'cli' for command line
    """
    print("="*60)
    print("BFSI CALL CENTER AI ASSISTANT")
    print("="*60)
    
    # Validate configuration
    print("\n📋 Validating configuration...")
    if not config.validate_config():
        print("\n❌ Configuration validation failed!")
        print("Please update paths in config.py")
        sys.exit(1)
    
    # Print configuration
    config.print_config()
    
    # Initialize orchestrator
    print("\n🚀 Initializing system...")
    orchestrator = BFSIOrchestrator(verbose=False)
    orchestrator.initialize()
    
    print("\n" + "="*60)
    print("✅ SYSTEM READY")
    print("="*60)
    
    # Launch based on mode
    if mode == "ui":
        # Gradio UI mode
        print("\n🌐 Starting Gradio UI...")
        ui = GradioUI(orchestrator)
        ui.launch()
        
    elif mode == "cli":
        # Command line mode
        print("\n💬 Command Line Mode")
        print("="*60)
        print("Type your questions (or 'quit' to exit)")
        print("="*60)
        
        while True:
            try:
                query = input("\n🔍 You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    orchestrator.print_stats()
                    break
                
                if not query:
                    continue
                
                # Get response
                ask(orchestrator, query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                orchestrator.print_stats()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    else:
        print(f"❌ Invalid mode: {mode}")
        print("Use 'ui' or 'cli'")
        sys.exit(1)

# ============================================================
# QUICK START FUNCTIONS
# ============================================================

def quick_test():
    """Run quick test queries"""
    print("="*60)
    print("QUICK TEST MODE")
    print("="*60)
    
    # Validate config
    if not config.validate_config():
        print("❌ Configuration validation failed!")
        sys.exit(1)
    
    # Initialize
    orchestrator = BFSIOrchestrator(verbose=True)
    orchestrator.initialize()
    
    # Test queries
    test_queries = [
        "What is the interest rate for personal loans?",
        "How do I check my loan eligibility?",
        "What is the LTV ratio for home loans?",
    ]
    
    print("\n" + "="*60)
    print("RUNNING TEST QUERIES")
    print("="*60)
    
    for query in test_queries:
        ask(orchestrator, query)
    
    # Print stats
    orchestrator.print_stats()

def setup_vector_db():
    """Create or recreate vector database"""
    print("="*60)
    print("VECTOR DATABASE SETUP")
    print("="*60)
    
    # Validate paths
    if not config.validate_config():
        print("❌ Configuration validation failed!")
        sys.exit(1)
    
    # Create database
    from vector_db_creator import VectorDBCreator
    
    creator = VectorDBCreator()
    stats = creator.create_database()
    
    print("\n✅ Database created successfully!")
    print("\n📊 Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BFSI Call Center AI Assistant"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ui", "cli", "test", "setup-db"],
        default="ui",
        help="Run mode: ui (Gradio), cli (command line), test (quick test), setup-db (create vector DB)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        quick_test()
    elif args.mode == "setup-db":
        setup_vector_db()
    else:
        main(mode=args.mode)
