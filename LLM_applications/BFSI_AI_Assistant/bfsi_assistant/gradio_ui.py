"""
Gradio UI for BFSI Call Center AI Assistant
Professional web interface with examples and metrics
"""

import gradio as gr
import time
from typing import Dict
import config
from orchestrator import BFSIOrchestrator

class GradioUI:
    """
    Gradio web interface for BFSI Assistant
    
    Features:
    - Clean, professional design
    - Example queries
    - Response time tracking
    - Tier visibility
    - Source citations
    """
    
    def __init__(self, orchestrator: BFSIOrchestrator):
        """
        Initialize the UI
        
        Args:
            orchestrator: Initialized BFSIOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.demo = None
    
    def query_with_metrics(self, question: str) -> str:
        """
        Process query and format response with metrics
        
        Args:
            question: User's question
            
        Returns:
            Formatted markdown response
        """
        if not question.strip():
            return "Please enter a question."
        
        # Track time
        start_time = time.time()
        
        # Get response
        result = self.orchestrator.query(question, return_metadata=False)
        
        response_time = time.time() - start_time
        
        # Format response
        answer = f"## 💬 Answer\n\n{result['answer']}\n\n"
        answer += "---\n\n"
        
        # Tier and confidence
        tier_name = result['tier'].replace('tier_', 'Tier ').upper()
        tier_emoji = {
            'tier_1': '⚡',
            'tier_2': '🤖',
            'tier_3': '📚'
        }.get(result['tier'], '❓')
        
        answer += f"**{tier_emoji} Tier Used:** {tier_name}\n\n"
        answer += f"**📊 Confidence:** {result.get('confidence', 'N/A').upper()}\n\n"
        answer += f"**⏱️ Response Time:** {response_time:.2f}s\n\n"
        
        # Sources
        if 'sources' in result and result['sources']:
            answer += "## 📚 Sources\n\n"
            for i, src in enumerate(result['sources'], 1):
                table_marker = " 📊" if src.get('contains_table') else ""
                answer += f"{i}. **{src['source']}** (Page {src['page']}){table_marker}\n"
            answer += "\n"
        
        # Matched query (for Tier 1)
        if 'matched_query' in result:
            answer += f"**🎯 Matched Query:** {result['matched_query']}\n\n"
        
        # Similarity score (for Tier 1)
        if 'similarity' in result:
            answer += f"**📈 Match Score:** {result['similarity']:.3f}\n"
        
        return answer
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=custom_css,
            title="BFSI Call Center AI"
        ) as self.demo:
            
            # Header
            gr.Markdown("""
            # 🏦 BFSI Call Center AI Assistant
            
            Ask questions about banking, loans, credit cards, insurance, and financial services.
            
            **System Features:**
            - ⚡ **Tier 1:** Instant responses from curated dataset (70% queries)
            - 🤖 **Tier 2:** AI-generated responses (20% queries)
            - 📚 **Tier 3:** Document retrieval with RAG (10% complex queries)
            """)
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the interest rate for personal loans?",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button(
                            "Submit",
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "Clear",
                            size="lg"
                        )
                
                with gr.Column(scale=3):
                    answer_output = gr.Markdown(
                        label="Response",
                        value="*Your answer will appear here...*"
                    )
            
            # Examples
            gr.Markdown("## 💡 Example Questions")
            
            gr.Examples(
                examples=[
                    ["What is the interest rate for personal loans?"],
                    ["How do I check my loan application status?"],
                    ["What documents are needed for a home loan?"],
                    ["What is the LTV ratio for home loans above 75 lakhs?"],
                    ["Can I prepay my loan without penalty?"],
                    ["What are the processing fees for different loan types?"],
                    ["How is EMI calculated?"],
                    ["I want to cancel my credit card"],
                    ["What are the charges for late payment?"],
                    ["Tell me about savings account interest rates"],
                ],
                inputs=question_input,
            )
            
            # Footer with stats
            gr.Markdown("""
            ---
            
            ### 📊 System Information
            
            **Dataset:** 524 BFSI conversation samples  
            **Model:** Phi-3.5-mini fine-tuned with LoRA  
            **Knowledge Base:** 11 PDF documents with 19 tables  
            **Search:** Hybrid retrieval with reranking
            
            ---
            
            *Built for BFSI Call Center Operations*
            """)
            
            # Event handlers
            submit_btn.click(
                fn=self.query_with_metrics,
                inputs=question_input,
                outputs=answer_output
            )
            
            clear_btn.click(
                fn=lambda: ("", "*Your answer will appear here...*"),
                inputs=None,
                outputs=[question_input, answer_output]
            )
            
            # Enter key support
            question_input.submit(
                fn=self.query_with_metrics,
                inputs=question_input,
                outputs=answer_output
            )
    
    def launch(
        self,
        share: bool = None,
        debug: bool = None,
        server_port: int = None
    ):
        """
        Launch the Gradio interface
        
        Args:
            share: Create public link (default from config)
            debug: Enable debug mode (default from config)
            server_port: Port number (default from config)
        """
        share = share if share is not None else config.GRADIO_SHARE
        debug = debug if debug is not None else config.GRADIO_DEBUG
        server_port = server_port or config.GRADIO_PORT
        
        if self.demo is None:
            self.create_interface()
        
        print("\n" + "="*60)
        print("🚀 LAUNCHING GRADIO INTERFACE")
        print("="*60)
        print(f"Share: {share}")
        print(f"Debug: {debug}")
        print(f"Port: {server_port}")
        print("="*60)
        
        self.demo.launch(
            share=share,
            debug=debug,
            server_port=server_port,
            server_name="0.0.0.0"  # Allow external access
        )

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING GRADIO UI MODULE")
    print("="*60)
    
    # Initialize orchestrator
    print("\nInitializing orchestrator...")
    orchestrator = BFSIOrchestrator(verbose=False)
    orchestrator.initialize()
    
    # Create and launch UI
    print("\nCreating UI...")
    ui = GradioUI(orchestrator)
    
    print("\nLaunching interface...")
    ui.launch()
