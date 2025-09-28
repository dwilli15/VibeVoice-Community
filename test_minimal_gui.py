"""
Minimal GUI test to isolate the JSON error
"""

import gradio as gr
import json

def test_analyze():
    """Test function that returns valid JSON"""
    return "Test analysis", json.dumps({"test": "data"})

def create_minimal_interface():
    with gr.Blocks() as interface:
        with gr.Tab("Test"):
            analyze_btn = gr.Button("Analyze")
            status_output = gr.Markdown("Ready")
            json_output = gr.JSON(value={"status": "ready"})
            
            analyze_btn.click(
                fn=test_analyze,
                outputs=[status_output, json_output]
            )
    
    return interface

if __name__ == "__main__":
    interface = create_minimal_interface()
    interface.launch(server_port=7869, share=False)
