#!/usr/bin/env python3
"""
Minimal test to isolate GUI issues
"""

import gradio as gr
import sys
import traceback

def test_minimal_gui():
    """Test minimal GUI functionality"""
    try:
        print("Testing minimal GUI...")

        with gr.Blocks(title="Test GUI") as interface:
            gr.Markdown("# Minimal Test")

            # Simple components
            text_input = gr.Textbox(label="Input")
            text_output = gr.Textbox(label="Output")
            btn = gr.Button("Test")

            def test_function(input_text):
                return f"Processed: {input_text}"

            btn.click(
                fn=test_function,
                inputs=[text_input],
                outputs=[text_output]
            )

        print("✓ Minimal GUI created successfully")
        return interface

    except Exception as e:
        print(f"✗ Minimal GUI failed: {e}")
        traceback.print_exc()
        return None

def test_ebook_gui_import():
    """Test importing the main GUI"""
    try:
        print("Testing ebook_gui import...")
        from ebook_gui import EbookConverterGUI

        print("✓ Import successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_ebook_gui_creation():
    """Test creating the GUI instance"""
    try:
        print("Testing GUI instance creation...")
        from ebook_gui import EbookConverterGUI

        gui = EbookConverterGUI()
        print("✓ GUI instance created")

        # Test interface creation
        print("Testing interface creation...")
        interface = gui.create_interface()
        print("✓ Interface created successfully")

        return True
    except Exception as e:
        print(f"✗ GUI creation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== GUI DIAGNOSTIC TEST ===")

    # Test 1: Minimal Gradio
    test_minimal_gui()

    # Test 2: Import
    if test_ebook_gui_import():
        # Test 3: Full GUI creation
        test_ebook_gui_creation()

    print("=== DIAGNOSTIC COMPLETE ===")
