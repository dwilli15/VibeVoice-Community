"""
Clean Startup Script for VibeVoice GUI
Ensures ports are clean before launching the interface
"""

import subprocess
import sys
import time
import os
import codecs
from pathlib import Path

# Fix Unicode encoding issues on Windows
if os.name == 'nt':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def run_port_cleanup(port: int = 7862):
    """Clean up any stuck processes on the specified port"""
    try:
        print(f"[CLEAN] Cleaning up port {port}...")
        
        # Use our port manager to clean up
        script_dir = Path(__file__).parent
        port_manager_path = script_dir / "port_manager.py"
        
        result = subprocess.run([
            sys.executable, str(port_manager_path), 
            "--port", str(port), 
            "--kill", 
            "--force"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[OK] Port {port} cleanup completed")
            if result.stdout.strip():
                print(result.stdout)
        else:
            print(f"[WARN] Port cleanup had issues: {result.stderr}")
        
        # Wait a moment for cleanup to settle
        time.sleep(2)
        
    except Exception as e:
        print(f"[WARN] Port cleanup error: {e}")

def start_gui(port: int = 7862, force_kill: bool = True):
    """Start the GUI with clean port"""
    
    if force_kill:
        run_port_cleanup(port)
    
    print(f"[START] Starting VibeVoice GUI on port {port}...")
    
    try:
        # Build command
        script_dir = Path(__file__).parent
        gui_script = script_dir / "ebook_gui.py"
        
        gui_host = os.environ.get("VIBEVOICE_GUI_HOST", "127.0.0.1")
        cmd = [sys.executable, str(gui_script), "--port", str(port), "--host", gui_host]
        if force_kill:
            cmd.append("--force-kill")
        
        print(f"[INFO] Binding GUI to host {gui_host}")
        
        # Start the GUI
        # Ensure UTF-8 environment for child process to avoid cp1252 decode errors
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # Force UTF-8 in Gradio/child regardless of user console codepage
        env.setdefault("LC_ALL", "C.UTF-8")  # harmless on Windows if ignored
        # Encourage more stable/faster CUDA allocator behavior
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:128")

        # Ensure Hugging Face cache points at the shared models directory so downloads are reused
        default_hf_home = Path("D:/omen/models/hf/hub")
        if default_hf_home.exists():
            env.setdefault("HF_HOME", str(default_hf_home))
            env.setdefault("HUGGINGFACE_HUB_CACHE", str(default_hf_home))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env
        )
        
        print(f"[INFO] GUI starting with PID {process.pid}")
        print("[INFO] The web interface should open automatically")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Stream output
        try:
            import io
            # Wrap raw stream to decode as UTF-8 with replacement for any stray bytes
            if process.stdout is not None:
                stream = io.TextIOWrapper(process.stdout, encoding='utf-8', errors='replace', newline='')
                for line in stream:
                    print(line.rstrip())
        except KeyboardInterrupt:
            print("\n[STOP] Stopping GUI...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[FORCE] Force killing GUI process...")
                process.kill()
        
        return process.returncode
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to start GUI: {e}")
        print("[TRACEBACK]" )
        traceback.print_exc()
        return 1

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean startup for VibeVoice GUI")
    parser.add_argument("--port", type=int, default=7862, help="Port to use")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip port cleanup")
    parser.add_argument("--check-only", action="store_true", help="Only check port status")
    
    args = parser.parse_args()
    
    if args.check_only:
        # Just check port status
        try:
            script_dir = Path(__file__).parent
            port_manager_path = script_dir / "port_manager.py"
            
            result = subprocess.run([
                sys.executable, str(port_manager_path), 
                "--port", str(args.port)
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return result.returncode
        except Exception as e:
            print(f"[ERROR] Port check failed: {e}")
            return 1
    
    # Start GUI with cleanup
    return start_gui(args.port, not args.no_cleanup)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
