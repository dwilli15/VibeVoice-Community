#!/usr/bin/env python3
"""
VibeVoice Setup and Installation Helper
Helps users set up the environment and install dependencies
"""

import sys
import subprocess
import os
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue

class VibeVoiceSetup:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_widgets()
        self.check_system()
        
        # For handling async operations
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.process_queue()
        
    def setup_window(self):
        """Configure the setup window"""
        self.root.title("VibeVoice Setup Assistant")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (350)
        y = (self.root.winfo_screenheight() // 2) - (300)
        self.root.geometry(f"700x600+{x}+{y}")
        
    def create_widgets(self):
        """Create setup widgets"""
        # Header
        header_frame = tk.Frame(self.root, bg="#667eea", height=80)
        header_frame.pack(fill="x", pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="VibeVoice Setup Assistant", 
                              font=("Helvetica", 18, "bold"),
                              bg="#667eea", fg="white")
        title_label.pack(expand=True)
        
        # Main content
        main_frame = tk.Frame(self.root, padx=20, pady=10)
        main_frame.pack(fill="both", expand=True)
        
        # System check section
        check_frame = tk.LabelFrame(main_frame, text="System Requirements", 
                                   font=("Helvetica", 12, "bold"),
                                   padx=10, pady=10)
        check_frame.pack(fill="x", pady=(0, 10))
        
        self.check_text = scrolledtext.ScrolledText(check_frame, height=8, width=70,
                                                   font=("Courier", 10))
        self.check_text.pack(fill="both", expand=True)
        
        check_btn = tk.Button(check_frame, text="🔍 Recheck System",
                             command=self.check_system)
        check_btn.pack(pady=(5, 0))
        
        # Installation options
        install_frame = tk.LabelFrame(main_frame, text="Installation Options", 
                                     font=("Helvetica", 12, "bold"),
                                     padx=10, pady=10)
        install_frame.pack(fill="x", pady=(0, 10))
        
        # Basic dependencies
        basic_btn = tk.Button(install_frame, 
                             text="📦 Install Basic Dependencies",
                             bg="#4CAF50", fg="white",
                             font=("Helvetica", 11),
                             relief="flat", padx=20, pady=8,
                             command=self.install_basic_deps)
        basic_btn.pack(fill="x", pady=2)
        
        basic_desc = tk.Label(install_frame, 
                             text="PyTorch, audio libraries, GUI dependencies",
                             font=("Helvetica", 9), fg="gray")
        basic_desc.pack(anchor="w", padx=20)
        
        # Docker setup
        docker_btn = tk.Button(install_frame, 
                              text="🐳 Setup Docker Environment",
                              bg="#2196F3", fg="white",
                              font=("Helvetica", 11),
                              relief="flat", padx=20, pady=8,
                              command=self.setup_docker)
        docker_btn.pack(fill="x", pady=(10, 2))
        
        docker_desc = tk.Label(install_frame, 
                              text="Build Docker image and configure containers",
                              font=("Helvetica", 9), fg="gray")
        docker_desc.pack(anchor="w", padx=20)
        
        # CUDA setup
        cuda_btn = tk.Button(install_frame, 
                            text="⚡ Install CUDA Support",
                            bg="#FF9800", fg="white",
                            font=("Helvetica", 11),
                            relief="flat", padx=20, pady=8,
                            command=self.install_cuda)
        cuda_btn.pack(fill="x", pady=(10, 2))
        
        cuda_desc = tk.Label(install_frame, 
                            text="GPU acceleration for faster processing",
                            font=("Helvetica", 9), fg="gray")
        cuda_desc.pack(anchor="w", padx=20)
        
        # Progress section
        progress_frame = tk.LabelFrame(main_frame, text="Installation Progress", 
                                      font=("Helvetica", 12, "bold"),
                                      padx=10, pady=10)
        progress_frame.pack(fill="x", pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                           variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(fill="x", pady=(0, 5))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(progress_frame, textvariable=self.status_var,
                               font=("Helvetica", 10))
        status_label.pack()
        
        # Action buttons
        action_frame = tk.Frame(main_frame)
        action_frame.pack(fill="x", pady=10)
        
        launch_btn = tk.Button(action_frame, 
                              text="🚀 Launch VibeVoice",
                              bg="#9C27B0", fg="white",
                              font=("Helvetica", 12, "bold"),
                              relief="flat", padx=30, pady=10,
                              command=self.launch_vibevoice)
        launch_btn.pack(side="right")
        
        help_btn = tk.Button(action_frame, 
                            text="❓ Help",
                            command=self.show_help)
        help_btn.pack(side="left")
        
    def check_system(self):
        """Check system requirements"""
        def check_task():
            results = []
            
            # Check Python version
            python_version = sys.version_info
            if python_version >= (3, 8):
                results.append(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            else:
                results.append(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (Requires 3.8+)")
            
            # Check pip
            try:
                import pip
                results.append("✅ pip available")
            except ImportError:
                results.append("❌ pip not available")
            
            # Check PyTorch
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    results.append(f"✅ PyTorch with CUDA: {gpu_name}")
                else:
                    results.append("⚠️ PyTorch (CPU only)")
            except ImportError:
                results.append("❌ PyTorch not installed")
            
            # Check Docker
            try:
                result = subprocess.run(["docker", "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    results.append(f"✅ {version}")
                else:
                    results.append("❌ Docker not working")
            except:
                results.append("❌ Docker not installed")
            
            # Check GPU drivers
            try:
                result = subprocess.run(["nvidia-smi"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    results.append("✅ NVIDIA drivers available")
                else:
                    results.append("⚠️ NVIDIA drivers not detected")
            except:
                results.append("⚠️ nvidia-smi not available")
            
            # Check audio libraries
            audio_libs = ["soundfile", "librosa", "numpy"]
            for lib in audio_libs:
                try:
                    __import__(lib)
                    results.append(f"✅ {lib}")
                except ImportError:
                    results.append(f"❌ {lib} not installed")
            
            # Check GUI libraries
            try:
                import tkinter
                results.append("✅ tkinter (GUI support)")
            except ImportError:
                results.append("❌ tkinter not available")
            
            try:
                import gradio
                results.append("✅ gradio (Web interface)")
            except ImportError:
                results.append("❌ gradio not installed")
            
            self.result_queue.put(("system_check", "\\n".join(results)))
            
        threading.Thread(target=check_task, daemon=True).start()
        self.status_var.set("Checking system...")
        
    def install_basic_deps(self):
        """Install basic dependencies"""
        def install_task():
            packages = [
                "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "gradio",
                "soundfile",
                "librosa", 
                "numpy",
                "Pillow",
                "transformers",
                "accelerate"
            ]
            
            total_packages = len(packages)
            for i, package in enumerate(packages):
                self.result_queue.put(("progress", (i / total_packages) * 100))
                self.result_queue.put(("status", f"Installing {package.split()[0]}..."))
                
                try:
                    result = subprocess.run([sys.executable, "-m", "pip", "install"] + package.split(),
                                          capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        self.result_queue.put(("error", f"Failed to install {package}: {result.stderr}"))
                        return
                except subprocess.TimeoutExpired:
                    self.result_queue.put(("error", f"Installation of {package} timed out"))
                    return
                    
            self.result_queue.put(("progress", 100))
            self.result_queue.put(("status", "Basic dependencies installed successfully!"))
            
        threading.Thread(target=install_task, daemon=True).start()
        
    def setup_docker(self):
        """Setup Docker environment"""
        def docker_task():
            try:
                self.result_queue.put(("status", "Building Docker image..."))
                self.result_queue.put(("progress", 25))
                
                # Build Docker image
                result = subprocess.run(["docker", "build", "-t", "vibevoice-community", "."],
                                      cwd=Path(__file__).parent,
                                      capture_output=True, text=True, timeout=600)
                
                if result.returncode != 0:
                    self.result_queue.put(("error", f"Docker build failed: {result.stderr}"))
                    return
                    
                self.result_queue.put(("progress", 75))
                self.result_queue.put(("status", "Creating Docker Compose configuration..."))
                
                # Ensure docker-compose.yml exists
                compose_file = Path(__file__).parent / "docker-compose.yml"
                if not compose_file.exists():
                    self.result_queue.put(("error", "docker-compose.yml not found"))
                    return
                    
                self.result_queue.put(("progress", 100))
                self.result_queue.put(("status", "Docker environment ready!"))
                
            except subprocess.TimeoutExpired:
                self.result_queue.put(("error", "Docker build timed out"))
            except Exception as e:
                self.result_queue.put(("error", f"Docker setup failed: {e}"))
                
        threading.Thread(target=docker_task, daemon=True).start()
        
    def install_cuda(self):
        """Install CUDA support"""
        def cuda_task():
            try:
                self.result_queue.put(("status", "Installing CUDA PyTorch..."))
                self.result_queue.put(("progress", 25))
                
                # Install CUDA version of PyTorch
                cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                      "--index-url", "https://download.pytorch.org/whl/cu121"]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode != 0:
                    self.result_queue.put(("error", f"CUDA installation failed: {result.stderr}"))
                    return
                    
                self.result_queue.put(("progress", 75))
                self.result_queue.put(("status", "Verifying CUDA installation..."))
                
                # Verify CUDA works
                verify_code = """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
"""
                
                result = subprocess.run([sys.executable, "-c", verify_code],
                                      capture_output=True, text=True, timeout=30)
                
                self.result_queue.put(("progress", 100))
                if "CUDA available: True" in result.stdout:
                    self.result_queue.put(("status", "CUDA support installed successfully!"))
                else:
                    self.result_queue.put(("status", "CUDA installed but GPU not detected"))
                    
            except subprocess.TimeoutExpired:
                self.result_queue.put(("error", "CUDA installation timed out"))
            except Exception as e:
                self.result_queue.put(("error", f"CUDA installation failed: {e}"))
                
        threading.Thread(target=cuda_task, daemon=True).start()
        
    def launch_vibevoice(self):
        """Launch VibeVoice launcher"""
        try:
            launcher_path = Path(__file__).parent / "launcher.py"
            if launcher_path.exists():
                subprocess.Popen([sys.executable, str(launcher_path)])
                self.root.destroy()
            else:
                messagebox.showerror("Error", "Launcher not found. Please run setup first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch VibeVoice: {e}")
            
    def show_help(self):
        """Show help information"""
        help_text = """VibeVoice Setup Help

System Requirements:
- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM (optional but recommended)

Installation Steps:
1. Check system requirements
2. Install basic dependencies
3. Setup Docker environment (for web interface)
4. Install CUDA support (for GPU acceleration)
5. Launch VibeVoice

Troubleshooting:
- If installation fails, check your internet connection
- For GPU issues, ensure NVIDIA drivers are installed
- For Docker issues, install Docker Desktop first

For more help, visit: https://github.com/your-repo/VibeVoice-Community"""

        messagebox.showinfo("Help", help_text)
        
    def process_queue(self):
        """Process results from background threads"""
        try:
            while True:
                result_type, data = self.result_queue.get_nowait()
                
                if result_type == "system_check":
                    self.check_text.delete(1.0, tk.END)
                    self.check_text.insert(1.0, data)
                    self.status_var.set("System check complete")
                    
                elif result_type == "progress":
                    self.progress_var.set(data)
                    
                elif result_type == "status":
                    self.status_var.set(data)
                    
                elif result_type == "error":
                    self.status_var.set("Error occurred")
                    messagebox.showerror("Installation Error", data)
                    self.progress_var.set(0)
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self.process_queue)


def main():
    """Main function to run the setup"""
    root = tk.Tk()
    app = VibeVoiceSetup(root)
    
    def on_closing():
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
