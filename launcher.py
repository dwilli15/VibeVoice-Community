#!/usr/bin/env python3
"""
VibeVoice Launcher - Central launcher for all VibeVoice interfaces
Provides options to launch desktop GUI, web interface, or manage containers
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
import webbrowser
from pathlib import Path
import socket

class VibeVoiceLauncher:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_widgets()
        self.check_services()
        
    def setup_window(self):
        """Configure the launcher window"""
        self.root.title("VibeVoice Launcher")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (250)
        y = (self.root.winfo_screenheight() // 2) - (200)
        self.root.geometry(f"500x400+{x}+{y}")
        
    def create_widgets(self):
        """Create launcher widgets"""
        # Header
        header_frame = tk.Frame(self.root, bg="#667eea", height=80)
        header_frame.pack(fill="x", pady=(0, 20))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="VibeVoice", 
                              font=("Helvetica", 20, "bold"),
                              bg="#667eea", fg="white")
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="Voice Synthesis Platform", 
                                 font=("Helvetica", 12),
                                 bg="#667eea", fg="white")
        subtitle_label.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, padx=30, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Interface selection
        interface_frame = tk.LabelFrame(main_frame, text="Choose Interface", 
                                       font=("Helvetica", 12, "bold"),
                                       padx=10, pady=10)
        interface_frame.pack(fill="x", pady=(0, 20))
        
        # Desktop GUI button
        desktop_btn = tk.Button(interface_frame, 
                               text="🖥️ Desktop Application",
                               font=("Helvetica", 11),
                               bg="#4CAF50", fg="white",
                               relief="flat", padx=20, pady=10,
                               command=self.launch_desktop_gui)
        desktop_btn.pack(fill="x", pady=5)
        
        desktop_desc = tk.Label(interface_frame, 
                               text="Native desktop interface with advanced features",
                               font=("Helvetica", 9),
                               fg="gray")
        desktop_desc.pack()
        
        # Web interface button
        web_btn = tk.Button(interface_frame, 
                           text="🌐 Web Interface",
                           font=("Helvetica", 11),
                           bg="#2196F3", fg="white",
                           relief="flat", padx=20, pady=10,
                           command=self.launch_web_interface)
        web_btn.pack(fill="x", pady=(15, 5))
        
        web_desc = tk.Label(interface_frame, 
                           text="Browser-based interface with real-time streaming",
                           font=("Helvetica", 9),
                           fg="gray")
        web_desc.pack()
        
        # Container management
        container_frame = tk.LabelFrame(main_frame, text="Container Management", 
                                       font=("Helvetica", 12, "bold"),
                                       padx=10, pady=10)
        container_frame.pack(fill="x", pady=(0, 20))
        
        # Container buttons frame
        btn_frame = tk.Frame(container_frame)
        btn_frame.pack(fill="x")
        
        start_btn = tk.Button(btn_frame, 
                             text="▶️ Start",
                             bg="#FF9800", fg="white",
                             relief="flat", padx=15, pady=5,
                             command=self.start_container)
        start_btn.pack(side="left", padx=(0, 5))
        
        stop_btn = tk.Button(btn_frame, 
                            text="⏹️ Stop",
                            bg="#F44336", fg="white",
                            relief="flat", padx=15, pady=5,
                            command=self.stop_container)
        stop_btn.pack(side="left", padx=5)
        
        restart_btn = tk.Button(btn_frame, 
                               text="🔄 Restart",
                               bg="#9C27B0", fg="white",
                               relief="flat", padx=15, pady=5,
                               command=self.restart_container)
        restart_btn.pack(side="left", padx=5)
        
        status_btn = tk.Button(btn_frame, 
                              text="📊 Status",
                              bg="#607D8B", fg="white",
                              relief="flat", padx=15, pady=5,
                              command=self.check_container_status)
        status_btn.pack(side="left", padx=5)
        
        # Status display
        self.status_text = tk.StringVar(value="Checking services...")
        status_label = tk.Label(main_frame, textvariable=self.status_text,
                               font=("Helvetica", 10),
                               fg="gray")
        status_label.pack(pady=10)
        
        # Quick actions
        actions_frame = tk.Frame(main_frame)
        actions_frame.pack(fill="x")
        
        folder_btn = tk.Button(actions_frame, 
                              text="📁 Output Folder",
                              command=self.open_output_folder)
        folder_btn.pack(side="left")
        
        logs_btn = tk.Button(actions_frame, 
                            text="📝 View Logs",
                            command=self.view_logs)
        logs_btn.pack(side="left", padx=(10, 0))
        
        help_btn = tk.Button(actions_frame, 
                            text="❓ Help",
                            command=self.show_help)
        help_btn.pack(side="right")
        
    def check_services(self):
        """Check status of services"""
        def check_task():
            # Check if web interface is running
            web_running = self.check_port(7860)
            
            # Check if Docker is available
            docker_available = self.check_docker()
            
            # Update status
            if web_running:
                status = "🟢 Web interface running on port 7860"
            elif docker_available:
                status = "🟡 Docker available, web interface stopped"
            else:
                status = "🔴 Docker not available"
                
            self.status_text.set(status)
            
        threading.Thread(target=check_task, daemon=True).start()
        
    def check_port(self, port):
        """Check if a port is in use"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result == 0
        except:
            return False
            
    def check_docker(self):
        """Check if Docker is available"""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
            
    def launch_desktop_gui(self):
        """Launch the desktop GUI"""
        try:
            script_path = Path(__file__).parent / "desktop_gui.py"
            subprocess.Popen([sys.executable, str(script_path)])
            messagebox.showinfo("Launched", "Desktop GUI started successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch desktop GUI: {e}")
            
    def launch_web_interface(self):
        """Launch or open the web interface"""
        if self.check_port(7860):
            # Web interface is already running
            webbrowser.open("http://localhost:7860")
            messagebox.showinfo("Opened", "Web interface opened in browser!")
        else:
            # Try to start the container
            if self.check_docker():
                self.start_container()
                messagebox.showinfo("Starting", 
                                   "Starting web interface... Please wait a moment then try again.")
            else:
                messagebox.showerror("Error", 
                                   "Docker not available. Please install Docker to use web interface.")
                
    def start_container(self):
        """Start the Docker container"""
        def start_task():
            try:
                # Use PowerShell script if available
                script_path = Path(__file__).parent / "manage-container.ps1"
                if script_path.exists():
                    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass",
                                  "-File", str(script_path), "start"], 
                                 check=True, timeout=300)
                else:
                    # Fallback to direct docker command
                    subprocess.run(["docker-compose", "up", "-d", "--build"], 
                                 check=True, timeout=300)
                                 
                self.status_text.set("🟢 Container started successfully")
                
            except subprocess.TimeoutExpired:
                self.status_text.set("⚠️ Container start timed out")
            except Exception as e:
                self.status_text.set(f"🔴 Start failed: {e}")
                
        threading.Thread(target=start_task, daemon=True).start()
        self.status_text.set("🟡 Starting container...")
        
    def stop_container(self):
        """Stop the Docker container"""
        def stop_task():
            try:
                script_path = Path(__file__).parent / "manage-container.ps1"
                if script_path.exists():
                    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass",
                                  "-File", str(script_path), "stop"], 
                                 check=True, timeout=60)
                else:
                    subprocess.run(["docker-compose", "down"], 
                                 check=True, timeout=60)
                                 
                self.status_text.set("🔴 Container stopped")
                
            except Exception as e:
                self.status_text.set(f"⚠️ Stop failed: {e}")
                
        threading.Thread(target=stop_task, daemon=True).start()
        self.status_text.set("🟡 Stopping container...")
        
    def restart_container(self):
        """Restart the Docker container"""
        def restart_task():
            try:
                script_path = Path(__file__).parent / "manage-container.ps1"
                if script_path.exists():
                    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass",
                                  "-File", str(script_path), "restart"], 
                                 check=True, timeout=300)
                else:
                    subprocess.run(["docker-compose", "down"], timeout=60)
                    subprocess.run(["docker-compose", "up", "-d", "--build"], 
                                 timeout=300)
                                 
                self.status_text.set("🟢 Container restarted")
                
            except Exception as e:
                self.status_text.set(f"⚠️ Restart failed: {e}")
                
        threading.Thread(target=restart_task, daemon=True).start()
        self.status_text.set("🟡 Restarting container...")
        
    def check_container_status(self):
        """Check and display container status"""
        def status_task():
            try:
                result = subprocess.run(["docker", "ps", "-a", "--filter", "name=vibe"], 
                                      capture_output=True, text=True, timeout=10)
                
                if "vibe" in result.stdout:
                    if "Up" in result.stdout:
                        self.status_text.set("🟢 Container running")
                    else:
                        self.status_text.set("🔴 Container stopped")
                else:
                    self.status_text.set("⚪ Container not found")
                    
            except Exception as e:
                self.status_text.set(f"⚠️ Status check failed: {e}")
                
        threading.Thread(target=status_task, daemon=True).start()
        
    def view_logs(self):
        """View container logs"""
        try:
            if sys.platform == "win32":
                subprocess.Popen(["powershell", "-Command", 
                                "docker logs -f vibe"])
            else:
                subprocess.Popen(["gnome-terminal", "--", "docker", "logs", "-f", "vibe"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open logs: {e}")
            
    def open_output_folder(self):
        """Open the output folder"""
        output_path = Path.home() / "VibeVoice_Output"
        output_path.mkdir(exist_ok=True)
        
        try:
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(output_path)])
            else:
                subprocess.run(["xdg-open", str(output_path)])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {e}")
            
    def show_help(self):
        """Show help information"""
        help_text = """VibeVoice Help

🖥️ Desktop Application:
- Native GUI with advanced features
- Batch processing capabilities
- Voice management tools

🌐 Web Interface:
- Browser-based interface
- Real-time streaming
- Public sharing options

🐳 Container Management:
- Start: Launch web interface in Docker
- Stop: Stop the Docker container
- Restart: Rebuild and restart container
- Status: Check container status

📁 Output Folder:
Generated audio files are saved here

📝 Logs:
View Docker container logs for debugging

For more help, visit the GitHub repository."""

        messagebox.showinfo("Help", help_text)


def main():
    """Main function to run the launcher"""
    root = tk.Tk()
    app = VibeVoiceLauncher(root)
    
    def on_closing():
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
