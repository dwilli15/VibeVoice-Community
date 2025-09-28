#!/usr/bin/env python3
"""
VibeVoice Desktop GUI - Modern tkinter-based interface for VibeVoice
Features: File management, batch processing, voice cloning, real-time preview
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
from datetime import datetime
import webbrowser

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from tts_backend import get_tts_backend, TTSRequest, Voice, TTSEngine
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    import torch
    import soundfile as sf
    import numpy as np
    TTS_BACKEND_AVAILABLE = True
    VIBEVOICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TTS modules not available: {e}")
    TTS_BACKEND_AVAILABLE = False
    VIBEVOICE_AVAILABLE = False

class VibeVoiceGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_variables()
        self.setup_styles()
        self.create_widgets()
        self.setup_bindings()
        
        # Threading for background tasks
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        
        # VibeVoice model
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Multi-model TTS backend
        self.tts_backend = None
        if TTS_BACKEND_AVAILABLE:
            self.tts_backend = get_tts_backend()
            
        # Available voices from all engines
        self.available_voices = {}
        self.selected_voice = None
        
        # Check for updates periodically
        self.root.after(100, self.process_queue)
        
    def setup_window(self):
        """Configure the main window"""
        self.root.title("VibeVoice Desktop - Voice Synthesis Studio")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set window icon (if available)
        try:
            icon_path = Path(__file__).parent / "Figures" / "VibeVoice_logo.png"
            if icon_path.exists():
                photo = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, photo)
        except:
            pass
            
        # Center window on screen
        self.center_window()
        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
    def setup_variables(self):
        """Initialize tkinter variables"""
        self.model_path = tk.StringVar(value="microsoft/VibeVoice-1.5B")
        self.output_dir = tk.StringVar(value=str(Path.home() / "VibeVoice_Output"))
        self.speaker_count = tk.IntVar(value=1)
        self.inference_steps = tk.IntVar(value=10)
        self.temperature = tk.DoubleVar(value=0.7)
        self.top_p = tk.DoubleVar(value=0.9)
        self.speed_factor = tk.DoubleVar(value=1.0)
        self.model_loaded = tk.BooleanVar(value=False)
        self.generation_progress = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="Ready")
        
        # Voice settings
        self.selected_voices = []
        self.voice_mappings = {}
        
    def setup_styles(self):
        """Configure custom styles"""
        style = ttk.Style()
        
        # Configure modern theme
        style.theme_use('clam')
        
        # Custom colors
        primary_color = "#667eea"
        secondary_color = "#764ba2"
        accent_color = "#f093fb"
        bg_color = "#f8fafc"
        
        # Configure button styles
        style.configure("Accent.TButton",
                       background=primary_color,
                       foreground="white",
                       focuscolor="none",
                       borderwidth=0,
                       relief="flat")
        
        style.map("Accent.TButton",
                 background=[('active', secondary_color),
                           ('pressed', accent_color)])
                           
        # Configure frame styles
        style.configure("Card.TFrame",
                       background="white",
                       relief="solid",
                       borderwidth=1)
                       
    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Create main sections
        self.create_header(main_frame)
        self.create_model_section(main_frame)
        self.create_main_content(main_frame)
        self.create_status_bar(main_frame)
        
    def create_header(self, parent):
        """Create the header section"""
        header_frame = ttk.Frame(parent, style="Card.TFrame", padding="15")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="VibeVoice Desktop",
                               font=("Helvetica", 24, "bold"))
        title_label.grid(row=0, column=0, sticky="w")
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, text="Professional Voice Synthesis Studio",
                                  font=("Helvetica", 12))
        subtitle_label.grid(row=1, column=0, sticky="w")
        
        # Action buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.grid(row=0, column=1, rowspan=2, sticky="e")
        
        ttk.Button(button_frame, text="🌐 Web Interface", 
                  command=self.open_web_interface).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="📁 Open Output", 
                  command=self.open_output_folder).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="⚙️ Settings", 
                  command=self.open_settings).pack(side="right", padx=(5, 0))
                  
        header_frame.columnconfigure(0, weight=1)
        
    def create_model_section(self, parent):
        """Create the model loading section"""
        model_frame = ttk.LabelFrame(parent, text="Model Configuration", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Model path
        ttk.Label(model_frame, text="TTS Engine:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        # Engine selection
        self.engine_var = tk.StringVar(value="Auto-Select")
        engine_combo = ttk.Combobox(model_frame, textvariable=self.engine_var, 
                                   values=["Auto-Select", "VibeVoice", "Coqui AI"], 
                                   state="readonly", width=15)
        engine_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=2, sticky="w", padx=(10, 5))
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=40)
        model_entry.grid(row=0, column=3, sticky="ew", padx=(0, 5))
        ttk.Button(model_frame, text="Browse", 
                  command=self.browse_model).grid(row=0, column=4)
        
        # Load/Unload buttons
        button_frame = ttk.Frame(model_frame)
        button_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))
        
        self.load_button = ttk.Button(button_frame, text="🔄 Discover Voices", 
                                     style="Accent.TButton",
                                     command=self.discover_voices)
        self.load_button.pack(side="left", padx=(0, 5))
        
        self.benchmark_button = ttk.Button(button_frame, text="⚡ Benchmark Voices", 
                                          command=self.benchmark_voices)
        self.benchmark_button.pack(side="left", padx=(0, 5))
        
        # Engine status
        self.engine_status_label = ttk.Label(button_frame, text="Ready to discover voices", 
                                           foreground="blue")
        self.engine_status_label.pack(side="left", padx=(10, 0))
        
        model_frame.columnconfigure(3, weight=1)
        
    def create_main_content(self, parent):
        """Create the main content area with notebook tabs"""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        
        # Text Generation Tab
        self.create_text_tab()
        
        # Batch Processing Tab
        self.create_batch_tab()
        
        # Voice Management Tab
        self.create_voice_tab()
        
        # Settings Tab
        self.create_settings_tab()
        
    def create_text_tab(self):
        """Create the text generation tab"""
        text_frame = ttk.Frame(self.notebook)
        self.notebook.add(text_frame, text="Text Generation")
        
        # Configure grid
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        
        # Input section
        input_frame = ttk.LabelFrame(text_frame, text="Script Input", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        # Text input
        self.text_input = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD,
                                                   font=("Courier", 11))
        self.text_input.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        # Load and save buttons
        ttk.Button(input_frame, text="Load Script", 
                  command=self.load_script).grid(row=1, column=0, sticky="w")
        ttk.Button(input_frame, text="Save Script", 
                  command=self.save_script).grid(row=1, column=1, padx=(5, 0))
        ttk.Button(input_frame, text="Clear", 
                  command=self.clear_script).grid(row=1, column=2, sticky="e")
        
        # Generation controls
        control_frame = ttk.LabelFrame(text_frame, text="Voice Selection & Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        control_frame.columnconfigure(1, weight=1)
        
        # Voice selection with detailed info
        ttk.Label(control_frame, text="Available Voices:").grid(row=0, column=0, sticky="nw", pady=(5, 0))
        
        # Voice tree with details
        voice_tree_frame = ttk.Frame(control_frame)
        voice_tree_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 10))
        voice_tree_frame.columnconfigure(0, weight=1)
        
        self.voice_tree = ttk.Treeview(voice_tree_frame, 
                                      columns=("Engine", "Language", "Gender", "Status"), 
                                      show="tree headings", height=6)
        self.voice_tree.grid(row=0, column=0, sticky="ew")
        
        # Voice tree headings
        self.voice_tree.heading("#0", text="Voice Name")
        self.voice_tree.heading("Engine", text="Engine")
        self.voice_tree.heading("Language", text="Language")
        self.voice_tree.heading("Gender", text="Gender")
        self.voice_tree.heading("Status", text="Status")
        
        # Voice tree column widths
        self.voice_tree.column("#0", width=200)
        self.voice_tree.column("Engine", width=100)
        self.voice_tree.column("Language", width=80)
        self.voice_tree.column("Gender", width=80)
        self.voice_tree.column("Status", width=100)
        
        # Voice tree scrollbar
        voice_scrollbar = ttk.Scrollbar(voice_tree_frame, orient="vertical")
        voice_scrollbar.grid(row=0, column=1, sticky="ns")
        self.voice_tree.config(yscrollcommand=voice_scrollbar.set)
        voice_scrollbar.config(command=self.voice_tree.yview)
        
        # Voice control buttons
        voice_btn_frame = ttk.Frame(control_frame)
        voice_btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 10))
        
        ttk.Button(voice_btn_frame, text="🔄 Refresh Voices", 
                  command=self.refresh_voices).pack(side="left")
        ttk.Button(voice_btn_frame, text="🎵 Preview Voice", 
                  command=self.preview_voice).pack(side="left", padx=(5, 0))
        ttk.Button(voice_btn_frame, text="⚡ Load Selected", 
                  command=self.load_selected_voice).pack(side="left", padx=(5, 0))
        
        # Selected voice info
        self.selected_voice_var = tk.StringVar(value="No voice selected")
        ttk.Label(voice_btn_frame, textvariable=self.selected_voice_var).pack(side="right")
        
        # Generation button
        self.generate_button = ttk.Button(control_frame, text="🎙️ Generate Speech", 
                                         style="Accent.TButton",
                                         command=self.generate_speech,
                                         state="disabled")
        self.generate_button.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.generation_progress,
                                           maximum=100, length=400)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        
    def create_batch_tab(self):
        """Create the batch processing tab"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="Batch Processing")
        
        batch_frame.columnconfigure(0, weight=1)
        batch_frame.rowconfigure(1, weight=1)
        
        # File list
        file_frame = ttk.LabelFrame(batch_frame, text="Files to Process", padding="10")
        file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        file_frame.columnconfigure(0, weight=1)
        
        # File listbox with scrollbar
        listbox_frame = ttk.Frame(file_frame)
        listbox_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        listbox_frame.columnconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(listbox_frame, height=6)
        self.file_listbox.grid(row=0, column=0, sticky="ew")
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.file_listbox.yview)
        
        # File management buttons
        file_buttons = ttk.Frame(file_frame)
        file_buttons.grid(row=1, column=0, sticky="ew")
        
        ttk.Button(file_buttons, text="Add Files", 
                  command=self.add_files).pack(side="left")
        ttk.Button(file_buttons, text="Add Folder", 
                  command=self.add_folder).pack(side="left", padx=(5, 0))
        ttk.Button(file_buttons, text="Remove Selected", 
                  command=self.remove_files).pack(side="left", padx=(5, 0))
        ttk.Button(file_buttons, text="Clear All", 
                  command=self.clear_files).pack(side="left", padx=(5, 0))
        
        # Batch processing controls
        batch_control_frame = ttk.LabelFrame(batch_frame, text="Batch Controls", padding="10")
        batch_control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        ttk.Button(batch_control_frame, text="🚀 Start Batch Processing", 
                  style="Accent.TButton",
                  command=self.start_batch_processing).pack(pady=10)
        
    def create_voice_tab(self):
        """Create the voice management tab"""
        voice_frame = ttk.Frame(self.notebook)
        self.notebook.add(voice_frame, text="Voice Management")
        
        voice_frame.columnconfigure(0, weight=1)
        voice_frame.columnconfigure(1, weight=1)
        
        # Available voices
        available_frame = ttk.LabelFrame(voice_frame, text="Available Voices", padding="10")
        available_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=5)
        available_frame.columnconfigure(0, weight=1)
        available_frame.rowconfigure(0, weight=1)
        
        self.available_voices_tree = ttk.Treeview(available_frame, columns=("Type", "Language"), 
                                                 show="tree headings")
        self.available_voices_tree.grid(row=0, column=0, sticky="nsew")
        
        # Voice tree headings
        self.available_voices_tree.heading("#0", text="Voice Name")
        self.available_voices_tree.heading("Type", text="Type")
        self.available_voices_tree.heading("Language", text="Language")
        
        # Custom voices
        custom_frame = ttk.LabelFrame(voice_frame, text="Custom Voice Training", padding="10")
        custom_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=5)
        
        ttk.Label(custom_frame, text="Voice Clone Training").pack(pady=(0, 10))
        ttk.Button(custom_frame, text="Upload Audio Samples", 
                  command=self.upload_voice_samples).pack(pady=5)
        ttk.Button(custom_frame, text="Train New Voice", 
                  command=self.train_voice).pack(pady=5)
        
    def create_settings_tab(self):
        """Create the settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        settings_frame.columnconfigure(0, weight=1)
        
        # Generation settings
        gen_frame = ttk.LabelFrame(settings_frame, text="Generation Settings", padding="10")
        gen_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        gen_frame.columnconfigure(1, weight=1)
        
        # Inference steps
        ttk.Label(gen_frame, text="Inference Steps:").grid(row=0, column=0, sticky="w")
        ttk.Scale(gen_frame, from_=1, to=50, variable=self.inference_steps, 
                 orient="horizontal").grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(gen_frame, textvariable=self.inference_steps).grid(row=0, column=2)
        
        # Temperature
        ttk.Label(gen_frame, text="Temperature:").grid(row=1, column=0, sticky="w")
        ttk.Scale(gen_frame, from_=0.1, to=2.0, variable=self.temperature, 
                 orient="horizontal").grid(row=1, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(gen_frame, textvariable=self.temperature).grid(row=1, column=2)
        
        # Top-p
        ttk.Label(gen_frame, text="Top-p:").grid(row=2, column=0, sticky="w")
        ttk.Scale(gen_frame, from_=0.1, to=1.0, variable=self.top_p, 
                 orient="horizontal").grid(row=2, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(gen_frame, textvariable=self.top_p).grid(row=2, column=2)
        
        # Speed factor
        ttk.Label(gen_frame, text="Speed Factor:").grid(row=3, column=0, sticky="w")
        ttk.Scale(gen_frame, from_=0.5, to=2.0, variable=self.speed_factor, 
                 orient="horizontal").grid(row=3, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(gen_frame, textvariable=self.speed_factor).grid(row=3, column=2)
        
        # Output settings
        output_frame = ttk.LabelFrame(settings_frame, text="Output Settings", padding="10")
        output_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_dir).grid(row=0, column=1, 
                                                                  sticky="ew", padx=(5, 5))
        ttk.Button(output_frame, text="Browse", 
                  command=self.browse_output_dir).grid(row=0, column=2)
        
    def create_status_bar(self, parent):
        """Create the status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        
        # Status label
        self.status_label = ttk.Label(status_frame, textvariable=self.status_text)
        self.status_label.grid(row=0, column=0, sticky="w")
        
        # Device indicator
        device_text = f"Device: {self.device}"
        if torch.cuda.is_available():
            device_text += f" ({torch.cuda.get_device_name()})"
        ttk.Label(status_frame, text=device_text).grid(row=0, column=1, sticky="e")
        
    def setup_bindings(self):
        """Setup keyboard shortcuts and event bindings"""
        self.root.bind('<Control-o>', lambda e: self.load_script())
        self.root.bind('<Control-s>', lambda e: self.save_script())
        self.root.bind('<F5>', lambda e: self.generate_speech())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        
    # Model Management Methods
    def browse_model(self):
        """Browse for model directory"""
        directory = filedialog.askdirectory(title="Select Model Directory")
        if directory:
            self.model_path.set(directory)
            
    def discover_voices(self):
        """Discover all available voices from TTS engines"""
        if not TTS_BACKEND_AVAILABLE:
            messagebox.showerror("Error", "TTS backend not available. Please install required packages.")
            return
            
        def discovery_task():
            try:
                self.status_text.set("Discovering voices...")
                self.engine_status_label.config(text="Discovering...", foreground="orange")
                
                # Initialize backend if not already done
                if not self.tts_backend:
                    self.tts_backend = get_tts_backend()
                
                # Get all available voices
                voices = self.tts_backend.get_available_voices()
                self.available_voices = voices
                
                # Get engine status
                engine_status = self.tts_backend.get_engine_status()
                
                self.result_queue.put(("voices_discovered", {
                    "voices": voices,
                    "engine_status": engine_status
                }))
                
            except Exception as e:
                self.result_queue.put(("discovery_error", str(e)))
                
        # Start discovery in background thread
        self.worker_thread = threading.Thread(target=discovery_task)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # Disable discovery button temporarily
        self.load_button.config(state="disabled")
        
    def refresh_voices(self):
        """Refresh the voice list display"""
        if not self.available_voices:
            self.discover_voices()
            return
            
        # Clear existing items
        for item in self.voice_tree.get_children():
            self.voice_tree.delete(item)
            
        # Populate voice tree
        for voice_id, voice in self.available_voices.items():
            # Determine status
            status = "Available"
            if self.tts_backend:
                model_key = f"{voice.engine.value}_{voice.model_path}"
                if model_key in self.tts_backend.models_loaded:
                    status = "Loaded"
                    
            self.voice_tree.insert("", "end", text=voice.name,
                                  values=(voice.engine.value.title(), 
                                         voice.language, 
                                         voice.gender, 
                                         status))
                                         
    def load_selected_voice(self):
        """Load the selected voice model"""
        selection = self.voice_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a voice first")
            return
            
        # Get selected voice
        item = self.voice_tree.item(selection[0])
        voice_name = item["text"]
        
        # Find the voice object
        selected_voice = None
        for voice_id, voice in self.available_voices.items():
            if voice.name == voice_name:
                selected_voice = voice
                break
                
        if not selected_voice:
            messagebox.showerror("Error", "Selected voice not found")
            return
            
        def load_task():
            try:
                self.status_text.set(f"Loading {voice_name}...")
                success = self.tts_backend.load_model(selected_voice)
                self.result_queue.put(("voice_loaded", {
                    "voice": selected_voice,
                    "success": success
                }))
            except Exception as e:
                self.result_queue.put(("voice_load_error", str(e)))
                
        # Start loading in background
        threading.Thread(target=load_task, daemon=True).start()
        
    def preview_voice(self):
        """Preview the selected voice"""
        selection = self.voice_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a voice first")
            return
            
        # Get selected voice
        item = self.voice_tree.item(selection[0])
        voice_name = item["text"]
        
        # Find the voice object
        selected_voice = None
        for voice_id, voice in self.available_voices.items():
            if voice.name == voice_name:
                selected_voice = voice
                break
                
        if not selected_voice:
            return
            
        # Generate preview
        preview_text = "Hello, this is a voice preview test."
        output_path = Path(self.output_dir.get()) / f"preview_{voice_name.replace(' ', '_')}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        def preview_task():
            try:
                self.status_text.set(f"Generating preview for {voice_name}...")
                request = TTSRequest(
                    text=preview_text,
                    voice=selected_voice,
                    output_path=str(output_path)
                )
                
                success = self.tts_backend.generate_speech(request)
                self.result_queue.put(("preview_generated", {
                    "voice": voice_name,
                    "success": success,
                    "path": str(output_path)
                }))
            except Exception as e:
                self.result_queue.put(("preview_error", str(e)))
                
        threading.Thread(target=preview_task, daemon=True).start()
        
    def benchmark_voices(self):
        """Benchmark all available voices"""
        if not self.available_voices:
            messagebox.showwarning("Warning", "Please discover voices first")
            return
            
        def benchmark_task():
            try:
                self.status_text.set("Benchmarking voices...")
                results = []
                
                for voice_id, voice in list(self.available_voices.items())[:5]:  # Limit to 5 for demo
                    result = self.tts_backend.benchmark_voice(voice)
                    results.append(result)
                    
                self.result_queue.put(("benchmark_complete", results))
                
            except Exception as e:
                self.result_queue.put(("benchmark_error", str(e)))
                
        threading.Thread(target=benchmark_task, daemon=True).start()
        
    # Text Generation Methods
    def load_script(self):
        """Load script from file"""
        file_path = filedialog.askopenfilename(
            title="Load Script",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.text_input.delete('1.0', tk.END)
                self.text_input.insert('1.0', content)
                self.status_text.set(f"Loaded script: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load script: {e}")
                
    def save_script(self):
        """Save script to file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Script",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                content = self.text_input.get('1.0', tk.END)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.status_text.set(f"Saved script: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save script: {e}")
                
    def clear_script(self):
        """Clear the script input"""
        self.text_input.delete('1.0', tk.END)
        
    def generate_speech(self):
        """Generate speech using the selected voice"""
        # Check if we have a selected voice
        selection = self.voice_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a voice first")
            return
            
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to generate")
            return
            
        # Get selected voice
        item = self.voice_tree.item(selection[0])
        voice_name = item["text"]
        
        # Find the voice object
        selected_voice = None
        for voice_id, voice in self.available_voices.items():
            if voice.name == voice_name:
                selected_voice = voice
                break
                
        if not selected_voice:
            messagebox.showerror("Error", "Selected voice not found")
            return
            
        def generation_task():
            try:
                self.status_text.set("Generating speech...")
                self.generation_progress.set(0)
                
                # Create output path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_voice_name = voice_name.replace(' ', '_').replace('/', '_')
                output_path = Path(self.output_dir.get()) / f"generated_{safe_voice_name}_{timestamp}.wav"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create TTS request
                request = TTSRequest(
                    text=text,
                    voice=selected_voice,
                    output_path=str(output_path),
                    speed=self.speed_factor.get()
                )
                
                # Update progress
                self.generation_progress.set(25)
                
                # Generate speech
                success = self.tts_backend.generate_speech(request)
                
                self.generation_progress.set(100)
                
                self.result_queue.put(("generation_complete", {
                    "success": success,
                    "path": str(output_path),
                    "voice": voice_name
                }))
                
            except Exception as e:
                self.result_queue.put(("generation_error", str(e)))
                
        # Start generation in background thread
        self.worker_thread = threading.Thread(target=generation_task)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # Disable generate button
        self.generate_button.config(state="disabled")
        
    # Batch Processing Methods
    def add_files(self):
        """Add files to batch processing list"""
        files = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        for file in files:
            self.file_listbox.insert(tk.END, file)
            
    def add_folder(self):
        """Add all text files from a folder"""
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            for file_path in Path(folder).rglob("*.txt"):
                self.file_listbox.insert(tk.END, str(file_path))
                
    def remove_files(self):
        """Remove selected files from list"""
        selection = self.file_listbox.curselection()
        for index in reversed(selection):
            self.file_listbox.delete(index)
            
    def clear_files(self):
        """Clear all files from list"""
        self.file_listbox.delete(0, tk.END)
        
    def start_batch_processing(self):
        """Start batch processing of files"""
        files = list(self.file_listbox.get(0, tk.END))
        if not files:
            messagebox.showwarning("Warning", "No files selected for batch processing")
            return
            
        if not self.model_loaded.get():
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        messagebox.showinfo("Info", f"Starting batch processing of {len(files)} files")
        
    # Voice Management Methods
    def upload_voice_samples(self):
        """Upload voice samples for training"""
        files = filedialog.askopenfilenames(
            title="Select Audio Samples",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if files:
            messagebox.showinfo("Info", f"Selected {len(files)} audio files for voice training")
            
    def train_voice(self):
        """Train a new voice model"""
        messagebox.showinfo("Info", "Voice training feature coming soon!")
        
    # Utility Methods
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            
    def open_output_folder(self):
        """Open the output folder"""
        output_path = Path(self.output_dir.get())
        output_path.mkdir(parents=True, exist_ok=True)
        
        if sys.platform == "win32":
            os.startfile(output_path)
        elif sys.platform == "darwin":
            os.system(f"open '{output_path}'")
        else:
            os.system(f"xdg-open '{output_path}'")
            
    def open_web_interface(self):
        """Open the web interface"""
        webbrowser.open("http://localhost:7860")
        
    def open_settings(self):
        """Open settings dialog"""
        self.notebook.select(3)  # Select settings tab
        
    def process_queue(self):
        """Process results from background threads"""
        try:
            while True:
                result_type, data = self.result_queue.get_nowait()
                
                if result_type == "voices_discovered":
                    self.load_button.config(state="normal")
                    self.available_voices = data["voices"]
                    self.refresh_voices()
                    
                    # Update engine status
                    engine_status = data["engine_status"]
                    total_voices = sum(status["total_voices"] for status in engine_status.values())
                    available_engines = [engine for engine, status in engine_status.items() if status["available"]]
                    
                    status_text = f"Found {total_voices} voices from {len(available_engines)} engines"
                    self.engine_status_label.config(text=status_text, foreground="green")
                    self.status_text.set(f"Voice discovery complete: {total_voices} voices available")
                    
                elif result_type == "discovery_error":
                    self.load_button.config(state="normal")
                    messagebox.showerror("Error", f"Failed to discover voices: {data}")
                    self.engine_status_label.config(text="Discovery failed", foreground="red")
                    self.status_text.set("Voice discovery failed")
                    
                elif result_type == "voice_loaded":
                    voice = data["voice"]
                    success = data["success"]
                    if success:
                        self.selected_voice = voice
                        self.selected_voice_var.set(f"Loaded: {voice.name}")
                        self.generate_button.config(state="normal")
                        self.refresh_voices()  # Update status display
                        self.status_text.set(f"Voice loaded: {voice.name}")
                    else:
                        messagebox.showerror("Error", f"Failed to load voice: {voice.name}")
                        
                elif result_type == "voice_load_error":
                    messagebox.showerror("Error", f"Voice loading failed: {data}")
                    
                elif result_type == "preview_generated":
                    if data["success"]:
                        self.status_text.set(f"Preview generated for {data['voice']}")
                        # Optionally play the preview
                        if messagebox.askyesno("Preview Ready", 
                                             f"Preview generated for {data['voice']}. Open output folder?"):
                            self.open_output_folder()
                    else:
                        messagebox.showerror("Error", f"Preview generation failed for {data['voice']}")
                        
                elif result_type == "preview_error":
                    messagebox.showerror("Error", f"Preview generation failed: {data}")
                    
                elif result_type == "benchmark_complete":
                    results = data
                    self.show_benchmark_results(results)
                    self.status_text.set("Benchmark complete")
                    
                elif result_type == "benchmark_error":
                    messagebox.showerror("Error", f"Benchmark failed: {data}")
                    
                elif result_type == "generation_complete":
                    self.generate_button.config(state="normal")
                    self.generation_progress.set(0)
                    
                    if data["success"]:
                        self.status_text.set(f"Generation complete: {Path(data['path']).name}")
                        messagebox.showinfo("Success", 
                                          f"Audio generated with {data['voice']}!\n\nFile: {data['path']}")
                    else:
                        messagebox.showerror("Error", f"Generation failed for {data['voice']}")
                    
                elif result_type == "generation_error":
                    self.generate_button.config(state="normal")
                    self.generation_progress.set(0)
                    messagebox.showerror("Error", f"Generation failed: {data}")
                    self.status_text.set("Generation failed")
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self.process_queue)
        
    def show_benchmark_results(self, results):
        """Show benchmark results in a new window"""
        result_window = tk.Toplevel(self.root)
        result_window.title("Voice Benchmark Results")
        result_window.geometry("600x400")
        
        # Create treeview for results
        tree_frame = ttk.Frame(result_window, padding="10")
        tree_frame.pack(fill="both", expand=True)
        
        tree = ttk.Treeview(tree_frame, 
                           columns=("Engine", "Time", "WPS", "Status"), 
                           show="tree headings")
        tree.pack(fill="both", expand=True)
        
        # Configure columns
        tree.heading("#0", text="Voice")
        tree.heading("Engine", text="Engine")
        tree.heading("Time", text="Time (s)")
        tree.heading("WPS", text="Words/Sec")
        tree.heading("Status", text="Status")
        
        # Add results
        for result in results:
            status = "✅ Success" if result["success"] else "❌ Failed"
            time_str = f"{result.get('generation_time', 0):.2f}" if result["success"] else "N/A"
            wps_str = f"{result.get('words_per_second', 0):.1f}" if result["success"] else "N/A"
            
            tree.insert("", "end", text=result["voice"],
                       values=(result["engine"], time_str, wps_str, status))
        

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = VibeVoiceGUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        root.destroy()


if __name__ == "__main__":
    main()
