# Port Management Guide for VibeVoice GUI

This guide explains how to resolve port conflicts and ensure clean startup of the VibeVoice GUI.

## Problem

The VibeVoice GUI sometimes gets "stuck" on ports, causing:
- "Empty" responses when troubleshooting
- Unable to start new instances
- Port already in use errors
- Browser showing connection refused

## Solution

The enhanced port management system automatically:
1. **Detects port conflicts** before starting
2. **Cleans up stuck processes** automatically
3. **Finds alternative ports** if needed
4. **Properly closes servers** when GUI exits

## Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
# Clean startup with automatic port cleanup
python start_gui.py

# Use specific port
python start_gui.py --port 7863

# Skip automatic cleanup
python start_gui.py --no-cleanup
```

### Option 2: Use the Batch File (Windows)
```bash
# Double-click or run:
start_gui.bat
```

### Option 3: Manual Port Management
```bash
# Check port status
python port_manager.py --port 7862

# Scan common ports
python port_manager.py --scan

# Kill processes on port (with confirmation)
python port_manager.py --port 7862 --kill

# Force kill processes (no confirmation)
python port_manager.py --port 7862 --kill --force

# Find next free port
python port_manager.py --find-free
```

## Utilities

### port_manager.py
**Purpose**: Manage ports and kill stuck processes
**Usage**: 
```bash
python port_manager.py [options]
  --port 7862       # Port to manage (default: 7862)
  --kill            # Kill processes on port
  --force           # Force kill without confirmation
  --scan            # Scan ports 7860-7870
  --find-free       # Find next available port
```

### start_gui.py
**Purpose**: Clean startup with automatic port management
**Usage**:
```bash
python start_gui.py [options]
  --port 7862       # Port to use (default: 7862)
  --no-cleanup      # Skip port cleanup
  --check-only      # Only check port status
```

### start_gui.bat
**Purpose**: Windows batch file for easy startup
**Usage**: Double-click or run from command prompt

## Enhanced GUI Features

The `ebook_gui.py` now includes:

1. **Automatic Port Detection**: Finds free ports automatically
2. **Process Cleanup**: Kills stuck processes on exit
3. **Graceful Shutdown**: Proper cleanup when closing
4. **Error Recovery**: Handles port conflicts gracefully

### New Command Line Options

```bash
python ebook_gui.py [options]
  --port 7862       # Port to use
  --host 0.0.0.0    # Host to bind to
  --force-kill      # Force kill processes on port before starting
```

## Troubleshooting

### Problem: Port is busy
```bash
# Check what's using the port
python port_manager.py --port 7862

# Kill processes and start fresh
python port_manager.py --port 7862 --kill --force
python start_gui.py
```

### Problem: GUI shows "empty" or connection refused
```bash
# Use the clean startup script
python start_gui.py

# Or force cleanup first
python start_gui.py --port 7862
```

### Problem: Can't find free port
```bash
# Scan for available ports
python port_manager.py --scan

# Find next free port
python port_manager.py --find-free

# Use a different port range
python start_gui.py --port 8000
```

### Problem: Multiple GUI instances
```bash
# Kill all Gradio processes
python port_manager.py --scan
# Then kill each busy port manually

# Or scan and kill common ports
for i in {7860..7870}; do python port_manager.py --port $i --kill --force; done
```

## Best Practices

1. **Always use start_gui.py** for launching
2. **Let the system find free ports** automatically
3. **Use Ctrl+C to stop** the GUI properly
4. **Run port cleanup** if experiencing issues
5. **Use --force-kill** if GUI was forcefully terminated

## Technical Details

### Port Detection Algorithm
1. Try requested port
2. If busy, attempt cleanup
3. Find next available port in range
4. Register cleanup handlers for proper shutdown

### Process Management
- Uses `psutil` to detect port usage
- Graceful termination with fallback to force kill
- Automatic cleanup on script exit
- Support for Windows process management

### Error Handling
- Unicode-safe output for Windows
- Graceful fallback when utilities fail
- Clear error messages and suggestions
- Automatic recovery from common issues

## Files Modified

- `ebook_gui.py`: Enhanced with port management
- `port_manager.py`: New utility for port management
- `start_gui.py`: Clean startup script
- `start_gui.bat`: Windows batch file for easy startup

## Dependencies

- `psutil`: For process and network management
- `socket`: For port availability checking
- Standard Python libraries

Install with:
```bash
pip install psutil
```
