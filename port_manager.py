"""
Port Management Utility for VibeVoice GUI
Helps clean up stuck ports and manage Gradio server instances
"""

import psutil
import socket
import time
import argparse
import sys
import os

# Fix Unicode encoding issues on Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

class PortManager:
    """Utility class for managing ports and processes"""
    
    @staticmethod
    def find_processes_on_port(port: int):
        """Find all processes using the specified port"""
        processes = []
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    try:
                        process = psutil.Process(conn.pid)
                        processes.append({
                            'pid': conn.pid,
                            'name': process.name(),
                            'cmdline': ' '.join(process.cmdline()),
                            'status': process.status()
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            print(f"Error scanning port {port}: {e}")
        return processes
    
    @staticmethod
    def kill_port_processes(port: int, force: bool = False):
        """Kill all processes using the specified port"""
        processes = PortManager.find_processes_on_port(port)
        
        if not processes:
            print(f"[OK] No processes found on port {port}")
            return True
        
        print(f"[INFO] Found {len(processes)} processes on port {port}:")
        for proc in processes:
            print(f"  PID {proc['pid']}: {proc['name']} - {proc['cmdline'][:100]}")
        
        if not force:
            confirm = input("Kill these processes? (y/N): ").lower().strip()
            if confirm != 'y':
                print("[ABORT] Aborted")
                return False
        
        killed = 0
        for proc in processes:
            try:
                process = psutil.Process(proc['pid'])
                print(f"[KILL] Terminating PID {proc['pid']} ({proc['name']})...")
                process.terminate()
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=5)
                    killed += 1
                    print(f"[OK] Terminated PID {proc['pid']}")
                except psutil.TimeoutExpired:
                    print(f"[FORCE] Force killing PID {proc['pid']}...")
                    process.kill()
                    killed += 1
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"[WARN] Could not kill PID {proc['pid']}: {e}")
        
        print(f"[DONE] Killed {killed}/{len(processes)} processes")
        return killed == len(processes)
    
    @staticmethod
    def is_port_free(port: int) -> bool:
        """Check if a port is free"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except OSError:
            return False
    
    @staticmethod
    def find_free_port(start_port: int = 7862, max_ports: int = 100) -> int:
        """Find the next free port starting from start_port"""
        for port in range(start_port, start_port + max_ports):
            if PortManager.is_port_free(port):
                return port
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_ports}")
    
    @staticmethod
    def scan_port_range(start_port: int = 7860, end_port: int = 7870):
        """Scan a range of ports and show their status"""
        print(f"[SCAN] Scanning ports {start_port}-{end_port}:")
        print("-" * 60)
        
        for port in range(start_port, end_port + 1):
            if PortManager.is_port_free(port):
                print(f"Port {port}: [FREE]")
            else:
                processes = PortManager.find_processes_on_port(port)
                if processes:
                    proc_names = [p['name'] for p in processes]
                    print(f"Port {port}: [BUSY] ({', '.join(set(proc_names))})")
                else:
                    print(f"Port {port}: [BUSY] (unknown)")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Port Management Utility for VibeVoice GUI")
    parser.add_argument("--port", type=int, default=7862, help="Port to manage")
    parser.add_argument("--kill", action="store_true", help="Kill processes on port")
    parser.add_argument("--force", action="store_true", help="Force kill without confirmation")
    parser.add_argument("--scan", action="store_true", help="Scan common ports")
    parser.add_argument("--find-free", action="store_true", help="Find next free port")
    parser.add_argument("--start-port", type=int, default=7860, help="Start port for scanning")
    parser.add_argument("--end-port", type=int, default=7870, help="End port for scanning")
    
    args = parser.parse_args()
    
    if args.scan:
        PortManager.scan_port_range(args.start_port, args.end_port)
    elif args.find_free:
        try:
            free_port = PortManager.find_free_port(args.port)
            print(f"[OK] Next free port: {free_port}")
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    elif args.kill:
        success = PortManager.kill_port_processes(args.port, args.force)
        if not success:
            sys.exit(1)
    else:
        # Default: show port status
        if PortManager.is_port_free(args.port):
            print(f"[OK] Port {args.port} is FREE")
        else:
            print(f"[BUSY] Port {args.port} is BUSY")
            processes = PortManager.find_processes_on_port(args.port)
            if processes:
                print("Processes using this port:")
                for proc in processes:
                    print(f"  PID {proc['pid']}: {proc['name']} - {proc['cmdline'][:100]}")
            
            print(f"\nTo kill processes: python {sys.argv[0]} --port {args.port} --kill")

if __name__ == "__main__":
    main()
