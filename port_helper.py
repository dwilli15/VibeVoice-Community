"""
VibeVoice GUI Port Management - Quick Reference
Run this to see available options and current port status
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("🎤 VIBEVOICE GUI - PORT MANAGEMENT HELPER")
    print("=" * 60)
    
    # Check current port status
    print("\n📊 CURRENT PORT STATUS:")
    try:
        result = subprocess.run([
            sys.executable, "port_manager.py", "--scan"
        ], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error checking ports: {e}")
    
    print("\n🚀 STARTUP OPTIONS:")
    print("1. Clean Startup (Recommended):")
    print("   python start_gui.py")
    print("   OR: start_gui.bat")
    print()
    print("2. Use Specific Port:")
    print("   python start_gui.py --port 7863")
    print()
    print("3. Manual GUI Launch:")
    print("   python ebook_gui.py --port 7862 --force-kill")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("If GUI gets stuck or shows 'empty':")
    print("1. Kill port processes: python port_manager.py --port 7862 --kill")
    print("2. Start clean:         python start_gui.py")
    print("3. Check status:        python port_manager.py --scan")
    
    print("\n🛠️ PORT UTILITIES:")
    print("• Check specific port:  python port_manager.py --port 7862")
    print("• Find free port:       python port_manager.py --find-free")
    print("• Force kill port:      python port_manager.py --port 7862 --kill --force")
    print("• Scan all ports:       python port_manager.py --scan")
    
    print("\n" + "=" * 60)
    print("💡 TIP: Always use 'python start_gui.py' for the cleanest experience!")
    print("=" * 60)
    
    # Interactive option
    print("\n🎯 QUICK ACTIONS:")
    print("1. Start GUI now")
    print("2. Check port 7862")
    print("3. Clean port 7862")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting GUI with clean port management...")
            subprocess.run([sys.executable, "start_gui.py"])
        elif choice == "2":
            print("\n📊 Checking port 7862...")
            result = subprocess.run([sys.executable, "port_manager.py", "--port", "7862"])
        elif choice == "3":
            print("\n🧹 Cleaning port 7862...")
            result = subprocess.run([sys.executable, "port_manager.py", "--port", "7862", "--kill", "--force"])
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
