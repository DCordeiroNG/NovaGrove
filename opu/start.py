#!/usr/bin/env python3
"""
Easy startup script for AI Persona Chat System
Handles setup and launches both backend and frontend
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   You have: Python {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_virtual_env():
    """Create virtual environment if it doesn't exist"""
    if not Path("venv").exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment exists")

def install_dependencies():
    """Install required packages"""
    print("📚 Installing dependencies...")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = Path("venv/Scripts/pip")
    else:
        pip_path = Path("venv/bin/pip")
    
    # Install requirements
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    print("✅ Dependencies installed")

def check_files():
    """Ensure all required files exist"""
    required_files = ["main.py", "index.html", "requirements.txt"]
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        print("   Please ensure all files are in the current directory")
        sys.exit(1)
    print("✅ All required files present")

def start_backend():
    """Start the FastAPI backend"""
    print("\n🚀 Starting backend server...")
    
    # Determine python path based on OS
    if sys.platform == "win32":
        python_path = Path("venv/Scripts/python")
    else:
        python_path = Path("venv/bin/python")
    
    # Start backend as subprocess
    backend_process = subprocess.Popen(
        [str(python_path), "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    print("   (First run will download AI model ~1.5GB)")
    
    # Give it some time to start
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)
        # Check if backend is responsive
        try:
            import requests
            response = requests.get("http://localhost:8000")
            if response.status_code == 200:
                print("✅ Backend is running!")
                return backend_process
        except:
            if i % 5 == 0:
                print(f"   Still starting... ({i}s)")
    
    print("⚠️  Backend is taking longer than expected to start")
    print("   Check the console for any errors")
    return backend_process

def start_frontend():
    """Start a simple HTTP server for the frontend"""
    print("\n🌐 Starting frontend server...")
    
    # Start simple HTTP server
    if sys.platform == "win32":
        python_path = Path("venv/Scripts/python")
    else:
        python_path = Path("venv/bin/python")
    
    frontend_process = subprocess.Popen(
        [str(python_path), "-m", "http.server", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(2)  # Give it a moment to start
    print("✅ Frontend server running at http://localhost:8080")
    
    return frontend_process

def open_browser():
    """Open the application in default browser"""
    print("\n🌍 Opening browser...")
    time.sleep(1)
    webbrowser.open("http://localhost:8080")

def main():
    """Main startup sequence"""
    print("🎭 AI Persona Chat System Startup")
    print("=" * 40)
    
    try:
        # Check environment
        check_python_version()
        check_files()
        
        # Setup
        create_virtual_env()
        install_dependencies()
        
        # Start services
        backend = start_backend()
        frontend = start_frontend()
        
        # Open browser
        open_browser()
        
        print("\n✨ AI Persona Chat is running!")
        print("\n📍 URLs:")
        print("   Frontend: http://localhost:8080")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        
        print("\n⚡ Quick Tips:")
        print("   - Each persona has unique trigger words")
        print("   - Watch the mood indicator change")
        print("   - Try different approaches with each persona")
        
        print("\n🛑 Press Ctrl+C to stop all services\n")
        
        # Keep running
        try:
            backend.wait()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            backend.terminate()
            frontend.terminate()
            print("✅ All services stopped")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()