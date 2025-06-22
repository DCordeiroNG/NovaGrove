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
    
    # Install requirements + requests for startup script
    result = subprocess.run([str(pip_path), "install", "-r", "requirements.txt", "requests"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies: {result.stderr}")
        sys.exit(1)
    
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
    
    # Start backend - show output for debugging
    print("   💡 Backend output will show below:")
    backend_process = subprocess.Popen(
        [str(python_path), "main.py"]
        # No stdout/stderr capture so you can see what's happening
    )
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    print("   (First run may take 30-60 seconds to load AI model)")
    
    # Give it some time to start and try to connect
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)
        
        # Try to check if backend is responsive
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            
            if result == 0:  # Port is open
                print("✅ Backend is running!")
                time.sleep(2)  # Give it a moment to fully initialize
                return backend_process
                
        except Exception:
            pass
        
        if i % 5 == 0 and i > 0:
            print(f"   Still starting... ({i}s)")
    
    print("⚠️  Backend is taking longer than expected")
    print("   Check the output above for any errors")
    print("   The system may still work - continuing anyway...")
    return backend_process

def start_frontend():
    """Start a simple HTTP server for the frontend"""
    print("\n🌐 Starting frontend server...")
    
    # Start simple HTTP server
    if sys.platform == "win32":
        python_path = Path("venv/Scripts/python")
    else:
        python_path = Path("venv/bin/python")
    
    print("   Starting HTTP server on port 8080...")
    frontend_process = subprocess.Popen(
        [str(python_path), "-m", "http.server", "8080"]
    )
    
    time.sleep(3)  # Give it a moment to start
    
    # Test if frontend is running
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8080))
        sock.close()
        
        if result == 0:
            print("✅ Frontend server running at http://localhost:8080")
        else:
            print("⚠️  Frontend may not have started properly")
            
    except Exception:
        print("⚠️  Could not verify frontend status")
    
    return frontend_process

def open_browser():
    """Open the application in default browser"""
    print("\n🌍 Opening browser...")
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8080")
        print("✅ Browser should open automatically")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please manually open: http://localhost:8080")

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
        print("\n" + "="*50)
        backend = start_backend()
        
        print("\n" + "="*50)
        frontend = start_frontend()
        
        # Open browser
        open_browser()
        
        print("\n" + "="*50)
        print("✨ AI Persona Chat is running!")
        print("\n📍 URLs:")
        print("   🎭 Frontend: http://localhost:8080")
        print("   🔧 Backend API: http://localhost:8000")
        print("   📖 API Docs: http://localhost:8000/docs")
        
        print("\n⚡ Quick Tips:")
        print("   - Try Enterprise Emma first")
        print("   - Use positive words like 'ROI', 'security'")
        print("   - Watch the mood indicator change")
        
        print("\n🛑 Press Ctrl+C to stop all services")
        print("   (You may need to press it twice)")
        print("\n" + "="*50)
        
        # Keep running and handle shutdown
        try:
            while True:
                # Check if processes are still running
                if backend.poll() is not None:
                    print("\n❌ Backend process stopped unexpectedly")
                    break
                if frontend.poll() is not None:
                    print("\n❌ Frontend process stopped unexpectedly")
                    break
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down gracefully...")
            
            try:
                backend.terminate()
                frontend.terminate()
                
                # Wait a bit for graceful shutdown
                time.sleep(2)
                
                # Force kill if still running
                if backend.poll() is None:
                    backend.kill()
                if frontend.poll() is None:
                    frontend.kill()
                    
                print("✅ All services stopped")
                
            except Exception as e:
                print(f"⚠️  Error during shutdown: {e}")
            
    except Exception as e:
        print(f"\n❌ Startup Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you're in the correct directory")
        print("   2. Check that all files exist")
        print("   3. Try running manually:")
        print("      python main.py")
        sys.exit(1)

if __name__ == "__main__":
    main()