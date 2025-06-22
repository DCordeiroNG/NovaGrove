#!/usr/bin/env python3
"""
Quick status check for AI Persona Chat download
Run this to see what's happening with your model download
"""

import os
import sys
import time
import psutil
from pathlib import Path

def check_download_status():
    """Check current download status and provide guidance"""
    print("🔍 AI Persona Chat - Download Status Check")
    print("=" * 50)
    
    # 1. Check if Python processes are running
    print("\n1️⃣ Checking running Python processes...")
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_processes:
        print(f"   Found {len(python_processes)} Python process(es) running:")
        for proc in python_processes:
            try:
                memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                print(f"   • PID {proc.info['pid']}: {memory_mb:.1f} MB RAM")
            except:
                print(f"   • PID {proc.info['pid']}: Unknown memory")
    else:
        print("   ❌ No Python processes found running")
    
    # 2. Check network activity
    print("\n2️⃣ Checking network activity...")
    try:
        net_before = psutil.net_io_counters()
        time.sleep(2)
        net_after = psutil.net_io_counters()
        
        bytes_recv = net_after.bytes_recv - net_before.bytes_recv
        bytes_sent = net_after.bytes_sent - net_before.bytes_sent
        
        if bytes_recv > 1024:  # More than 1KB received
            print(f"   📥 Network activity detected: {bytes_recv/1024:.1f} KB/s download")
        else:
            print("   📴 No significant network activity")
            
    except Exception as e:
        print(f"   ⚠️  Could not check network: {e}")
    
    # 3. Check cache directory
    print("\n3️⃣ Checking model cache...")
    try:
        from transformers import TRANSFORMERS_CACHE
        cache_dir = Path(TRANSFORMERS_CACHE)
        print(f"   📁 Cache location: {cache_dir}")
        
        if cache_dir.exists():
            print(f"   ✅ Cache directory exists")
            
            # Check for DialoGPT model
            model_dir = cache_dir / "models--microsoft--DialoGPT-medium"
            if model_dir.exists():
                print(f"   📦 DialoGPT directory found!")
                
                # Check files and sizes
                total_size = 0
                file_count = 0
                
                for file_path in model_dir.rglob('*'):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        total_size += size
                        file_count += 1
                
                total_mb = total_size / (1024 * 1024)
                print(f"   📊 Current size: {total_mb:.1f} MB ({file_count} files)")
                
                # DialoGPT-medium is approximately 1.5GB when complete
                expected_size_mb = 1500
                progress = (total_mb / expected_size_mb) * 100
                
                if progress < 5:
                    print(f"   📥 Download just started ({progress:.1f}% complete)")
                elif progress < 95:
                    print(f"   ⏳ Download in progress ({progress:.1f}% complete)")
                else:
                    print(f"   ✅ Download appears complete ({progress:.1f}%)")
                    
            else:
                print(f"   ❌ DialoGPT model directory not found")
                print(f"   💡 Model hasn't started downloading yet")
        else:
            print(f"   ❌ Cache directory doesn't exist")
            
    except ImportError:
        print("   ❌ transformers library not available")
    except Exception as e:
        print(f"   ⚠️  Error checking cache: {e}")
    
    # 4. Check disk space
    print("\n4️⃣ Checking disk space...")
    try:
        if sys.platform == "win32":
            import shutil
            total, used, free = shutil.disk_usage("C:\\")
        else:
            statvfs = os.statvfs('/')
            free = statvfs.f_frsize * statvfs.f_available
            total = statvfs.f_frsize * statvfs.f_blocks
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        
        print(f"   💾 Free space: {free_gb:.1f} GB / {total_gb:.1f} GB")
        
        if free_gb < 2:
            print(f"   ⚠️  Warning: Less than 2GB free (model needs ~1.5GB)")
        else:
            print(f"   ✅ Sufficient space available")
            
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
    
    # 5. Provide recommendations
    print("\n5️⃣ Recommendations:")
    
    if not python_processes:
        print("   💡 No Python running - your process may have stopped")
        print("   🔄 Try restarting: python main.py")
    
    # Check if download seems stuck
    try:
        from transformers import TRANSFORMERS_CACHE
        model_dir = Path(TRANSFORMERS_CACHE) / "models--microsoft--DialoGPT-medium"
        if model_dir.exists():
            # Check if any files were modified recently
            latest_mod = 0
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    mod_time = file_path.stat().st_mtime
                    latest_mod = max(latest_mod, mod_time)
            
            if latest_mod > 0:
                minutes_since_update = (time.time() - latest_mod) / 60
                if minutes_since_update > 5:
                    print(f"   ⚠️  No file updates for {minutes_since_update:.1f} minutes")
                    print("   🔄 Download may be stuck - consider restarting")
                else:
                    print(f"   ✅ Files updated {minutes_since_update:.1f} minutes ago")
    except:
        pass
    
    print("\n📋 Next Steps:")
    print("   1. If stuck >10 min: Press Ctrl+C and restart")
    print("   2. Try: python download_model.py (for progress bars)")
    print("   3. Or use demo mode to test UI while downloading")
    print("   4. Monitor with: python monitor_download.py")

def monitor_realtime():
    """Monitor download in real-time"""
    print("\n🔄 Starting real-time monitoring (Press Ctrl+C to stop)...")
    
    try:
        from transformers import TRANSFORMERS_CACHE
        model_dir = Path(TRANSFORMERS_CACHE) / "models--microsoft--DialoGPT-medium"
        
        last_size = 0
        
        while True:
            if model_dir.exists():
                total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                total_mb = total_size / (1024 * 1024)
                
                if total_size > last_size:
                    speed_mbps = (total_size - last_size) / (1024 * 1024) / 2  # MB per 2 seconds
                    print(f"\r📥 Downloaded: {total_mb:.1f} MB ({speed_mbps:.2f} MB/s)    ", end="", flush=True)
                else:
                    print(f"\r⏸️  Downloaded: {total_mb:.1f} MB (no activity)    ", end="", flush=True)
                
                last_size = total_size
            else:
                print(f"\r⏳ Waiting for download to start...    ", end="", flush=True)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n✅ Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    check_download_status()
    
    # Ask if user wants real-time monitoring
    try:
        response = input("\n🔍 Start real-time monitoring? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            monitor_realtime()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
