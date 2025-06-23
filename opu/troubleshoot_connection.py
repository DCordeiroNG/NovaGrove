#!/usr/bin/env python3
"""
Troubleshoot AI Persona Chat connectivity issues
"""

import requests
import json
import time
import socket
from urllib.parse import urlparse

def test_backend_connectivity():
    """Test if backend is accessible"""
    print("🔍 Testing Backend Connectivity")
    print("=" * 40)
    
    # Test different URLs
    test_urls = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000"
    ]
    
    for url in test_urls:
        print(f"\n📍 Testing: {url}")
        
        # Test basic connection
        try:
            response = requests.get(f"{url}/", timeout=5)
            print(f"   ✅ Basic connection: {response.status_code}")
            
            # Test personas endpoint
            personas_response = requests.get(f"{url}/api/personas", timeout=5)
            personas = personas_response.json()
            print(f"   ✅ Personas endpoint: {len(personas)} personas loaded")
            
            # Show first persona as test
            if personas:
                first_persona = personas[0]
                print(f"   📋 Sample persona: {first_persona['name']} ({first_persona['title']})")
            
            print(f"   🎯 This URL works! Use: {url}")
            return url
            
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection refused")
        except requests.exceptions.Timeout:
            print(f"   ⏰ Connection timeout")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n❌ No working backend URL found!")
    return None

def test_port_accessibility():
    """Test if port 8000 is accessible"""
    print(f"\n🔌 Testing Port Accessibility")
    print("-" * 30)
    
    for host in ['localhost', '127.0.0.1', '0.0.0.0']:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            
            if host == '0.0.0.0':
                # Can't connect to 0.0.0.0, try 127.0.0.1 instead
                result = sock.connect_ex(('127.0.0.1', 8000))
            else:
                result = sock.connect_ex((host, 8000))
            
            sock.close()
            
            if result == 0:
                print(f"   ✅ {host}:8000 is accessible")
            else:
                print(f"   ❌ {host}:8000 is not accessible")
                
        except Exception as e:
            print(f"   ❌ {host}:8000 error: {e}")

def generate_fixed_frontend():
    """Generate a fixed version of index.html"""
    print(f"\n🔧 Generating Fixed Frontend")
    print("-" * 30)
    
    # Read the current index.html
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the API_BASE with a more robust version
        old_line = "const API_BASE = 'http://localhost:8000';"
        new_line = """// Try multiple possible backend URLs
        const POSSIBLE_BACKENDS = [
            'http://localhost:8000',
            'http://127.0.0.1:8000'
        ];
        
        let API_BASE = 'http://localhost:8000'; // Default"""
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            
            # Also update the loadPersonas function to be more robust
            old_load_function = """async function loadPersonas() {
            try {
                const response = await fetch(`${API_BASE}/api/personas`);
                const personas = await response.json();
                displayPersonas(personas);
            } catch (error) {
                console.error('Error loading personas:', error);
                document.getElementById('personaGrid').innerHTML = '<p>Error loading personas. Make sure the backend is running.</p>';
            }
        }"""
            
            new_load_function = """async function loadPersonas() {
            // Try different backend URLs
            for (const backendUrl of POSSIBLE_BACKENDS) {
                try {
                    console.log(`Trying backend: ${backendUrl}`);
                    const response = await fetch(`${backendUrl}/api/personas`);
                    if (response.ok) {
                        const personas = await response.json();
                        API_BASE = backendUrl; // Use this backend for future requests
                        console.log(`✅ Connected to backend: ${backendUrl}`);
                        displayPersonas(personas);
                        return;
                    }
                } catch (error) {
                    console.log(`❌ Failed to connect to ${backendUrl}:`, error);
                }
            }
            
            // If we get here, no backend worked
            console.error('Could not connect to any backend');
            document.getElementById('personaGrid').innerHTML = `
                <div style="text-align: center; padding: 2rem; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3>❌ Cannot connect to backend</h3>
                    <p>Please check:</p>
                    <ol style="text-align: left; margin: 1rem 0;">
                        <li>Backend is running (python main.py)</li>
                        <li>No firewall blocking port 8000</li>
                        <li>Check browser console (F12) for errors</li>
                    </ol>
                    <button onclick="loadPersonas()" style="padding: 0.5rem 1rem; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        Retry Connection
                    </button>
                </div>
            `;
        }"""
            
            if old_load_function in content:
                content = content.replace(old_load_function, new_load_function)
            
            # Write the fixed version
            with open('index_fixed.html', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("   ✅ Created index_fixed.html with better error handling")
            print("   💡 Try opening index_fixed.html instead of index.html")
            
        else:
            print("   ⚠️  Could not find expected code pattern in index.html")
            
    except Exception as e:
        print(f"   ❌ Error reading/writing files: {e}")

def main():
    """Run all troubleshooting steps"""
    print("🔧 AI Persona Chat - Connection Troubleshooter")
    print("=" * 50)
    
    # Test backend connectivity
    working_url = test_backend_connectivity()
    
    # Test port accessibility
    test_port_accessibility()
    
    # Generate fixed frontend
    generate_fixed_frontend()
    
    print(f"\n" + "=" * 50)
    print("📋 Troubleshooting Summary:")
    
    if working_url:
        print(f"✅ Backend is working at: {working_url}")
        print(f"💡 The issue might be in the frontend connection")
        print(f"🔧 Try using index_fixed.html which has better error handling")
    else:
        print(f"❌ Backend is not accessible")
        print(f"🔄 Try restarting the backend:")
        print(f"   1. Press Ctrl+C to stop current backend")
        print(f"   2. Run: python main.py")
        print(f"   3. Wait for 'Backend ready!' message")
    
    print(f"\n🌐 Browser Debug Steps:")
    print(f"   1. Open browser developer tools (F12)")
    print(f"   2. Go to Console tab")
    print(f"   3. Refresh the page")
    print(f"   4. Look for network errors or CORS issues")
    
    print(f"\n🔍 Manual Test:")
    print(f"   Open this URL directly in browser:")
    print(f"   http://localhost:8000/api/personas")
    print(f"   You should see JSON data with persona information")

if __name__ == "__main__":
    main()
