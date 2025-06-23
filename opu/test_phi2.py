#!/usr/bin/env python3
"""
Simple Phi-2 test script to verify the model is working
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_phi2():
    print("🧪 Phi-2 Standalone Test")
    print("=" * 30)
    
    try:
        print("📄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
        print("🧠 Loading model...")
        
        # Force CPU mode for stability
        device = "cpu"
        torch_dtype = torch.float32
        print(f"🖥️ Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Move to device
        model = model.to(device)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded successfully on {device}!")
        print()
        
        # Test prompts
        test_prompts = [
            "Hi Phi, how are you today?",
            "What is 2 + 2?",
            "Tell me a fun fact about pandas.",
            "Hello, can you introduce yourself?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"🧪 Test {i}: '{prompt}'")
            print("⏳ Generating response...")
            
            start_time = time.time()
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with minimal parameters for speed
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,     # Short responses for testing
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the new part
            response = full_response[len(prompt):].strip()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Response ({duration:.1f}s): {response}")
            print(f"📊 Full output: {full_response}")
            print("-" * 50)
            
            # Break if taking too long
            if duration > 60:
                print("⚠️ Response taking too long, stopping tests")
                break
        
        print("\n🎉 Phi-2 is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test():
    """Ultra-simple test with minimal parameters"""
    print("\n🏃 Quick Test Mode")
    print("-" * 20)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print(f"Input: {prompt}")
        print("Generating...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.8,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {response}")
        
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Phi-2 Model Test Suite")
    print("=" * 40)
    
    # Check if model files exist
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        print("✅ Model files found")
    except Exception as e:
        print(f"❌ Model files not found: {e}")
        print("💡 Run your main.py first to download the model")
        exit(1)
    
    print("\nChoose test mode:")
    print("1. Full test (4 prompts, detailed)")
    print("2. Quick test (1 prompt, minimal)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = test_phi2()
        elif choice == "2":
            success = quick_test()
        else:
            print("Invalid choice, running quick test...")
            success = quick_test()
        
        if success:
            print("\n🎯 Phi-2 is working! The issue is likely in the persona prompt or timeout.")
            print("💡 Try reducing max_new_tokens in your main.py to 20-30")
        else:
            print("\n❌ Phi-2 has issues. Check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")