#!/usr/bin/env python3
"""
Test Smart Models Installation
Verifies that all smart models are working correctly
"""

import os
import sys
from pathlib import Path

def test_anthropic():
    """Test Claude 3.5 Sonnet"""
    try:
        import anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in environment")
            return False
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Test with a simple message
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say 'Claude 3.5 Sonnet is working!' in exactly those words."}
            ]
        )
        
        response = message.content[0].text.strip()
        if "Claude 3.5 Sonnet is working!" in response:
            print("✅ Claude 3.5 Sonnet is working correctly")
            return True
        else:
            print(f"❌ Claude response unexpected: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Claude 3.5 Sonnet test failed: {e}")
        return False

def test_openai():
    """Test OpenAI GPT-4o"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        client = OpenAI(api_key=api_key)
        
        # Test with a simple message
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Say 'GPT-4o is working!' in exactly those words."}
            ],
            max_tokens=50
        )
        
        response = chat.choices[0].message.content.strip()
        if "GPT-4o is working!" in response:
            print("✅ GPT-4o is working correctly")
            return True
        else:
            print(f"❌ GPT-4o response unexpected: {response}")
            return False
            
    except Exception as e:
        print(f"❌ GPT-4o test failed: {e}")
        return False

def test_marker():
    """Test Marker PDF processing"""
    try:
        import subprocess
        
        # Check if marker command is available
        result = subprocess.run(["marker", "--help"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Marker is installed and available")
            return True
        else:
            print("❌ Marker command not found")
            return False
            
    except Exception as e:
        print(f"❌ Marker test failed: {e}")
        return False

def test_surya():
    """Test Surya OCR"""
    try:
        import surya
        from surya import settings
        
        print("✅ Surya is installed and importable")
        print(f"   Surya version: {surya.__version__ if hasattr(surya, '__version__') else 'unknown'}")
        return True
        
    except ImportError:
        print("❌ Surya not installed or not importable")
        return False
    except Exception as e:
        print(f"❌ Surya test failed: {e}")
        return False

def test_basic_dependencies():
    """Test basic dependencies"""
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy")
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} is available")
        except ImportError:
            print(f"❌ {name} is not installed")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🧪 Testing Smart Models Installation")
    print("=" * 40)
    
    tests = [
        ("Basic Dependencies", test_basic_dependencies),
        ("Claude 3.5 Sonnet", test_anthropic),
        ("GPT-4o", test_openai),
        ("Marker PDF", test_marker),
        ("Surya OCR", test_surya),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        results[test_name] = test_func()
    
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    print("=" * 40)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Smart models are ready to use.")
        print("\nNext steps:")
        print("1. Run: python smart_models_api.py")
        print("2. Test at: http://localhost:8000/health")
    else:
        print("\n⚠️  Some tests failed. Check the installation:")
        print("1. Run: python install_smart_models.py")
        print("2. Set up API keys in .env file")
        print("3. Run this test again")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
