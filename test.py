# simple_test.py - Test Gemini directly
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    exit()

print(f"🔑 Using API key: {api_key[:10]}...")

genai.configure(api_key=api_key)

# Test different model names
models_to_test = [
    "gemini-1.5-flash",
    "gemini-1.5-pro", 
    "gemini-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro"
]

for model_name in models_to_test:
    try:
        print(f"\n🧪 Testing model: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello")
        print(f"✅ SUCCESS: {model_name}")
        print(f"Response: {response.text}")
        break
    except Exception as e:
        print(f"❌ FAILED: {model_name} - {str(e)}")

# List available models
print("\n📋 Available models:")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")