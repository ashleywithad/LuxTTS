"""
Example client for LuxTTS OpenAI-Compatible API

This script demonstrates how to use the LuxTTS API in an OpenAI-compatible way.
"""

import requests
import json


def test_tts_api(base_url="http://localhost:8000"):
    """Test the TTS API with a sample request"""

    # Health check
    print("Checking API health...")
    response = requests.get(f"{base_url}/health")
    print(f"Health: {response.json()}")

    # List models
    print("\nListing available models...")
    response = requests.get(f"{base_url}/v1/models")
    models = response.json()
    print(f"Models: {json.dumps(models, indent=2)}")

    # Generate speech (OpenAI-compatible format)
    print("\nGenerating speech...")
    tts_request = {
        "model": "luxtts",
        "input": "Hello, this is a test of the LuxTTS API!",
        "voice": "default",
        "response_format": "wav",
        "speed": 1.0,
        # Optional LuxTTS-specific parameters
        "rms": 0.01,
        "t_shift": 0.9,
        "num_steps": 4
    }

    response = requests.post(
        f"{base_url}/v1/audio/speech",
        json=tts_request,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        # Save audio to file
        output_file = "output_speech.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✓ Audio saved to {output_file}")
        print(f"  File size: {len(response.content)} bytes")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def test_with_curl():
    """Generate a curl command for testing"""
    curl_command = """curl -X POST http://localhost:8000/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "luxtts",
    "input": "Hello, this is a test!",
    "voice": "default",
    "response_format": "wav"
  }' \\
  --output output.wav"""
    print("\nCurl command for testing:")
    print(curl_command)


if __name__ == "__main__":
    print("=" * 60)
    print("LuxTTS OpenAI-Compatible API Client Example")
    print("=" * 60)

    # Check if a custom base URL is provided
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    try:
        test_tts_api(base_url)
        test_with_curl()
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to the API server.")
        print("  Make sure the server is running:")
        print("  python api_server.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
