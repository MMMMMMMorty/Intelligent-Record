"""
Test script for Intelligengt Record API
"""

import requests
import json
import sys
import base64

BASE_URL = "http://localhost:8080"
ASR_URL = "http://localhost:8001"


def test_health():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_asr_models():
    """Test ASR engine models endpoint"""
    print("\nTesting ASR models...")
    try:
        response = requests.get(f"{ASR_URL}/v1/models", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_list_transcriptions():
    """Test list transcriptions"""
    print("\nTesting list transcriptions...")
    try:
        response = requests.get(f"{BASE_URL}/api/transcriptions", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_create_transcription():
    """Test create transcription"""
    print("\nTesting create transcription...")
    try:
        data = {
            "title": "Test Recording",
            "text": "这是一段测试文本。",
            "language": "zh"
        }
        response = requests.post(
            f"{BASE_URL}/api/transcriptions",
            json=data,
            timeout=5
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_websocket():
    """Test WebSocket connection"""
    print("\nTesting WebSocket...")
    try:
        import websocket

        ws_url = f"ws://localhost:8080/ws/stream"
        ws = websocket.create_connection(ws_url, timeout=5)

        # Send ping
        ws.send(json.dumps({"type": "ping"}))
        response = ws.recv()
        print(f"Ping response: {response}")

        # Send reset
        ws.send(json.dumps({"type": "reset"}))
        response = ws.recv()
        print(f"Reset response: {response}")

        ws.close()
        print("WebSocket test passed!")
        return True
    except ImportError:
        print("websocket-client not installed. Install with: pip install websocket-client")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("=" * 50)
    print("Intelligengt Record - API Test")
    print("=" * 50)

    tests = [
        ("Health Check", test_health),
        ("ASR Models", test_asr_models),
        ("List Transcriptions", test_list_transcriptions),
        ("Create Transcription", test_create_transcription),
        ("WebSocket", test_websocket),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 50}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
