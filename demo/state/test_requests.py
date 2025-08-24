import requests
import time

def test_with_requests():
    try:
        print("Testing with requests library...")
        response = requests.get('http://192.168.8.36:8004/v1/models', timeout=15)
        print(f'Status: {response.status_code}')
        print(f'Content: {response.text[:200]}')
        return True
    except Exception as e:
        print(f'Requests error: {e}')
        return False

def test_with_curl():
    import subprocess
    try:
        print("Testing with curl via subprocess...")
        result = subprocess.run(['curl', '-s', 'http://192.168.8.36:8004/v1/models'], 
                              capture_output=True, text=True, timeout=15)
        print(f'Return code: {result.returncode}')
        if result.stdout:
            print(f'Output: {result.stdout[:200]}')
        if result.stderr:
            print(f'Error: {result.stderr}')
        return result.returncode == 0
    except Exception as e:
        print(f'Curl error: {e}')
        return False

if __name__ == '__main__':
    print("=== Testing connectivity ===")
    requests_ok = test_with_requests()
    print(f'Requests successful: {requests_ok}')
    print()
    curl_ok = test_with_curl()
    print(f'Curl successful: {curl_ok}')