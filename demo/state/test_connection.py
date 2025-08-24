import httpx
import asyncio

async def test_connection():
    try:
        # 尝试不使用代理
        async with httpx.AsyncClient(timeout=15.0, proxies=None) as client:
            print("Attempting connection...")
            response = await client.get('http://192.168.8.36:8004/v1/models')
            print(f'Status: {response.status_code}')
            print(f'Content: {response.text[:200]}')
            return True
    except Exception as e:
        print(f'Error type: {type(e).__name__}')
        print(f'Error details: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    result = asyncio.run(test_connection())
    print(f'Connection successful: {result}')