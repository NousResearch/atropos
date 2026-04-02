import requests
import time
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Dummy Actor starting against {args.server}")
    
    # 1. Register with server
    try:
        resp = requests.post(f"{args.server}/register-env", json={
            "max_token_length": 1024,
            "desired_name": "dummy_actor",
            "weight": 1.0,
            "group_size": 1
        })
        resp.raise_for_status()
        data = resp.json()
        print(f"Registered (ID: {data['env_id']})")
        
        # 2. Stay alive
        while True:
            time.sleep(10)
            
    except Exception as e:
        print(f"Dummy Actor failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
