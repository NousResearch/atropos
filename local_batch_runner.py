# local_batch_runner.py
import requests
import subprocess
import json
import time
import os

def get_identity_token():
    # Ensure gcloud is in PATH or provide full path
    try:
        token = subprocess.check_output(['gcloud', 'auth', 'print-identity-token'], text=True).strip()
        if not token:
            raise Exception("gcloud auth print-identity-token returned empty token.")
        return token
    except Exception as e:
        print(f"Error getting identity token: {e}")
        print("Ensure 'gcloud' is in your PATH and you are authenticated.")
        raise

# IMPORTANT: Update this URL if your spatial-research-pipeline service URL is different
CLOUD_RUN_URL = "https://spatial-research-pipeline-kjr6u65fta-uc.a.run.app/run-experiment"
TOTAL_EXPERIMENTS_TO_RUN = 800
BATCH_SIZE_PER_CALL = 10 

NUM_CALLS = TOTAL_EXPERIMENTS_TO_RUN // BATCH_SIZE_PER_CALL
# If TOTAL_EXPERIMENTS_TO_RUN is not a multiple of BATCH_SIZE_PER_CALL, add one more call for remainder
if TOTAL_EXPERIMENTS_TO_RUN % BATCH_SIZE_PER_CALL != 0:
    NUM_CALLS +=1


print(f"Will make {NUM_CALLS} calls with batch_size={BATCH_SIZE_PER_CALL} (or less for final call) to run {TOTAL_EXPERIMENTS_TO_RUN} experiments.")

for i in range(NUM_CALLS):
    current_batch_size = BATCH_SIZE_PER_CALL
    if (i == NUM_CALLS - 1) and (TOTAL_EXPERIMENTS_TO_RUN % BATCH_SIZE_PER_CALL != 0) : # last call and there's a remainder
        current_batch_size = TOTAL_EXPERIMENTS_TO_RUN % BATCH_SIZE_PER_CALL

    print(f"Running call {i+1} of {NUM_CALLS} with batch_size={current_batch_size}...")
    try:
        token = get_identity_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {"batch_size": current_batch_size}
        response = requests.post(CLOUD_RUN_URL, headers=headers, json=payload, timeout=1600) # Increased timeout for larger batch
        response.raise_for_status() # Will raise an HTTPError for bad responses (4xx or 5xx)
        print(f"Call {i+1} successful: {response.json()}")
    except requests.exceptions.HTTPError as http_err:
        print(f"Call {i+1} HTTP error: {http_err} - {http_err.response.text if http_err.response else 'No response text'}")
    except Exception as e:
        print(f"Call {i+1} failed: {e}")
    
    if (i + 1) < NUM_CALLS:
        wait_time = 10 # seconds
        print(f"Waiting {wait_time} seconds before next call...")
        time.sleep(wait_time) 

print(f"Finished all calls to run {TOTAL_EXPERIMENTS_TO_RUN} experiments.")
# Ensure no trailing whitespace or problematic characters at EOF 