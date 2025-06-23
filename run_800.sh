#!/bin/bash
echo "Starting 800 experiment batch run..."
for i in {1..800}; do
    if (( i % 50 == 0 )); then
        echo "Progress: $i/800 experiments ($(( i * 100 / 800 ))%)"
        TODAY_DATE_FOR_GCS=$(date -u +'%Y-%m-%d')
        # Use gsutil stat to check if file exists before catting to avoid error messages if file is new
        if gsutil -q stat gs://gen-lang-client-0029379200-research-data/experiments/${TODAY_DATE_FOR_GCS}.jsonl; then
            stored=$(gsutil cat gs://gen-lang-client-0029379200-research-data/experiments/${TODAY_DATE_FOR_GCS}.jsonl 2>/dev/null | wc -l)
            echo "Actually stored in GCS today (${TODAY_DATE_FOR_GCS}.jsonl): $stored experiments."
        else
            echo "No experiment file found for today (${TODAY_DATE_FOR_GCS}.jsonl) yet."
        fi
    fi

    # Get a fresh token for each curl to avoid expiry during the long run
    TOKEN=$(gcloud auth print-identity-token)

    # Make the curl call
    curl -X POST -H "Authorization: Bearer ${TOKEN}" \
         -H "Content-Type: application/json" \
         -d '{"batch_size": 1}' \
         https://spatial-research-pipeline-kjr6u65fta-uc.a.run.app/run-experiment \
         -s -o /dev/null # Silent, discard output

    # Check curl exit status; if non-zero, log an error and continue (or break if preferred)
    CURL_EXIT_CODE=$?
    if [ $CURL_EXIT_CODE -ne 0 ]; then
        echo "Warning: Experiment $i encountered curl error (exit code: $CURL_EXIT_CODE). Check service logs."
    fi

    sleep 0.5
done

echo "800 experiment batch run finished."
echo "Final count for today:"
TODAY_DATE_FOR_GCS=$(date -u +'%Y-%m-%d')
if gsutil -q stat gs://gen-lang-client-0029379200-research-data/experiments/${TODAY_DATE_FOR_GCS}.jsonl; then
    stored=$(gsutil cat gs://gen-lang-client-0029379200-research-data/experiments/${TODAY_DATE_FOR_GCS}.jsonl 2>/dev/null | wc -l)
    echo "Total experiments stored in GCS today (${TODAY_DATE_FOR_GCS}.jsonl): $stored experiments."
else
    echo "No experiment file found for today (${TODAY_DATE_FOR_GCS}.jsonl) at the end of the run."
fi
