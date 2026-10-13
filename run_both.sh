
#!/bin/bash
set -e

echo "Starting backward simulation..."
python dataRun_m1_back.py

echo "Starting forward simulation..."
python dataRun_m1_forward.py

echo "Both simulations completed."
