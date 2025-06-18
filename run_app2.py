from pyngrok import ngrok
import subprocess
import time

# Set your ngrok authtoken
ngrok.set_auth_token("2vnrsVXhhfXwJjnzIREX2ohliFx_5pwVREGVRWV4SWh3UctWm")

# Start Streamlit in a separate process
streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"])

# Wait a moment for Streamlit to start
time.sleep(5)

# Set up ngrok tunnel to Streamlit's default port (8501)
public_url = ngrok.connect(8501)
print(f"Streamlit app is running at: {public_url}")

try:
    # Keep the script running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    streamlit_process.terminate()
    ngrok.kill()