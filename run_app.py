import subprocess, time, sys

# 1) Launch Streamlit
streamlit_proc = subprocess.Popen(["streamlit", "run", "app.py"])
time.sleep(5)  # give Streamlit time to start

# 2) Open a LocalTunnel on port 8501
tunnel_proc = subprocess.Popen(["lt", "--port", "8501"])
print("LocalTunnel is now forwarding → http://localhost:8501")

# 3) Keep alive until you hit Ctrl+C
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down…")
    streamlit_proc.terminate()
    tunnel_proc.terminate()
    sys.exit(0)