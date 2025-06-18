import subprocess
import time
import sys
import shutil

# 1) Launch Streamlit
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Give Streamlit a moment to spin up
time.sleep(5)

# 2) Only start LocalTunnel if the `lt` binary exists
if shutil.which("lt"):
    subprocess.Popen(["lt", "--port", "8501"])
    print("✅ LocalTunnel forwarding → http://localhost:8501")
else:
    print("⚠️  `lt` not found; skipping LocalTunnel (expected on Cloud)")

# 3) Wait until someone stops us
try:
    streamlit_proc.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down…")
    streamlit_proc.terminate()
    # if we started a tunnel, shut that down too
    if 'tunnel_proc' in locals():
        tunnel_proc.terminate()
    sys.exit(0)
