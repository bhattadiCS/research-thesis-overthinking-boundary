# =============================================================================
# Google Colab SSH Setup with VS Code Disconnect Auto-Shutdown
# =============================================================================
# Saves GPU credits by detecting when VS Code disconnects from the SSH tunnel
# and automatically de-allocating the Colab VM.
# =============================================================================

# 0. Clean up any corrupted VS Code server installations
import subprocess, os, time, shutil

# Remove corrupted vscode-server if it exists
vscode_server_path = os.path.expanduser("~/.vscode-server")
if os.path.exists(vscode_server_path):
    shutil.rmtree(vscode_server_path, ignore_errors=True)
    print("🧹 Cleaned up old .vscode-server installation")

# 1. Install colab-ssh for the tunnel
subprocess.run(["pip", "install", "-q", "colab_ssh", "--upgrade"], check=True)

# 2. Setup the SSH tunnel (Cloudflare)
from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared

# Set your password here (used for the SSH connection)
SSH_PASSWORD = "machineLearning123$"

# 3. Initialize your repository securely
repo_url = "https://github.com/bhattadiCS/research-thesis-overthinking-boundary.git"
init_git_cloudflared(repo_url)

# 4. Launch the SSH tunnel
launch_ssh_cloudflared(password=SSH_PASSWORD)

# 5. Install your ResearchThesis requirements
subprocess.run(
    ["pip", "install", "-q", "-r",
     "/content/research-thesis-overthinking-boundary/requirements-colab.txt"],
    check=True,
)
subprocess.run(["nvidia-smi"])

# =============================================================================
# 6. Smart Keep-Alive with VS Code Disconnect Detection
# =============================================================================
print("\n" + "=" * 60)
print("✅ SSH tunnel is fully active.")
print("🔍 Monitoring for VS Code disconnection...")
print("   - Will auto-shutdown if VS Code disconnects")
print("   - Or create /content/SHUTDOWN_COLAB.txt to stop manually")
print("=" * 60 + "\n")

SHUTDOWN_TRIGGER_FILE = "/content/SHUTDOWN_COLAB.txt"
CHECK_INTERVAL_SECONDS = 30          # How often to check (seconds)
GRACE_PERIOD_MINUTES = 5             # Wait this long for initial VS Code connection
DISCONNECT_THRESHOLD_CHECKS = 3     # Must see 0 connections N times in a row to shutdown
                                     # (avoids false positives from brief reconnects)


def count_ssh_sessions():
    """
    Count active SSH client sessions by looking for sshd child processes.
    The main sshd daemon always runs (1 process). Each VS Code / terminal
    connection spawns additional sshd children. So active = total - 1.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-c", "sshd"],
            capture_output=True, text=True
        )
        total = int(result.stdout.strip()) if result.returncode == 0 else 0
        # Subtract the 1 root sshd daemon process
        return max(0, total - 1)
    except Exception:
        return -1  # Error state, don't act on it


def vscode_server_is_running():
    """Check if a VS Code server process is alive (backup signal)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vscode-server|code-server"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


# Clean up leftover shutdown trigger from a previous run
if os.path.exists(SHUTDOWN_TRIGGER_FILE):
    os.remove(SHUTDOWN_TRIGGER_FILE)

# --- Phase 1: Grace period — wait for VS Code to connect ---
grace_end = time.time() + (GRACE_PERIOD_MINUTES * 60)
vs_code_connected_once = False

print(f"⏳ Grace period: waiting up to {GRACE_PERIOD_MINUTES} min for VS Code to connect...")

while time.time() < grace_end:
    if os.path.exists(SHUTDOWN_TRIGGER_FILE):
        break
    sessions = count_ssh_sessions()
    if sessions > 0 or vscode_server_is_running():
        vs_code_connected_once = True
        print(f"🔗 VS Code connected! (SSH sessions: {sessions})")
        break
    time.sleep(CHECK_INTERVAL_SECONDS)

if not vs_code_connected_once and not os.path.exists(SHUTDOWN_TRIGGER_FILE):
    print(f"⚠️  No VS Code connection detected within {GRACE_PERIOD_MINUTES} min.")
    print("   Continuing anyway — will still watch for disconnects.")

# --- Phase 2: Monitor for disconnection ---
consecutive_disconnects = 0

while not os.path.exists(SHUTDOWN_TRIGGER_FILE):
    time.sleep(CHECK_INTERVAL_SECONDS)

    sessions = count_ssh_sessions()
    vscode_alive = vscode_server_is_running()

    if sessions > 0 or vscode_alive:
        # VS Code is connected — reset counter
        if consecutive_disconnects > 0:
            print(f"🔗 VS Code reconnected (sessions: {sessions})")
        consecutive_disconnects = 0
        vs_code_connected_once = True
    else:
        # No connection detected
        if vs_code_connected_once:
            consecutive_disconnects += 1
            remaining = DISCONNECT_THRESHOLD_CHECKS - consecutive_disconnects
            print(
                f"⚠️  No SSH sessions detected "
                f"({consecutive_disconnects}/{DISCONNECT_THRESHOLD_CHECKS}). "
                f"Auto-shutdown in ~{remaining * CHECK_INTERVAL_SECONDS}s if no reconnect."
            )
            if consecutive_disconnects >= DISCONNECT_THRESHOLD_CHECKS:
                print("\n🛑 VS Code appears disconnected — triggering auto-shutdown!")
                break

# --- Shutdown ---
reason = (
    "Manual shutdown trigger file detected"
    if os.path.exists(SHUTDOWN_TRIGGER_FILE)
    else "VS Code disconnection detected"
)
print(f"\n{'=' * 60}")
print(f"🛑 Shutdown Reason: {reason}")
print("Gracefully terminating Colab VM to save GPU credits...")
print(f"{'=' * 60}\n")

from google.colab import runtime
runtime.unassign()
