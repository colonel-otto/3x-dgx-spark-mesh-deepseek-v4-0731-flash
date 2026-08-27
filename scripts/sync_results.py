import subprocess
import os

fix_cmd = "import os, glob; [os.rename(p, p.replace(chr(13), '')) for p in glob.glob('/home/sparkmain/bench-repo/results/**/*', recursive=True) if chr(13) in p]"
subprocess.run(["ssh", "sparkmain", f"python3 -c \"{fix_cmd}\""], check=True)

directories = [
    ("/home/sparkmain/bench-repo/results/20260827T175824Z-issue25-profile-a", "results/20260827-issue25-profile-a"),
    ("/home/sparkmain/bench-repo/results/20260827T205027Z-issue25-profile-b", "results/20260827-issue25-profile-b"),
    ("/home/sparkmain/bench-repo/results/20260827T215648Z-node-count-131k-15rep", "results/20260827-tp3-131k-15rep"),
]

for remote_dir, local_dir in directories:
    os.makedirs(local_dir, exist_ok=True)
    raw_files = subprocess.check_output(["ssh", "sparkmain", f"ls {remote_dir}"]).decode()
    files = [f.strip() for f in raw_files.splitlines() if f.strip()]
    for f in files:
        dest = os.path.join(local_dir, f)
        subprocess.run(["scp", f"sparkmain:{remote_dir}/{f}", dest], check=True)

print("SUCCESS: All files downloaded.")
