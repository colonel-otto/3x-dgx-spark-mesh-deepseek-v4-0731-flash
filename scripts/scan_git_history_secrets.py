import subprocess
import re

secret_patterns = [
    (r'(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'Bearer token'),
    (r'ghp_[a-zA-Z0-9]{36}', 'GitHub Personal Access Token'),
    (r'gho_[a-zA-Z0-9]{36}', 'GitHub OAuth Token'),
    (r'hf_[a-zA-Z0-9]{34,}', 'HuggingFace Token'),
    (r'sk-[a-zA-Z0-9]{32,}', 'OpenAI API Key'),
    (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', 'Private Key Block'),
    (r'password\s*[:=]\s*["\'][^"\']{6,}["\']', 'Password string'),
]

commits = subprocess.check_output(['git', 'rev-list', '--all'], text=True).strip().split('\n')
findings = []
for c in commits:
    diff = subprocess.check_output(['git', 'show', c], text=True, errors='replace')
    for line in diff.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            for pat, desc in secret_patterns:
                m = re.search(pat, line)
                if m:
                    matched_str = m.group(0)
                    if any(w in line.lower() for w in ['example', 'placeholder', 'xxx', 'your_token', 'token=<your_token>']):
                        continue
                    findings.append((c, desc, matched_str, line.strip()[:100]))

print(f'Total authentication/secret findings in git history: {len(findings)}')
for c, desc, matched, snippet in findings:
    print(f'Commit {c[:8]} | {desc} | {matched} | {snippet}')
