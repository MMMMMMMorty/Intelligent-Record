#!/usr/bin/env python3
"""
Patch qwen_asr to add max_model_len and max_num_seqs support
"""
import os
import re
import sys

# Try multiple possible installation paths
possible_paths = [
    '/opt/conda/lib/python3.10/site-packages/qwen_asr/model.py',
    '/usr/local/lib/python3.10/site-packages/qwen_asr/model.py',
    '/opt/conda/lib/python3.10/dist-packages/qwen_asr/model.py',
]

# Also try to detect from environment
try:
    import qwen_asr
    detected_path = os.path.join(os.path.dirname(qwen_asr.__file__), 'model.py')
    if detected_path not in possible_paths:
        possible_paths.insert(0, detected_path)
except Exception:
    pass  # Ignore import errors during build

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if not model_path:
    print(f'File not found. Tried: {possible_paths}')
    sys.exit(1)

print(f'Found qwen_asr model at: {model_path}')

with open(model_path, 'r') as f:
    content = f.read()

# Check if already patched
if 'max_model_len' in content:
    print('Already patched')
    sys.exit(0)

# Pattern to find gpu_memory_utilization in LLM method signature
pattern = r'(gpu_memory_utilization:\s*float\s*=\s*0\.9)'
replacement = r'\1, max_model_len: int = 256, max_num_seqs: int = 1'
content = re.sub(pattern, replacement, content)

# Now add these parameters to the vLLM init
# Look for gpu_memory_utilization=gpu_memory_utilization
pattern2 = r'(gpu_memory_utilization=gpu_memory_utilization)'
replacement2 = r'\1, max_model_len=max_model_len, max_num_seqs=max_num_seqs'
content = re.sub(pattern2, replacement2, content)

with open(model_path, 'w') as f:
    f.write(content)

print(f'Successfully patched {model_path}')
