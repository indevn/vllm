@AGENTS.md

Local GPU note:

- This workspace uses a CUDA 13.0 PyTorch build (`torch 2.11.0+cu130`) while
  the system NVIDIA driver reports 570.133.20 / CUDA 12.8. Do not modify
  system drivers or system CUDA components.
- For GPU commands in this repo, use the user-local compatibility libraries
  from this checkout:

```bash
LD_LIBRARY_PATH="$PWD/.conda/cuda-compat:$PWD/.conda/lib:$PWD/.conda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}" \
  ./.conda/bin/python ...
```

- The compat library verified in this workspace is
  `.conda/cuda-compat/libcuda.so.580.95.05`. Keep this as a per-process
  environment prefix only; do not write it into system-wide configuration.
