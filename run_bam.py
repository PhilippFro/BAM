"""Run Python with this environment's PyTorch libraries and preserved GPU driver paths.

Examples:
    python -E -s run_bam.py train_joint_torch.py --device cuda --smoke-test
    python -E -s run_bam.py -m torch.distributed.run --standalone --nproc-per-node=4 train_joint_torch.py --device cuda

Only child-process settings change. Activate the intended virtual environment
first. The installer can bootstrap an empty environment through this launcher.
This launcher itself imports only the Python standard library.
"""

import os
from pathlib import Path
import re
import sys
import sysconfig


def runtime_environment(environ, torch_lib, executable):
    env = dict(environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"  # also inherited by torchrun workers
    env["PYTHON_EXEC"] = executable  # torchrun must use the same interpreter
    # Prefer this venv's native torch libraries; retain NVIDIA driver paths and
    # other unrelated libraries required by the container's GPU runtime.
    paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
             if p and "/torch/lib" not in p.replace("\\", "/")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(torch_lib), *paths])
    preloads = [p for p in re.split(r"[\s:]+", env.get("LD_PRELOAD", "")) if p]
    preloads = [p for p in preloads if not Path(p).name.startswith(("libtorch", "libc10"))]
    if preloads:
        env["LD_PRELOAD"] = " ".join(preloads)
    else:
        env.pop("LD_PRELOAD", None)
    return env


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        return
    torch_lib = Path(sysconfig.get_path("platlib")) / "torch" / "lib"
    installing = Path(sys.argv[1]).resolve() == Path(__file__).with_name("install_torch.py").resolve()
    if not torch_lib.is_dir() and not installing:
        raise SystemExit("PyTorch is not installed in this Python environment. Activate the intended venv first.")
    env = runtime_environment(os.environ, torch_lib, sys.executable)
    os.execve(sys.executable, [sys.executable, "-E", "-s", *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
