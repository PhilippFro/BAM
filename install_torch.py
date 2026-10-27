"""Install the PyTorch requirements into the Python environment running this file."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-indexes", action="store_true",
        help="Use public package indexes without inherited pip configuration (fixes stale extra-index URLs)",
    )
    parser.add_argument(
        "--skip-torch", action="store_true",
        help="Skip the CPU-wheel installation step; use this after installing your chosen CUDA build",
    )
    parser.add_argument("--cuda", action="store_true",
                        help="Install PyTorch 2.14.1+cu130 instead of the CPU build and verify CUDA")
    parser.add_argument("--smoke-test", action="store_true", help="Run five training updates after installation (CUDA with --cuda, otherwise CPU)")
    args = parser.parse_args(argv)
    if args.cuda and args.skip_torch:
        parser.error("--cuda and --skip-torch cannot be combined")
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required; Python 3.12 is the tested environment")
    env = os.environ.copy()
    if args.clean_indexes:
        # Only affects child processes, never the user's shell or pip config files.
        for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS",
                    "PIP_NO_INDEX", "PIP_TRUSTED_HOST"):
            env.pop(key, None)
        env["PIP_CONFIG_FILE"] = os.devnull
    pip = [sys.executable, "-m", "pip"]
    if not args.skip_torch:
        build = "cu130" if args.cuda else "cpu"
        requirement = "torch==2.14.1+cu130" if args.cuda else "torch>=2.6,<3"
        subprocess.run(
            pip + ["install", requirement, "--index-url", f"https://download.pytorch.org/whl/{build}"],
            env=env, check=True, cwd=ROOT,
        )
    install = pip + ["install", "-r", str(ROOT / "requirements-torch.txt")]
    if args.clean_indexes:
        install += ["--index-url", "https://pypi.org/simple"]
    subprocess.run(install, env=env, check=True, cwd=ROOT)
    subprocess.run(pip + ["check"], env=env, check=True, cwd=ROOT)
    # Package metadata can be valid while the native loader finds an incompatible
    # libtorch from a container's system Python. Exercise the runtime as well.
    runtime_check = subprocess.run(
        [sys.executable, "-c",
         "import sys, torch; "
         "print('Python:', sys.executable, flush=True); "
         "print('PyTorch:', torch.__version__, torch.__file__, flush=True); "
         "x = torch.tensor([2.0], requires_grad=True); "
         "x.square().sum().backward(); "
         "assert x.grad.item() == 4.0; "
         "print('PyTorch CPU import and backward check passed', flush=True)"],
        env=env, cwd=ROOT,
    )
    if runtime_check.returncode:
        print(
            "Packages are installed, but the PyTorch runtime check failed. "
            "If the traceback names another Python installation's libtorch, "
            "see the native-library conflict section in README.md. "
            "--clean-indexes only controls pip downloads; it does not isolate shared libraries.",
            file=sys.stderr,
        )
        raise SystemExit(runtime_check.returncode)
    if args.cuda:
        subprocess.run(
            [sys.executable, "-c",
             "import torch; "
             "assert torch.cuda.is_available(), 'CUDA build installed, but no GPU is visible'; "
             "print('CUDA runtime:', torch.version.cuda, 'visible GPUs:', torch.cuda.device_count(), flush=True); "
             "x = torch.tensor([2.0], device='cuda', requires_grad=True); "
             "x.square().sum().backward(); "
             "assert x.grad.item() == 4.0; "
             "print('CUDA backward check passed', flush=True)"],
            env=env, check=True, cwd=ROOT,
        )
    print(f"Dependencies installed and runtime verified using {sys.executable}", flush=True)
    if args.smoke_test:
        subprocess.run([sys.executable, str(ROOT / "train_joint_torch.py"), "--smoke-test",
                        "--device", "cuda" if args.cuda else "cpu"],
                       env=env, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
