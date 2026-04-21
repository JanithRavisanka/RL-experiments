"""
utils/device_utils.py
─────────────────────
Utility to detect and configure the best available compute device on Mac.

Mac GPU support:
  - Apple Silicon (M1/M2/M3/M4): PyTorch MPS (Metal Performance Shaders)
  - Intel Mac: CPU only (no CUDA)

Stable-Baselines3 accepts device as a string: "cuda", "mps", or "cpu".
"""

import torch
import platform
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def get_device(verbose: bool = True) -> torch.device:
    """
    Return the best available PyTorch device for this Mac.

    Priority: MPS (Apple GPU) > CPU
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        device_str = "mps"
        if verbose:
            _print_device_info("Apple MPS (Metal GPU) ⚡", "green", device_str)
    else:
        device = torch.device("cpu")
        device_str = "cpu"
        if verbose:
            _print_device_info("CPU (No GPU available)", "yellow", device_str)
    return device


def get_device_str(verbose: bool = True) -> str:
    """Return device string for Stable-Baselines3."""
    device = get_device(verbose=verbose)
    return str(device)


def _print_device_info(label: str, color: str, device_str: str):
    """Pretty-print device information."""
    text = Text()
    text.append("  Device  : ", style="bold white")
    text.append(f"{label}\n", style=f"bold {color}")
    text.append("  Platform: ", style="bold white")
    text.append(f"{platform.processor()}\n", style="cyan")
    text.append("  PyTorch : ", style="bold white")
    text.append(f"{torch.__version__}\n", style="cyan")
    text.append("  SB3 key : ", style="bold white")
    text.append(f'device="{device_str}"', style="magenta")

    console.print(Panel(text, title="[bold blue]🖥️  Compute Device", border_style="blue"))


def verify_mps_tensor():
    """Quick smoke-test to verify MPS tensors work."""
    device = get_device(verbose=False)
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x @ y  # matrix multiply on GPU
    return z.device


if __name__ == "__main__":
    dev = get_device()
    console.print(f"\n✅ Test tensor on [bold]{dev}[/bold]: {verify_mps_tensor()}")
