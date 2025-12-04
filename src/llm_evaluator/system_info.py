"""
System information collection for reproducible benchmarks
"""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        pass


@dataclass
class SystemInfo:
    """System information for benchmark reproducibility"""

    # Hardware
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    gpu_info: Optional[str]
    gpu_vram_gb: Optional[float]

    # Software
    os_name: str
    os_version: str
    python_version: str
    ollama_version: Optional[str]

    # Runtime
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Convert to markdown format"""
        gpu_line = f"{self.gpu_info}" if self.gpu_info else "None detected"
        if self.gpu_vram_gb:
            gpu_line += f" ({self.gpu_vram_gb:.1f} GB VRAM)"

        # Add note if CPU name suggests more cores than detected
        cpu_line = f"{self.cpu_model}"
        if "16-Core" in self.cpu_model and self.cpu_cores < 16:
            cpu_line += f" (⚠️ Only {self.cpu_cores} cores detected by OS)"

        return f"""
## System Information

### Hardware
- **CPU:** {cpu_line}
- **Cores:** {self.cpu_cores} physical cores, {self.cpu_threads} threads
- **RAM:** {self.ram_total_gb:.1f} GB
- **GPU:** {gpu_line}

### Software
- **OS:** {self.os_name} {self.os_version}
- **Python:** {self.python_version}
- **Ollama:** {self.ollama_version or "Not installed"}

### Runtime
- **Timestamp:** {self.timestamp}
"""


def get_cpu_model() -> str:
    """Get CPU model name"""
    try:
        if platform.system() == "Windows":
            import winreg  # type: ignore[import-not-found]

            key = winreg.OpenKey(  # type: ignore[attr-defined]
                winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined]
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name: str = winreg.QueryValueEx(key, "ProcessorNameString")[0]  # type: ignore[attr-defined]
            winreg.CloseKey(key)  # type: ignore[attr-defined]
            return cpu_name.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            return result.stdout.strip()
    except Exception:
        pass

    return platform.processor() or "Unknown"


def get_gpu_info() -> tuple[Optional[str], Optional[float]]:
    """Get GPU information if available

    Returns:
        (gpu_name, vram_gb) tuple
    """
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split(",")
                gpu_name = parts[0].strip()
                # Extract VRAM in MB and convert to GB
                vram_str = parts[1].strip().replace(" MiB", "")
                vram_gb = float(vram_str) / 1024
                return gpu_name, vram_gb
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Try AMD rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "GPU" in line:
                    return line.strip(), None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # macOS Metal (Apple Silicon)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=5
            )
            if "Apple" in result.stdout:
                return "Apple Silicon (Metal)", None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return None, None


def get_ollama_version() -> Optional[str]:
    """Get Ollama version if installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip().replace("ollama version ", "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def collect_system_info() -> SystemInfo:
    """Collect all system information"""

    # CPU info
    cpu_model = get_cpu_model()
    cpu_cores = psutil.cpu_count(logical=False) or 0  # Physical cores
    cpu_threads = psutil.cpu_count(logical=True) or 0  # Logical processors (threads)

    # RAM info
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024**3)

    # GPU info
    gpu_info, gpu_vram_gb = get_gpu_info()

    # OS info
    os_name = platform.system()
    os_version = platform.release()

    # Python version
    python_version = platform.python_version()

    # Ollama version
    ollama_version = get_ollama_version()

    # Timestamp
    timestamp = datetime.now().isoformat()

    return SystemInfo(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_total_gb=ram_gb,
        gpu_info=gpu_info,
        gpu_vram_gb=gpu_vram_gb,
        os_name=os_name,
        os_version=os_version,
        python_version=python_version,
        ollama_version=ollama_version,
        timestamp=timestamp,
    )


def print_system_info() -> None:
    """Print system information to console"""
    info = collect_system_info()
    print(info.to_markdown())


if __name__ == "__main__":
    print_system_info()
