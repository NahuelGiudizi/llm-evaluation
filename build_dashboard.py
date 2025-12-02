#!/usr/bin/env python3
"""
Build script for LLM Benchmark Dashboard

Builds the React UI directly to the package static folder.
"""

import subprocess
import sys
from pathlib import Path


def get_npm_command():
    """Get the correct npm command for the current OS"""
    if sys.platform == "win32":
        return "npm.cmd"
    return "npm"


def main():
    # Paths
    root_dir = Path(__file__).parent
    ui_dir = root_dir / "ui"
    src_static = root_dir / "src" / "llm_evaluator" / "dashboard" / "static"
    npm = get_npm_command()
    
    print("🔨 Building LLM Benchmark Dashboard...")
    print(f"   UI source: {ui_dir}")
    print(f"   Target:    {src_static}")
    
    # Check Node.js and npm
    try:
        result = subprocess.run([npm, "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ npm not found. Please install Node.js first.")
            sys.exit(1)
        print(f"   npm version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js first.")
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    result = subprocess.run(
        [npm, "install"],
        cwd=ui_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ npm install failed:\n{result.stderr}")
        sys.exit(1)
    
    # Build UI (Vite outputs directly to src/llm_evaluator/dashboard/static/)
    print("\n🏗️  Building production bundle...")
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=ui_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ Build failed:\n{result.stderr}")
        sys.exit(1)
    
    if not src_static.exists():
        print("❌ Build did not produce static/ directory")
        sys.exit(1)
    
    # Count files
    files = list(src_static.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    print(f"\n✅ Dashboard built successfully!")
    print(f"   Files: {file_count}")
    print(f"   Size:  {total_size / 1024:.1f} KB")
    print(f"   Path:  {src_static}")
    print("\n🚀 Ready for: pip install . or python -m build")


if __name__ == "__main__":
    main()
