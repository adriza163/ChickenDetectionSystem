#!/usr/bin/env python3
"""
Script untuk install dependencies yang kompatibel
"""

import subprocess
import sys

def install_package(package):
    """Install package menggunakan pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} berhasil diinstall")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing {package}: {e}")
        return False
    return True

def main():
    """Install semua dependencies"""
    
    # Uninstall gradio versi lama jika ada
    print("🔄 Uninstalling existing gradio...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "gradio", "-y"])
    except:
        pass
    
    # List dependencies dengan versi yang kompatibel
    packages = [
        "gradio==3.50.2",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0", 
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0"
    ]
    
    print("📦 Installing dependencies...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n🎉 Installation complete: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("\n✨ Semua dependencies berhasil diinstall!")
        print("Sekarang Anda bisa menjalankan: python app_gradio.py")
    else:
        print("\n⚠️ Beberapa packages gagal diinstall. Coba install manual:")
        print("pip install gradio==3.50.2")

if __name__ == "__main__":
    main()