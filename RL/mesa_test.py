import subprocess

def installCompleteMesaSupport():
    """安装完整的Mesa软件渲染支持"""
    try:
        print("正在安装完整的Mesa支持...")
        
        MESA_PACKAGES = [
            'mesa-utils',
            'mesa-common-dev', 
            'libgl1-mesa-glx',
            'libgl1-mesa-dri',
            'libglapi-mesa',
            'libosmesa6-dev',  # 重要：OSMesa软件渲染
            'libglu1-mesa-dev',
            'freeglut3-dev',
            'mesa-utils-extra',
            'xvfb'  # 虚拟帧缓冲
        ]
        
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        
        for package in MESA_PACKAGES:
            print(f"正在安装 {package}...")
            subprocess.run(['sudo', 'apt', 'install', '-y', package], check=True)
            
        print("✅ Mesa完整支持安装完成！")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装过程中出现错误: {e}")

if __name__ == "__main__":
    installCompleteMesaSupport()