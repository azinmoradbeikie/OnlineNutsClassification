import platform
import psutil
import cpuinfo

def get_hardware_specs():
    """Get detailed hardware specifications"""
    print("\n" + "="*50)
    print("HARDWARE SPECIFICATIONS")
    print("="*50)
    
    # CPU Information
    try:
        cpu_info = cpuinfo.get_cpu_info()
        print(f"CPU: {cpu_info['brand_raw']}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    except:
        print(f"CPU: {platform.processor()}")
    
    # RAM Information
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"RAM: {ram_gb:.1f} GB")
    
    # OS Information
    print(f"OS: {platform.system()} {platform.release()}")
    
    # Python and Library Versions
    print(f"Python: {platform.python_version()}")
    print("="*50 + "\n")