import os
import platform

def get_system_metadata():
    """
    Experimental: Projenin çalıştığı ortam bilgilerini toplar.
    Windows tabanlı geliştirmelerde path uyumluluğu için eklenmiştir.
    """
    return {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "project_root": os.getcwd(),
        "env_status": "development-first-start"
    }