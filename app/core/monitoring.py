"""
Advanced monitoring and health check utilities
"""

import psutil
import os
import time
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, ConfigDict


class SystemHealth(BaseModel):
    """System health status model"""

    status: str
    timestamp: datetime
    uptime: float
    memory_usage: Dict[str, Any]
    disk_usage: Dict[str, Any]
    cpu_usage: float
    active_processes: int

    model_config = ConfigDict()


class HealthChecker:
    """Advanced health checking with system metrics"""

    def __init__(self):
        self.start_time = time.time()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percentage": memory.percent,
        }

    def get_disk_info(self) -> Dict[str, Any]:
        """Get disk usage information"""
        disk = psutil.disk_usage("/")
        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percentage": (disk.used / disk.total) * 100,
        }

    def get_cpu_info(self) -> float:
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)

    def get_process_count(self) -> int:
        """Get number of active processes"""
        return len(psutil.pids())

    def check_temp_directory_space(self, temp_dir: str = "/tmp") -> Dict[str, Any]:
        """Check if temp directory has enough space"""
        try:
            if os.path.exists(temp_dir):
                disk = psutil.disk_usage(temp_dir)
                free_gb = disk.free / (1024**3)
                return {
                    "path": temp_dir,
                    "free_space_gb": round(free_gb, 2),
                    "sufficient": free_gb > 1.0,  # At least 1GB free
                }
        except Exception:
            pass

        return {"path": temp_dir, "free_space_gb": 0, "sufficient": False}

    def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        uptime = time.time() - self.start_time
        memory = self.get_memory_info()
        disk = self.get_disk_info()
        cpu = self.get_cpu_info()
        processes = self.get_process_count()

        # Determine overall status
        status = "healthy"
        if memory["percentage"] > 90 or disk["percentage"] > 95 or cpu > 95:
            status = "unhealthy"
        elif memory["percentage"] > 80 or disk["percentage"] > 85 or cpu > 80:
            status = "warning"

        return SystemHealth(
            status=status,
            timestamp=datetime.now(),
            uptime=uptime,
            memory_usage=memory,
            disk_usage=disk,
            cpu_usage=cpu,
            active_processes=processes,
        )


# Global health checker instance
health_checker = HealthChecker()
