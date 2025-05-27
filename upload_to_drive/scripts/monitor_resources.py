import psutil
import time
import logging
from datetime import datetime
import os
import shutil

def setup_logging():
    """Setup logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'system_monitor.log')),
            logging.StreamHandler()
        ]
    )

def cleanup_old_checkpoints(checkpoint_dir, max_checkpoints=5):
    """Clean up old checkpoints to free up disk space."""
    try:
        if not os.path.exists(checkpoint_dir):
            return
            
        checkpoints = []
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth') and not file.startswith('best_model'):
                path = os.path.join(checkpoint_dir, file)
                checkpoints.append((path, os.path.getmtime(path)))
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: x[1])
        
        # Remove oldest checkpoints if we have more than max_checkpoints
        while len(checkpoints) > max_checkpoints:
            path, _ = checkpoints.pop(0)
            os.remove(path)
            logging.info(f"Removed old checkpoint: {path}")
            
    except Exception as e:
        logging.error(f"Error cleaning up checkpoints: {str(e)}")

def monitor_resources(interval=60):
    """Monitor system resources at specified intervals."""
    setup_logging()
    logging.info("Starting system resource monitoring...")
    
    try:
        while True:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024 * 1024 * 1024)  # Convert to GB
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used = disk.used / (1024 * 1024 * 1024)  # Convert to GB
            
            # Log metrics
            logging.info(
                f"CPU: {cpu_percent}% | "
                f"Memory: {memory_percent}% ({memory_used:.2f} GB) | "
                f"Disk: {disk_percent}% ({disk_used:.2f} GB)"
            )
            
            # Check for critical conditions
            if memory_percent > 85:
                logging.warning("High memory usage detected! Consider reducing batch size.")
            if disk_percent > 85:
                logging.warning("High disk usage detected! Cleaning up old checkpoints...")
                cleanup_old_checkpoints('models')
            if cpu_percent > 85:
                logging.warning("High CPU usage detected! System may become unresponsive.")
            
            # Additional warnings for very critical conditions
            if disk_percent > 95:
                logging.critical("CRITICAL: Disk space almost full! Training may be interrupted.")
            if memory_percent > 95:
                logging.critical("CRITICAL: Memory almost full! System may crash.")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Error in monitoring: {str(e)}")

if __name__ == "__main__":
    monitor_resources() 