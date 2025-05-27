import os
import subprocess
import logging
import signal
import sys
import time
from pathlib import Path

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'production.log')),
            logging.StreamHandler()
        ]
    )

def run_command(command, env=None):
    """Run a command and return the process."""
    return subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run SRR-GAN in production mode')
    parser.add_argument('--models_dir', type=str, default='models',
                      help='Directory containing trained models')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    parser.add_argument('--api_port', type=int, default=8000,
                      help='Port for the backend API')
    parser.add_argument('--frontend_port', type=int, default=3000,
                      help='Port for the frontend')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logging.info("Starting SRR-GAN in production mode...")
    
    # Create necessary directories
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Set environment variables
    env = dict(os.environ)
    env['MODELS_DIR'] = args.models_dir
    env['REACT_APP_API_URL'] = f'http://localhost:{args.api_port}'
    
    # Start backend server
    logging.info("Starting backend server...")
    backend_process = run_command(
        ['uvicorn', 'src.api.app:app',
         '--host', '0.0.0.0',
         '--port', str(args.api_port),
         '--workers', '4'],
        env=env
    )
    
    # Wait for backend to start
    time.sleep(5)
    
    # Start frontend server
    logging.info("Starting frontend server...")
    frontend_process = run_command(
        ['serve', '-s', 'frontend',
         '-l', str(args.frontend_port)],
        env=env
    )
    
    # Handle process output
    def handle_output(process, prefix):
        for line in process.stdout:
            logging.info(f"{prefix}: {line.strip()}")
        for line in process.stderr:
            logging.error(f"{prefix}: {line.strip()}")
    
    # Start output handling threads
    import threading
    backend_thread = threading.Thread(
        target=handle_output,
        args=(backend_process, "Backend")
    )
    frontend_thread = threading.Thread(
        target=handle_output,
        args=(frontend_process, "Frontend")
    )
    backend_thread.daemon = True
    frontend_thread.daemon = True
    backend_thread.start()
    frontend_thread.start()
    
    # Handle cleanup on exit
    def cleanup(signum, frame):
        logging.info("Shutting down production servers...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(signal.SIGINT, None)

if __name__ == '__main__':
    main() 