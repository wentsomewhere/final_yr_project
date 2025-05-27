import os
import argparse
import subprocess
import logging
from pathlib import Path

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'frontend.log')),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Run SRR-GAN React frontend')
    parser.add_argument('--port', type=int, default=3000,
                      help='Port to run the frontend on')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save frontend logs')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                      help='URL of the backend API')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logging.info("Starting React frontend...")
    
    # Set environment variables
    os.environ['REACT_APP_API_URL'] = args.api_url
    
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
    os.chdir(frontend_dir)
    
    # Install dependencies if node_modules doesn't exist
    if not os.path.exists('node_modules'):
        logging.info("Installing frontend dependencies...")
        subprocess.run(['npm', 'install'], check=True)
    
    # Start development server
    logging.info(f"Starting development server on port {args.port}...")
    subprocess.run(
        ['npm', 'start'],
        env=dict(os.environ, PORT=str(args.port)),
        check=True
    )

if __name__ == '__main__':
    main() 