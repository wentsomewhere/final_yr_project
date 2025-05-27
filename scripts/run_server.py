import os
import argparse
import uvicorn
import logging
from pathlib import Path

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'server.log')),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Run SRR-GAN FastAPI server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to run the server on')
    parser.add_argument('--models_dir', type=str, default='models',
                      help='Directory containing trained models')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save server logs')
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of worker processes')
    parser.add_argument('--reload', action='store_true',
                      help='Enable auto-reload on code changes')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logging.info("Starting FastAPI server...")
    
    # Set environment variables
    os.environ['MODELS_DIR'] = args.models_dir
    
    # Run server
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )

if __name__ == '__main__':
    main() 