import os
import argparse
import subprocess
import logging
import shutil
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    # Check for Node.js and npm
    try:
        subprocess.run(['node', '--version'], capture_output=True, check=True)
        subprocess.run(['npm', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_deps.append("Node.js and npm")
    
    # Check for Python dependencies
    try:
        import uvicorn
        import fastapi
    except ImportError:
        missing_deps.append("Python dependencies (uvicorn, fastapi)")
    
    if missing_deps:
        logging.error("Missing required dependencies:")
        for dep in missing_deps:
            logging.error(f"- {dep}")
        logging.error("\nPlease install the missing dependencies and try again.")
        logging.error("For Node.js and npm: Download from https://nodejs.org/")
        logging.error("For Python dependencies: Run 'pip install -r requirements.txt'")
        sys.exit(1)

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'build.log')),
            logging.StreamHandler()
        ]
    )

def build_frontend(frontend_dir: str, api_url: str):
    """Build the React frontend for production."""
    logging.info("Building frontend...")
    
    # Set environment variables
    env = dict(os.environ)
    env['REACT_APP_API_URL'] = api_url
    
    # Install dependencies if needed
    if not os.path.exists(os.path.join(frontend_dir, 'node_modules')):
        logging.info("Installing frontend dependencies...")
        try:
            subprocess.run(['npm', 'install'], cwd=frontend_dir, env=env, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install frontend dependencies: {str(e)}")
            raise
        except FileNotFoundError:
            logging.error("npm command not found. Please install Node.js and npm.")
            raise
    
    # Build frontend
    logging.info("Creating production build...")
    try:
        subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to build frontend: {str(e)}")
        raise

def build_backend(backend_dir: str, models_dir: str):
    """Build the FastAPI backend for production."""
    logging.info("Building backend...")
    
    # Create necessary directories
    os.makedirs(models_dir, exist_ok=True)
    
    # Set environment variables
    env = dict(os.environ)
    env['MODELS_DIR'] = models_dir

def create_production_structure(build_dir: str, frontend_dir: str, backend_dir: str):
    """Create the production directory structure."""
    logging.info("Creating production directory structure...")
    
    # Create build directory
    os.makedirs(build_dir, exist_ok=True)
    
    # Copy frontend build
    frontend_build = os.path.join(frontend_dir, 'build')
    if os.path.exists(frontend_build):
        shutil.copytree(frontend_build, os.path.join(build_dir, 'frontend'))
    else:
        logging.error("Frontend build directory not found. Build may have failed.")
        raise FileNotFoundError("Frontend build directory not found")
    
    # Copy backend files
    backend_files = [
        'src',
        'requirements.txt',
        'README.md',
        'LICENSE'
    ]
    for file in backend_files:
        src = os.path.join(backend_dir, file)
        dst = os.path.join(build_dir, file)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    
    # Create production scripts
    scripts_dir = os.path.join(build_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Create production run script
    run_script = os.path.join(scripts_dir, 'run_production.py')
    with open(run_script, 'w') as f:
        f.write('''import os
import subprocess
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logging.info("Starting SRR-GAN in production mode...")
    
    # Start backend server
    backend_process = subprocess.Popen(
        ['uvicorn', 'src.api.app:app', '--host', '0.0.0.0', '--port', '8000'],
        env=dict(os.environ, MODELS_DIR='models')
    )
    
    # Start frontend server
    frontend_process = subprocess.Popen(
        ['serve', '-s', 'frontend', '-l', '3000']
    )
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == '__main__':
    main()
''')
    
    # Create production requirements
    with open(os.path.join(build_dir, 'requirements.prod.txt'), 'w') as f:
        f.write('''uvicorn==0.15.0
fastapi==0.68.1
python-multipart==0.0.5
serve==14.0.1
''')

def main():
    parser = argparse.ArgumentParser(description='Build SRR-GAN for production')
    parser.add_argument('--build_dir', type=str, default='build',
                      help='Directory to store the production build')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                      help='URL of the backend API in production')
    parser.add_argument('--models_dir', type=str, default='models',
                      help='Directory containing trained models')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save build logs')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logging.info("Starting production build...")
    
    # Check dependencies
    check_dependencies()
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    frontend_dir = os.path.join(project_root, 'frontend')
    backend_dir = project_root
    
    try:
        # Build frontend
        build_frontend(frontend_dir, args.api_url)
        
        # Build backend
        build_backend(backend_dir, args.models_dir)
        
        # Create production structure
        create_production_structure(args.build_dir, frontend_dir, backend_dir)
        
        logging.info(f"Production build completed! Output directory: {args.build_dir}")
    except Exception as e:
        logging.error(f"Build failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 