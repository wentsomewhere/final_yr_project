import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from typing import List, Dict
import json
from datetime import datetime

from ..models.srrgan import SRRGAN
from ..models.ocr import OCRModule
from ..evaluation.metrics import MetricsCalculator

app = FastAPI(title="SRR-GAN API",
             description="Super-Resolution based Recognition with GAN for Low-Resolved Text Images",
             version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRRGAN().to(device)
ocr = OCRModule(device=device)

# Load pre-trained weights
MODEL_PATH = os.getenv('MODEL_PATH', 'checkpoints/best_model.pt')
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
else:
    print(f"Warning: Model checkpoint not found at {MODEL_PATH}")

# Initialize metrics calculator
metrics_calculator = MetricsCalculator()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((256, 256))
    
    # Convert to tensor and normalize
    image = torch.from_numpy(np.array(image)).float()
    image = image.permute(2, 0, 1) / 255.0
    image = image.unsqueeze(0)
    
    return image

def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """Convert model output tensor to PIL Image."""
    # Convert to numpy and denormalize
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """Enhance a low-resolution text image."""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Generate enhanced image
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
        
        # Convert to PIL Image
        enhanced_image = postprocess_image(enhanced_tensor)
        
        # Save enhanced image
        output_buffer = io.BytesIO()
        enhanced_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        # Get OCR results
        ocr_result = ocr.decode_predictions(ocr.predict(enhanced_tensor))
        
        return {
            "status": "success",
            "message": "Image enhanced successfully",
            "ocr_text": ocr_result,
            "enhanced_image": output_buffer.getvalue()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_enhance")
async def batch_enhance_images(files: List[UploadFile] = File(...)):
    """Enhance multiple low-resolution text images."""
    results = []
    
    for file in files:
        try:
            # Process each image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Preprocess image
            input_tensor = preprocess_image(image)
            input_tensor = input_tensor.to(device)
            
            # Generate enhanced image
            with torch.no_grad():
                enhanced_tensor = model(input_tensor)
            
            # Convert to PIL Image
            enhanced_image = postprocess_image(enhanced_tensor)
            
            # Save enhanced image
            output_buffer = io.BytesIO()
            enhanced_image.save(output_buffer, format="PNG")
            output_buffer.seek(0)
            
            # Get OCR results
            ocr_result = ocr.decode_predictions(ocr.predict(enhanced_tensor))
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "ocr_text": ocr_result,
                "enhanced_image": output_buffer.getvalue()
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return results

@app.get("/metrics")
async def get_metrics():
    """Get current model performance metrics."""
    return metrics_calculator.get_metrics_summary()

@app.post("/evaluate")
async def evaluate_model(file: UploadFile = File(...), ground_truth: str = None):
    """Evaluate model performance on a single image."""
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Generate enhanced image
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
        
        # Get OCR results
        ocr_result = ocr.decode_predictions(ocr.predict(enhanced_tensor))
        
        # Calculate metrics if ground truth is provided
        metrics = {}
        if ground_truth:
            metrics = metrics_calculator.calculate_ocr_metrics(
                [ocr_result], [ground_truth]
            )
        
        return {
            "status": "success",
            "ocr_text": ocr_result,
            "metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": os.path.exists(MODEL_PATH)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 