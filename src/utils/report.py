import os
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        ))
    
    def _create_comparison_image(self, original: Image.Image, enhanced: Image.Image, 
                               attention_map: Image.Image = None) -> Image.Image:
        """Create a side-by-side comparison image."""
        # Resize images to same height
        height = min(original.height, enhanced.height)
        width = original.width + enhanced.width
        if attention_map:
            width += attention_map.width
            
        comparison = Image.new('RGB', (width, height))
        
        # Paste images
        comparison.paste(original, (0, 0))
        comparison.paste(enhanced, (original.width, 0))
        if attention_map:
            comparison.paste(attention_map, (original.width + enhanced.width, 0))
            
        return comparison
    
    def _create_metrics_table(self, metrics: Dict[str, float]) -> Table:
        """Create a table of metrics."""
        data = [['Metric', 'Value']]
        for metric, value in metrics.items():
            data.append([metric, f'{value:.4f}'])
            
        table = Table(data, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table
    
    def generate_report(self, 
                       original_image: Image.Image,
                       enhanced_image: Image.Image,
                       original_text: str,
                       enhanced_text: str,
                       metrics: Dict[str, float],
                       attention_map: Image.Image = None,
                       language: str = 'en') -> str:
        """Generate a PDF report with before/after comparisons and metrics."""
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'srrgan_report_{timestamp}.pdf'
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        story = []
        
        # Title
        story.append(Paragraph('SRR-GAN Enhancement Report', self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Language information
        story.append(Paragraph(f'Language: {language.upper()}', self.styles['CustomHeading']))
        story.append(Spacer(1, 12))
        
        # Image comparison
        story.append(Paragraph('Image Comparison', self.styles['CustomHeading']))
        comparison = self._create_comparison_image(original_image, enhanced_image, attention_map)
        comparison_path = os.path.join(self.output_dir, f'comparison_{timestamp}.png')
        comparison.save(comparison_path)
        story.append(RLImage(comparison_path, width=6*inch, height=3*inch))
        story.append(Spacer(1, 12))
        
        # Text comparison
        story.append(Paragraph('Text Recognition Results', self.styles['CustomHeading']))
        text_data = [
            ['Original Text', original_text],
            ['Enhanced Text', enhanced_text]
        ]
        text_table = Table(text_data, colWidths=[2*inch, 4*inch])
        text_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(text_table)
        story.append(Spacer(1, 12))
        
        # Metrics
        story.append(Paragraph('Performance Metrics', self.styles['CustomHeading']))
        story.append(self._create_metrics_table(metrics))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary files
        os.remove(comparison_path)
        
        return filepath 