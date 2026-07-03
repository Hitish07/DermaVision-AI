import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def generate_pdf_report(scan_data: dict, output_dir: str = "reports") -> str:
    """
    Generates a hospital-style PDF report for the scan.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"report_{scan_data['scan_id']}.pdf")
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Header - Hospital Branding
    c.setFont("Helvetica-Bold", 24)
    c.setFillColorRGB(0.14, 0.39, 0.92) # Medical Blue #2563EB
    c.drawString(50, height - 50, "DermaVision Clinical Research Center")
    
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(50, height - 70, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 85, f"Patient Scan ID: {scan_data['scan_id']}")
    
    c.setLineWidth(1)
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(50, height - 100, width - 50, height - 100)
    
    # Diagnosis Details
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 130, "AI Diagnostic Summary")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 155, f"Primary Detection: {scan_data['disease']}")
    c.drawString(50, height - 175, f"Model Confidence: {scan_data['confidence'] * 100:.1f}%")
    
    # Risk Badge
    c.setFont("Helvetica-Bold", 12)
    risk = scan_data.get('risk', 'Unknown')
    if risk == 'High': c.setFillColorRGB(0.93, 0.26, 0.26) # Danger Red
    elif risk == 'Medium': c.setFillColorRGB(0.96, 0.61, 0.04) # Warning Amber
    else: c.setFillColorRGB(0.13, 0.77, 0.36) # Success Green
    c.drawString(50, height - 195, f"Clinical Risk Level: {risk}")
    c.setFillColorRGB(0, 0, 0)
    
    # Recommendation
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 225, "Clinical Recommendation:")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 240, scan_data.get('recommendation', 'Routine monitoring advised.'))
    
    # Quality Metrics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(300, height - 130, "Image Quality Assessment")
    c.setFont("Helvetica", 10)
    c.drawString(300, height - 155, f"Overall Score: {scan_data.get('quality_score', 0):.1f}/100")
    c.drawString(300, height - 170, f"Blur: {scan_data.get('blur_score', 'Unknown')}")
    c.drawString(300, height - 185, f"Brightness: {scan_data.get('brightness', 'Unknown')}")
    c.drawString(300, height - 200, f"Noise: {scan_data.get('noise', 'Unknown')}")
    
    # Probability Chart (Text representation)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(300, height - 235, "Top Predictions")
    c.setFont("Helvetica", 10)
    y_pos = height - 250
    if 'top_3' in scan_data:
        for p in scan_data['top_3']:
            c.drawString(300, y_pos, f"{p.get('class_name', 'Unknown')}: {p.get('confidence', 0)*100:.1f}%")
            y_pos -= 15

    # Images
    img_y_start = height - 500
    if 'original_image' in scan_data and os.path.exists(scan_data['original_image']):
        try:
            c.drawString(50, img_y_start + 210, "Original Image:")
            c.drawImage(ImageReader(scan_data['original_image']), 50, img_y_start, width=200, height=200, preserveAspectRatio=True)
        except Exception as e:
            logger.warning(f"Could not add original image to PDF: {e}")
            
    if 'gradcam_image' in scan_data and os.path.exists(scan_data['gradcam_image']):
        try:
            c.drawString(300, img_y_start + 210, "Grad-CAM Overlay (XAI):")
            c.drawImage(ImageReader(scan_data['gradcam_image']), 300, img_y_start, width=200, height=200, preserveAspectRatio=True)
        except Exception as e:
            logger.warning(f"Could not add GradCAM image to PDF: {e}")

    # QR Code Placeholder
    c.setStrokeColorRGB(0,0,0)
    c.rect(480, height - 85, 50, 50)
    c.setFont("Helvetica", 8)
    c.drawString(485, height - 60, "QR Code")

    # Disclaimer Footer
    c.setFont("Helvetica-Bold", 8)
    c.drawString(50, 50, "MEDICAL DISCLAIMER: DermaVision AI is a research and educational platform.")
    c.setFont("Helvetica", 8)
    c.drawString(50, 40, "It is NOT FDA approved and should never be used as a substitute for professional medical diagnosis or advice.")
    
    c.save()
    return pdf_path
