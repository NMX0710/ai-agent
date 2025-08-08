from langchain.tools import tool
from ..settings import FILE_SAVE_DIR
from reportlab.pdfgen import canvas
import os

@tool(description="Generate a PDF with the given file name and content")
def generate_pdf(file_name: str, content: str) -> str:
    """Generate a PDF with the given file name and content."""
    pdf_dir = os.path.join(FILE_SAVE_DIR, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    file_path = os.path.join(pdf_dir, file_name)

    try:
        c = canvas.Canvas(file_path)
        c.setFont("Helvetica", 12)
        c.drawString(72, 800, content)
        c.save()
        return f"PDF generated successfully at {file_path}"
    except Exception as e:
        return f"Error generating PDF: {str(e)}"
