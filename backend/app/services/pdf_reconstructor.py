"""
PDF Reconstructor Service for ResumeForge
Recreates PDFs with optimized text while preserving original layout
Based on the proven algorithm from reference.md
"""

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import structlog

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black, Color
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

logger = structlog.get_logger()


class PDFReconstructor:
    """
    PDF reconstruction service that preserves original layout
    Uses GPT-4o Vision coordinates and ReportLab for precise positioning
    """
    
    def __init__(self):
        """Initialize PDF reconstructor with default fonts and settings"""
        self.temp_dir = Path(tempfile.gettempdir()) / "resumeforge_pdfs"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Default page settings
        self.default_pagesize = letter  # 8.5" x 11"
        self.margin = 0.75 * inch
        
        # Font settings
        self.default_fonts = {
            'normal': ('Helvetica', 11),
            'bold': ('Helvetica-Bold', 11),
            'header': ('Helvetica-Bold', 16),
            'subheader': ('Helvetica-Bold', 14),
            'small': ('Helvetica', 9)
        }
        
        # Try to register better fonts if available
        self._register_fonts()
    
    def _register_fonts(self):
        """Register additional fonts for better typography"""
        try:
            # Try to register common system fonts
            font_paths = [
                '/System/Library/Fonts/Arial.ttf',  # macOS
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
                'C:/Windows/Fonts/arial.ttf'  # Windows
            ]
            
            for font_path in font_paths:
                if Path(font_path).exists():
                    pdfmetrics.registerFont(TTFont('Arial', font_path))
                    self.default_fonts['normal'] = ('Arial', 11)
                    self.default_fonts['bold'] = ('Arial-Bold', 11)
                    break
                    
        except Exception as e:
            logger.warning("Could not register custom fonts, using defaults", error=str(e))
    
    async def create_optimized_resume_pdf(
        self, 
        markdown_content: str, 
        layout_coords: Dict,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Create optimized resume PDF with preserved layout
        
        Args:
            markdown_content: Optimized resume content in markdown
            layout_coords: Layout coordinates from GPT-4o Vision
            output_filename: Optional custom filename
            
        Returns:
            Path: Path to generated PDF file
        """
        try:
            # Generate output path
            if not output_filename:
                output_filename = f"optimized_resume_{hash(markdown_content)}.pdf"
            
            output_path = self.temp_dir / output_filename
            
            # Parse markdown into sections
            sections = self._parse_markdown_sections(markdown_content)
            
            # Create PDF with layout coordinates
            self._create_pdf_with_layout(
                sections=sections,
                layout_coords=layout_coords,
                output_path=output_path,
                document_type="resume"
            )
            
            logger.info("Resume PDF created successfully", 
                       output_path=str(output_path),
                       sections_count=len(sections))
            
            return output_path
            
        except Exception as e:
            logger.error("Resume PDF creation failed", error=str(e))
            raise ValueError(f"PDF creation failed: {str(e)}")
    
    async def create_cover_letter_pdf(
        self, 
        cover_text: str, 
        layout_coords: Dict,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Create cover letter PDF with matching layout style
        
        Args:
            cover_text: Cover letter content
            layout_coords: Layout coordinates for styling consistency
            output_filename: Optional custom filename
            
        Returns:
            Path: Path to generated PDF file
        """
        try:
            if not output_filename:
                output_filename = f"cover_letter_{hash(cover_text)}.pdf"
            
            output_path = self.temp_dir / output_filename
            
            # Parse cover letter content
            sections = self._parse_cover_letter_content(cover_text)
            
            # Create PDF with simplified layout
            self._create_cover_letter_pdf(
                sections=sections,
                layout_coords=layout_coords,
                output_path=output_path
            )
            
            logger.info("Cover letter PDF created successfully", 
                       output_path=str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error("Cover letter PDF creation failed", error=str(e))
            raise ValueError(f"Cover letter PDF creation failed: {str(e)}")
    
    def _parse_markdown_sections(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Parse markdown content into structured sections
        
        Returns:
            List of section dictionaries with type, content, and formatting
        """
        sections = []
        lines = markdown_content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if line.startswith('# '):
                # Main header (name)
                sections.append({
                    'type': 'header',
                    'content': line[2:].strip(),
                    'level': 1
                })
            elif line.startswith('## '):
                # Section headers
                sections.append({
                    'type': 'section_header',
                    'content': line[3:].strip(),
                    'level': 2
                })
                current_section = line[3:].strip().lower()
            elif line.startswith('### '):
                # Subsection headers
                sections.append({
                    'type': 'subsection_header',
                    'content': line[4:].strip(),
                    'level': 3
                })
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet points
                sections.append({
                    'type': 'bullet',
                    'content': line[2:].strip(),
                    'section': current_section
                })
            elif line.startswith('**') and line.endswith('**'):
                # Bold text (job titles, etc.)
                sections.append({
                    'type': 'bold_text',
                    'content': line[2:-2].strip(),
                    'section': current_section
                })
            else:
                # Regular paragraph text
                sections.append({
                    'type': 'paragraph',
                    'content': line,
                    'section': current_section
                })
        
        return sections
    
    def _parse_cover_letter_content(self, cover_text: str) -> List[Dict[str, Any]]:
        """Parse cover letter content into sections"""
        sections = []
        paragraphs = [p.strip() for p in cover_text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if i == 0:
                # First paragraph might be a header
                sections.append({
                    'type': 'cover_header',
                    'content': paragraph
                })
            else:
                sections.append({
                    'type': 'cover_paragraph',
                    'content': paragraph
                })
        
        return sections
    
    def _create_pdf_with_layout(
        self, 
        sections: List[Dict], 
        layout_coords: Dict, 
        output_path: Path,
        document_type: str
    ):
        """
        Create PDF using ReportLab with precise layout positioning
        Based on reference.md make_pdf_from_layout function
        """
        # Get page dimensions from layout coordinates
        page_dims = layout_coords.get('page_dimensions', {})
        page_width = page_dims.get('width', 612)  # Default letter width
        page_height = page_dims.get('height', 792)  # Default letter height
        
        # Create canvas
        c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
        
        # Apply layout coordinates to sections
        self._apply_layout_coordinates(c, sections, layout_coords)
        
        # Finalize PDF
        c.showPage()
        c.save()
    
    def _apply_layout_coordinates(
        self, 
        canvas_obj: canvas.Canvas, 
        sections: List[Dict], 
        layout_coords: Dict
    ):
        """
        Apply GPT-4o Vision layout coordinates to position text precisely
        This is the key innovation from reference.md
        """
        layout_sections = layout_coords.get('sections', [])
        fonts_info = layout_coords.get('fonts', [])
        
        # Create mapping of section types to layout coordinates
        layout_map = {}
        for layout_section in layout_sections:
            section_type = layout_section.get('type', '')
            bbox = layout_section.get('bbox', [])
            if len(bbox) == 4:
                layout_map[section_type] = {
                    'x1': bbox[0], 'y1': bbox[1],
                    'x2': bbox[2], 'y2': bbox[3],
                    'content_type': layout_section.get('content_type', '')
                }
        
        # Apply fonts and styling
        self._match_fonts_and_styling(canvas_obj, fonts_info)
        
        # Position each section according to layout coordinates
        for section in sections:
            section_type = section['type']
            content = section['content']
            
            # Map section types to layout coordinates
            layout_key = self._map_section_to_layout(section_type, section.get('section', ''))
            
            if layout_key in layout_map:
                coords = layout_map[layout_key]
                self._draw_text_in_bbox(canvas_obj, content, coords, section_type)
            else:
                # Fallback positioning if no coordinates found
                self._draw_text_fallback(canvas_obj, content, section_type)
    
    def _map_section_to_layout(self, section_type: str, section_name: str) -> str:
        """Map parsed section types to layout coordinate keys"""
        mapping = {
            'header': 'header',
            'section_header': self._get_section_layout_key(section_name),
            'subsection_header': self._get_section_layout_key(section_name),
            'paragraph': self._get_section_layout_key(section_name),
            'bullet': self._get_section_layout_key(section_name),
            'bold_text': self._get_section_layout_key(section_name)
        }
        return mapping.get(section_type, 'experience')  # Default fallback
    
    def _get_section_layout_key(self, section_name: str) -> str:
        """Map section names to layout coordinate keys"""
        if not section_name:
            return 'experience'
        
        section_lower = section_name.lower()
        if 'summary' in section_lower or 'profile' in section_lower:
            return 'summary'
        elif 'experience' in section_lower or 'work' in section_lower:
            return 'experience'
        elif 'skill' in section_lower:
            return 'skills'
        elif 'education' in section_lower:
            return 'education'
        else:
            return 'experience'  # Default
    
    def _match_fonts_and_styling(self, canvas_obj: canvas.Canvas, fonts_info: List[Dict]):
        """Apply font information from layout analysis"""
        if not fonts_info:
            # Use default font
            canvas_obj.setFont(*self.default_fonts['normal'])
            return
        
        # Use the first font as primary
        primary_font = fonts_info[0]
        font_name = primary_font.get('name', 'Helvetica')
        font_size = primary_font.get('size', 11)
        
        # Map font names to available fonts
        if 'arial' in font_name.lower():
            font_name = 'Arial' if 'Arial' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
        
        canvas_obj.setFont(font_name, font_size)
    
    def _draw_text_in_bbox(
        self, 
        canvas_obj: canvas.Canvas, 
        text: str, 
        coords: Dict, 
        section_type: str
    ):
        """Draw text within specified bounding box coordinates"""
        x1, y1 = coords['x1'], coords['y1']
        x2, y2 = coords['x2'], coords['y2']
        
        # Calculate text position (top-left of bbox)
        text_x = x1
        text_y = y2  # ReportLab uses bottom-left origin, so use y2 for top
        
        # Adjust font size based on section type
        if section_type == 'header':
            canvas_obj.setFont(*self.default_fonts['header'])
        elif section_type == 'section_header':
            canvas_obj.setFont(*self.default_fonts['subheader'])
        elif section_type == 'bold_text':
            canvas_obj.setFont(*self.default_fonts['bold'])
        else:
            canvas_obj.setFont(*self.default_fonts['normal'])
        
        # Handle text wrapping for long content
        max_width = x2 - x1
        wrapped_lines = self._wrap_text(text, max_width, canvas_obj)
        
        # Draw each line
        line_height = 14  # Approximate line height
        for i, line in enumerate(wrapped_lines):
            line_y = text_y - (i * line_height)
            if line_y > y1:  # Only draw if within bbox
                canvas_obj.drawString(text_x, line_y, line)
    
    def _draw_text_fallback(self, canvas_obj: canvas.Canvas, text: str, section_type: str):
        """Fallback text positioning when no coordinates available"""
        # Simple fallback positioning
        y_position = 750 - (len(getattr(self, '_fallback_y_offset', 0)) * 20)
        
        if section_type == 'header':
            canvas_obj.setFont(*self.default_fonts['header'])
            y_position = 750
        elif section_type == 'section_header':
            canvas_obj.setFont(*self.default_fonts['subheader'])
        else:
            canvas_obj.setFont(*self.default_fonts['normal'])
        
        canvas_obj.drawString(50, y_position, text)
        
        # Track y offset for next fallback
        if not hasattr(self, '_fallback_y_offset'):
            self._fallback_y_offset = []
        self._fallback_y_offset.append(text)
    
    def _wrap_text(self, text: str, max_width: float, canvas_obj: canvas.Canvas) -> List[str]:
        """Wrap text to fit within specified width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_width = canvas_obj.stringWidth(test_line)
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word too long - just add it
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _create_cover_letter_pdf(
        self, 
        sections: List[Dict], 
        layout_coords: Dict, 
        output_path: Path
    ):
        """Create cover letter PDF with simplified layout"""
        page_dims = layout_coords.get('page_dimensions', {})
        page_width = page_dims.get('width', 612)
        page_height = page_dims.get('height', 792)
        
        c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
        
        # Simple top-down layout for cover letter
        y_position = page_height - 100  # Start near top
        
        for section in sections:
            content = section['content']
            section_type = section['type']
            
            if section_type == 'cover_header':
                c.setFont(*self.default_fonts['bold'])
                y_position -= 30
            else:
                c.setFont(*self.default_fonts['normal'])
                y_position -= 20
            
            # Wrap text for cover letter
            wrapped_lines = self._wrap_text(content, page_width - 100, c)
            
            for line in wrapped_lines:
                c.drawString(50, y_position, line)
                y_position -= 15
            
            y_position -= 10  # Extra space between sections
        
        c.showPage()
        c.save()


# Singleton instance
pdf_reconstructor = PDFReconstructor()


async def get_pdf_reconstructor() -> PDFReconstructor:
    """Dependency injection for FastAPI"""
    return pdf_reconstructor
