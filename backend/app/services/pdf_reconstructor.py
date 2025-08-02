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
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image

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

    def markdown_to_pdf_with_layout(
        self,
        markdown_content: str,
        layout_coords: Optional[Dict] = None,
        output_filename: Optional[str] = None,
        document_type: str = "resume",
    ) -> Path:
        """
        Backwards-compatible adapter for legacy callers expecting markdown_to_pdf_with_layout.
        Delegates to the current implementation methods.

        Args:
            markdown_content: Content in Markdown
            layout_coords: Layout coordinates dict; when None or empty, use fallback formatting
            output_filename: Optional output filename
            document_type: "resume" or "cover_letter"

        Returns:
            Path to generated PDF
        """
        try:
            # Normalize layout coords
            layout_coords = layout_coords or {}

            # Choose path based on document type
            if document_type == "resume":
                # If we have layout sections/blocks, prefer reconstruction with layout
                has_layout = bool(layout_coords.get("text_blocks") or layout_coords.get("sections"))
                if has_layout:
                    return self.reconstruct_pdf_with_smoldocling(
                        optimized_markdown=markdown_content,
                        layout_coords=layout_coords,
                        output_filename=output_filename or f"optimized_resume_{hash(markdown_content)}.pdf",
                    )
                # Fallback to clean professional PDF
                return self._create_fallback_pdf(
                    markdown_content=markdown_content,
                    output_filename=output_filename or f"fallback_resume_{hash(markdown_content)}.pdf",
                )

            # Cover letter path uses the professional fallback generator
            return self._create_fallback_pdf(
                markdown_content=markdown_content,
                output_filename=output_filename or f"cover_letter_{hash(markdown_content)}.pdf",
            )

        except Exception as e:
            logger.error("markdown_to_pdf_with_layout failed; using fallback", error=str(e))
            return self._create_fallback_pdf(
                markdown_content=markdown_content,
                output_filename=output_filename or f"fallback_{document_type}_{hash(markdown_content)}.pdf",
            )
    
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

    def pdf_to_text(self, pdf_path: Path) -> str:
        """
        Extract plain text from PDF using PyMuPDF
        Based on reference.md algorithm

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            logger.info("Extracting text from PDF", pdf_path=str(pdf_path))

            # Open PDF document
            doc = fitz.open(pdf_path)

            # Extract text from all pages
            text_content = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    text_content.append(text)

            doc.close()

            # Join all pages with double newlines
            full_text = "\n\n".join(text_content)

            logger.info(
                "PDF text extraction completed",
                pages=len(text_content),
                characters=len(full_text)
            )

            return full_text

        except Exception as e:
            logger.error("Failed to extract text from PDF", error=str(e), pdf_path=str(pdf_path))
            raise

    def first_page_to_png(self, pdf_path: Path, dpi: int = 150) -> Path:
        """
        Convert first page of PDF to PNG image for layout analysis
        Based on reference.md algorithm

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion (default 150)

        Returns:
            Path to the generated PNG image
        """
        try:
            logger.info("Converting first page to PNG", pdf_path=str(pdf_path), dpi=dpi)

            # Convert first page to image
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=1,
                last_page=1,
                fmt='PNG'
            )

            if not images:
                raise ValueError("No images generated from PDF")

            # Save to temporary file
            png_path = self.temp_dir / f"page_{pdf_path.stem}_{dpi}dpi.png"
            images[0].save(png_path, "PNG")

            logger.info(
                "PNG conversion completed",
                png_path=str(png_path),
                image_size=images[0].size
            )

            return png_path

        except Exception as e:
            logger.error("Failed to convert PDF to PNG", error=str(e), pdf_path=str(pdf_path))
            raise

    def reconstruct_pdf_with_smoldocling(
        self,
        optimized_markdown: str,
        layout_coords: Dict,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Reconstruct PDF using SmolDocling layout coordinates
        This is the core layout-preserving algorithm

        Args:
            optimized_markdown: Optimized resume content in markdown
            layout_coords: Layout coordinates from SmolDocling
            output_filename: Optional custom filename

        Returns:
            Path to the generated PDF file
        """
        try:
            # Generate output path
            if not output_filename:
                output_filename = f"optimized_resume_{hash(optimized_markdown)}.pdf"

            output_path = self.temp_dir / output_filename

            # Parse markdown into sections
            sections = self._parse_markdown_sections(optimized_markdown)

            # Get image size for coordinate conversion
            image_size = layout_coords.get('image_size', (612, 792))
            text_blocks = layout_coords.get('text_blocks', [])

            # If no layout blocks found, use fallback PDF creation for better results
            if not text_blocks:
                logger.info("No layout blocks found, using fallback PDF creation for better formatting")
                return self._create_fallback_pdf(optimized_markdown, output_filename)

            # Create PDF with ReportLab
            c = canvas.Canvas(str(output_path), pagesize=self.default_pagesize)

            # Map sections to layout blocks and draw
            self._draw_sections_with_layout(c, sections, text_blocks, image_size)

            # Save PDF
            c.save()

            logger.info(
                "PDF reconstruction completed",
                output_path=str(output_path),
                sections_count=len(sections),
                blocks_count=len(text_blocks)
            )

            return output_path

        except Exception as e:
            logger.error("PDF reconstruction failed", error=str(e))
            # Fallback to simple layout
            return self._create_fallback_pdf(optimized_markdown, output_filename)

    def _parse_markdown_sections(self, markdown: str) -> Dict[str, str]:
        """Parse markdown content into resume sections"""
        sections = {}
        current_section = "header"
        current_content = []

        lines = markdown.split('\n')

        for line in lines:
            line = line.strip()

            # Detect section headers
            if line.startswith('# '):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Start new section
                current_section = "header"
                current_content = [line[2:]]  # Remove '# '

            elif line.startswith('## '):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Determine section type
                section_title = line[3:].lower()
                if 'summary' in section_title or 'profile' in section_title:
                    current_section = "summary"
                elif 'experience' in section_title or 'work' in section_title:
                    current_section = "experience"
                elif 'skill' in section_title:
                    current_section = "skills"
                elif 'education' in section_title:
                    current_section = "education"
                else:
                    current_section = "other"

                current_content = [line[3:]]  # Remove '## '

            else:
                if line:  # Skip empty lines
                    current_content.append(line)

        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _draw_sections_with_layout(
        self,
        canvas_obj,
        sections: Dict[str, str],
        text_blocks: List[Dict],
        image_size: Tuple[int, int]
    ):
        """Draw sections using SmolDocling layout coordinates"""

        # Convert image coordinates to PDF coordinates
        img_width, img_height = image_size
        pdf_width, pdf_height = self.default_pagesize

        # Scale factors for coordinate conversion
        x_scale = pdf_width / img_width
        y_scale = pdf_height / img_height

        # Map sections to blocks
        section_mapping = {
            'header': ['header', 'name', 'contact'],
            'summary': ['summary', 'profile', 'objective'],
            'experience': ['experience', 'work', 'employment'],
            'skills': ['skills', 'technical', 'competencies'],
            'education': ['education', 'academic', 'qualifications']
        }

        # Precompute available block types for diagnostics
        available_block_types = [str(b.get('type', '')).lower() for b in text_blocks]

        for section_name, content in sections.items():
            # Find matching text block
            matching_block = None
            for block in text_blocks:
                block_type = block.get('type', '').lower()
                for mapped_type in section_mapping.get(section_name, [section_name]):
                    if mapped_type in block_type:
                        matching_block = block
                        break
                if matching_block:
                    break

            if matching_block:
                # Get coordinates
                bbox = matching_block.get('bbox', {})
                x = bbox.get('x', 50) * x_scale
                y = (img_height - bbox.get('y', 100) - bbox.get('height', 50)) * y_scale  # Flip Y coordinate
                width = bbox.get('width', 500) * x_scale
                height = bbox.get('height', 50) * y_scale

                # Draw content in the specified area
                self._draw_text_in_area(canvas_obj, content, x, y, width, height, section_name)
            else:
                # Fallback positioning when no layout blocks are found
                logger.warning(
                    "No layout block found for section",
                    section=section_name,
                    available_layout_types=available_block_types[:10]  # cap for log brevity
                )

                # Use default positioning based on section order
                section_order = ['header', 'summary', 'experience', 'skills', 'education', 'other']
                section_index = section_order.index(section_name) if section_name in section_order else len(section_order)

                # Default positioning
                x = 50  # Left margin
                y = pdf_height - 100 - (section_index * 120)  # Top margin with spacing
                width = pdf_width - 100  # Full width minus margins
                height = 100  # Default section height

                # Draw content with fallback positioning
                self._draw_text_in_area(canvas_obj, content, x, y, width, height, section_name)

    def _draw_text_in_area(
        self,
        canvas_obj,
        text: str,
        x: float,
        y: float,
        width: float,
        height: float,
        section_type: str
    ):
        """Draw text within specified area with appropriate formatting"""

        # Choose font based on section type
        if section_type == 'header':
            font_name, font_size = self.default_fonts['header']
        elif section_type in ['summary', 'experience']:
            font_name, font_size = self.default_fonts['normal']
        else:
            font_name, font_size = self.default_fonts['normal']

        # Set font
        canvas_obj.setFont(font_name, font_size)

        # Simple text wrapping and drawing
        lines = self._wrap_text(text, width, font_name, font_size)

        current_y = y + height - font_size  # Start from top of area
        line_height = font_size * 1.2

        for line in lines:
            if current_y < y:  # Don't draw below the area
                break
            canvas_obj.drawString(x, current_y, line)
            current_y -= line_height

    def _wrap_text(self, text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
        """Simple text wrapping for PDF content"""
        from reportlab.pdfbase.pdfmetrics import stringWidth

        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            if stringWidth(test_line, font_name, font_size) <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, add it anyway
                    lines.append(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _create_fallback_pdf(self, markdown_content: str, output_filename: Optional[str] = None) -> Path:
        """Create a clean, professional PDF when layout extraction fails"""
        try:
            if not output_filename:
                output_filename = f"fallback_resume_{hash(markdown_content)}.pdf"

            output_path = self.temp_dir / output_filename

            # Create professional layout with custom styles
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=self.default_pagesize,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )

            # Create custom styles for professional appearance
            styles = getSampleStyleSheet()

            # Custom header style
            header_style = ParagraphStyle(
                'CustomHeader',
                parent=styles['Title'],
                fontSize=18,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=black
            )

            # Custom section header style
            section_style = ParagraphStyle(
                'CustomSection',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10,
                spaceBefore=15,
                textColor=black,
                borderWidth=1,
                borderColor=black,
                borderPadding=5
            )

            # Custom body style
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                alignment=TA_LEFT,
                leftIndent=10
            )

            story = []

            # Parse and format markdown content
            sections = self._parse_markdown_sections(markdown_content)

            for section_name, content in sections.items():
                if section_name == 'header':
                    # Professional header
                    story.append(Paragraph(content, header_style))
                    story.append(Spacer(1, 20))
                else:
                    # Section header with professional styling
                    story.append(Paragraph(section_name.title(), section_style))

                    # Format content with bullet points and proper spacing
                    formatted_content = self._format_section_content(content)
                    story.append(Paragraph(formatted_content, body_style))
                    story.append(Spacer(1, 15))

            doc.build(story)

            logger.info("Professional fallback PDF created successfully", output_path=str(output_path))
            return output_path

        except Exception as e:
            logger.error("Fallback PDF creation failed", error=str(e))
            raise

    def _format_section_content(self, content: str) -> str:
        """Format section content for professional appearance"""
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Convert markdown bullets to HTML bullets
            if line.startswith('- ') or line.startswith('* '):
                line = f"• {line[2:]}"
            elif line.startswith('  - ') or line.startswith('  * '):
                line = f"  ◦ {line[4:]}"

            # Bold text formatting
            line = line.replace('**', '<b>').replace('**', '</b>')

            formatted_lines.append(line)

        return '<br/>'.join(formatted_lines)

    async def create_optimized_resume_pdf(
        self,
        markdown_content: str,
        layout_coords: Dict,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Create optimized resume PDF with preserved layout using SmolDocling coordinates

        Args:
            markdown_content: Optimized resume content in markdown
            layout_coords: Layout coordinates from SmolDocling
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
            sections = self._parse_markdown_sections_structured(markdown_content)
            
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
    
    def _parse_markdown_sections_structured(self, markdown_content: str) -> List[Dict[str, Any]]:
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
        wrapped_lines = self._wrap_text_with_canvas(text, max_width, canvas_obj)
        
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
    
    def _wrap_text_with_canvas(self, text: str, max_width: float, canvas_obj: canvas.Canvas) -> List[str]:
        """Wrap text to fit within specified width using canvas object"""
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
        """Create professional business letter PDF with proper formatting"""
        from datetime import datetime

        page_dims = layout_coords.get('page_dimensions', {})
        page_width = page_dims.get('width', 612)
        page_height = page_dims.get('height', 792)

        c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

        # Professional business letter layout
        left_margin = 72  # 1 inch
        right_margin = page_width - 72
        top_margin = page_height - 72
        line_height = 14
        paragraph_spacing = 20

        y_position = top_margin

        # Date (right-aligned)
        current_date = datetime.now().strftime("%B %d, %Y")
        c.setFont("Helvetica", 11)
        date_width = c.stringWidth(current_date, "Helvetica", 11)
        c.drawString(right_margin - date_width, y_position, current_date)
        y_position -= paragraph_spacing * 2

        # Recipient Address (if available in sections)
        recipient_section = next((s for s in sections if s.get('type') == 'recipient'), None)
        if recipient_section:
            c.setFont("Helvetica", 11)
            recipient_lines = recipient_section['content'].split('\n')
            for line in recipient_lines:
                if line.strip():
                    c.drawString(left_margin, y_position, line.strip())
                    y_position -= line_height
            y_position -= paragraph_spacing

        # Salutation
        c.setFont("Helvetica", 11)
        c.drawString(left_margin, y_position, "Dear Hiring Manager,")
        y_position -= paragraph_spacing * 1.5

        # Letter body content
        body_content = ""
        for section in sections:
            if section.get('type') not in ['recipient', 'salutation', 'signature']:
                body_content += section['content'] + "\n\n"

        # Process body paragraphs
        paragraphs = [p.strip() for p in body_content.split('\n\n') if p.strip()]

        for paragraph in paragraphs:
            # Wrap paragraph text
            max_width = right_margin - left_margin - 20
            wrapped_lines = self._wrap_text_professional(paragraph, max_width, c)

            for line in wrapped_lines:
                if y_position < 150:  # Check for page break
                    c.showPage()
                    y_position = top_margin

                c.drawString(left_margin, y_position, line)
                y_position -= line_height

            y_position -= paragraph_spacing  # Space between paragraphs

        # Professional closing
        y_position -= paragraph_spacing
        if y_position < 150:
            c.showPage()
            y_position = top_margin

        c.drawString(left_margin, y_position, "Sincerely,")
        y_position -= paragraph_spacing * 2

        # Signature line
        c.drawString(left_margin, y_position, "[Your Name]")

        c.showPage()
        c.save()

    def _wrap_text_professional(self, text: str, max_width: float, canvas_obj) -> List[str]:
        """Professional text wrapping for business letters"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            text_width = canvas_obj.stringWidth(test_line, "Helvetica", 11)

            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, force break
                    lines.append(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines


# Singleton instance
pdf_reconstructor = PDFReconstructor()


async def get_pdf_reconstructor() -> PDFReconstructor:
    """Dependency injection for FastAPI"""
    return pdf_reconstructor
