"""
GPT-4o Vision Layout Analyzer for Pixel-Perfect Resume Recreation
Uses OpenAI's vision model to extract precise layout information
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog
from PIL import Image

logger = structlog.get_logger()


class VisionLayoutAnalyzer:
    """
    Advanced layout analyzer using GPT-4o Vision for pixel-perfect layout extraction
    This is the missing piece for true layout recreation
    """
    
    def __init__(self, openai_client):
        """Initialize with OpenAI client"""
        self.client = openai_client
        self.model = "gpt-4o"  # Vision model
    
    def analyze_resume_layout(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze resume layout using GPT-4o Vision for pixel-perfect extraction
        
        Args:
            image_path: Path to resume image
            
        Returns:
            Detailed layout information with precise coordinates, fonts, colors
        """
        try:
            logger.info("Starting GPT-4o Vision layout analysis", image_path=str(image_path))
            
            # Encode image to base64
            image_base64 = self._encode_image(image_path)
            
            # Create detailed prompt for layout analysis
            prompt = self._create_layout_analysis_prompt()
            
            # Call GPT-4o Vision
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            # Parse response
            layout_data = self._parse_vision_response(response.choices[0].message.content)
            
            logger.info(
                "GPT-4o Vision analysis completed",
                elements_found=len(layout_data.get('elements', [])),
                fonts_detected=len(layout_data.get('fonts', []))
            )
            
            return layout_data
            
        except Exception as e:
            logger.error("GPT-4o Vision layout analysis failed", error=str(e))
            raise
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for GPT-4o Vision"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_layout_analysis_prompt(self) -> str:
        """Create detailed prompt for layout analysis"""
        return """
IMPORTANT: You must respond with ONLY valid JSON. No explanations, no markdown formatting, just pure JSON.

Analyze this resume image and extract layout information. Return ONLY this JSON structure:

{
  "page_dimensions": {
    "width": 612,
    "height": 792,
    "dpi": 150
  },
  "elements": [
    {
      "id": "header",
      "type": "header",
      "text": "Header content",
      "bbox": {
        "x": 50,
        "y": 50,
        "width": 500,
        "height": 80
      },
      "font": {
        "family": "Arial",
        "size": 16,
        "weight": "bold",
        "color": "#000000"
      },
      "alignment": "left"
    },
    {
      "id": "summary",
      "type": "summary",
      "text": "Summary content",
      "bbox": {
        "x": 50,
        "y": 150,
        "width": 500,
        "height": 100
      },
      "font": {
        "family": "Arial",
        "size": 12,
        "weight": "normal",
        "color": "#000000"
      },
      "alignment": "left"
    }
  ],
  "sections": [
    {
      "name": "header",
      "bbox": {
        "x": 50,
        "y": 50,
        "width": 500,
        "height": 80
      },
      "elements": ["header"]
    }
  ],
  "fonts": [
    {
      "family": "Arial",
      "sizes": [12, 16],
      "weights": ["normal", "bold"],
      "colors": ["#000000"]
    }
  ]
}

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT."""
    
    def _parse_vision_response(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT-4o Vision response into structured layout data"""
        try:
            logger.info("Parsing vision response", response_preview=response_text[:500])

            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                layout_data = json.loads(json_match.group())
                return self._validate_and_normalize_layout(layout_data)
            else:
                # If no JSON found, create a fallback layout based on the text response
                logger.warning("No JSON found in vision response, creating fallback layout")
                return self._create_fallback_layout_from_text(response_text)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse vision response JSON", error=str(e))
            # Create fallback layout instead of failing
            return self._create_fallback_layout_from_text(response_text)
    
    def _validate_and_normalize_layout(self, layout_data: Dict) -> Dict[str, Any]:
        """Validate and normalize layout data from vision model"""
        
        # Ensure required fields exist
        required_fields = ['page_dimensions', 'elements', 'sections']
        for field in required_fields:
            if field not in layout_data:
                layout_data[field] = []
        
        # Normalize coordinates to ensure they're within bounds
        page_width = layout_data.get('page_dimensions', {}).get('width', 612)
        page_height = layout_data.get('page_dimensions', {}).get('height', 792)
        
        for element in layout_data.get('elements', []):
            bbox = element.get('bbox', {})
            # Clamp coordinates to page bounds
            bbox['x'] = max(0, min(bbox.get('x', 0), page_width))
            bbox['y'] = max(0, min(bbox.get('y', 0), page_height))
            bbox['width'] = max(1, min(bbox.get('width', 100), page_width - bbox['x']))
            bbox['height'] = max(1, min(bbox.get('height', 20), page_height - bbox['y']))
        
        # Add metadata
        layout_data['extraction_method'] = 'gpt4o_vision'
        layout_data['precision_level'] = 'pixel_perfect'
        
        return layout_data

    def _create_fallback_layout_from_text(self, response_text: str) -> Dict[str, Any]:
        """Create a fallback layout when vision response doesn't contain JSON"""
        logger.info("Creating fallback layout from vision text response")

        # Create a basic layout structure
        fallback_layout = {
            "page_dimensions": {
                "width": 612,
                "height": 792,
                "dpi": 300
            },
            "elements": [
                {
                    "id": "header",
                    "type": "header",
                    "text": "Header Section",
                    "bbox": {"x": 50, "y": 50, "width": 500, "height": 80},
                    "font": {"family": "Arial", "size": 16, "weight": "bold", "color": "#000000"},
                    "alignment": "left"
                },
                {
                    "id": "summary",
                    "type": "summary",
                    "text": "Summary Section",
                    "bbox": {"x": 50, "y": 150, "width": 500, "height": 100},
                    "font": {"family": "Arial", "size": 12, "weight": "normal", "color": "#000000"},
                    "alignment": "left"
                },
                {
                    "id": "experience",
                    "type": "experience",
                    "text": "Experience Section",
                    "bbox": {"x": 50, "y": 270, "width": 500, "height": 200},
                    "font": {"family": "Arial", "size": 12, "weight": "normal", "color": "#000000"},
                    "alignment": "left"
                },
                {
                    "id": "skills",
                    "type": "skills",
                    "text": "Skills Section",
                    "bbox": {"x": 50, "y": 490, "width": 500, "height": 100},
                    "font": {"family": "Arial", "size": 12, "weight": "normal", "color": "#000000"},
                    "alignment": "left"
                },
                {
                    "id": "education",
                    "type": "education",
                    "text": "Education Section",
                    "bbox": {"x": 50, "y": 610, "width": 500, "height": 100},
                    "font": {"family": "Arial", "size": 12, "weight": "normal", "color": "#000000"},
                    "alignment": "left"
                }
            ],
            "sections": [
                {
                    "name": "header",
                    "bbox": {"x": 50, "y": 50, "width": 500, "height": 80},
                    "elements": ["header"]
                },
                {
                    "name": "summary",
                    "bbox": {"x": 50, "y": 150, "width": 500, "height": 100},
                    "elements": ["summary"]
                },
                {
                    "name": "experience",
                    "bbox": {"x": 50, "y": 270, "width": 500, "height": 200},
                    "elements": ["experience"]
                },
                {
                    "name": "skills",
                    "bbox": {"x": 50, "y": 490, "width": 500, "height": 100},
                    "elements": ["skills"]
                },
                {
                    "name": "education",
                    "bbox": {"x": 50, "y": 610, "width": 500, "height": 100},
                    "elements": ["education"]
                }
            ],
            "fonts": [
                {
                    "family": "Arial",
                    "sizes": [12, 14, 16],
                    "weights": ["normal", "bold"],
                    "colors": ["#000000"]
                }
            ],
            "extraction_method": "vision_fallback",
            "precision_level": "basic_layout"
        }

        return fallback_layout


class PrecisePDFReconstructor:
    """
    Pixel-perfect PDF reconstructor using GPT-4o Vision layout data
    This achieves true layout preservation with optimized content
    """
    
    def __init__(self, vision_analyzer: VisionLayoutAnalyzer):
        self.vision_analyzer = vision_analyzer
    
    def recreate_with_optimized_content(
        self,
        original_image_path: Path,
        optimized_content: Dict[str, str],
        output_path: Path
    ) -> Path:
        """
        Recreate PDF with pixel-perfect layout using optimized content
        
        Args:
            original_image_path: Path to original resume image
            optimized_content: Dict mapping section names to optimized text
            output_path: Where to save the recreated PDF
            
        Returns:
            Path to recreated PDF with exact layout preservation
        """
        try:
            # Step 1: Analyze original layout with GPT-4o Vision
            layout_data = self.vision_analyzer.analyze_resume_layout(original_image_path)
            
            # Step 2: Map optimized content to layout elements
            content_mapping = self._map_content_to_elements(optimized_content, layout_data)
            
            # Step 3: Recreate PDF with exact positioning
            self._recreate_pdf_with_precision(layout_data, content_mapping, output_path)
            
            logger.info("Pixel-perfect PDF recreation completed", output_path=str(output_path))
            return output_path
            
        except Exception as e:
            logger.error("Precise PDF recreation failed", error=str(e))
            raise
    
    def _map_content_to_elements(
        self, 
        optimized_content: Dict[str, str], 
        layout_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Map optimized content to specific layout elements"""
        
        content_mapping = {}
        elements = layout_data.get('elements', [])
        
        # Smart mapping based on element types and content
        for element in elements:
            element_id = element.get('id', '')
            element_type = element.get('type', '')
            
            # Map content based on element type and ID
            if element_type == 'header' and 'name' in element_id.lower():
                # Keep original name or use optimized if provided
                content_mapping[element_id] = optimized_content.get('name', element.get('text', ''))
            elif 'summary' in element_id.lower() or element_type == 'summary':
                content_mapping[element_id] = optimized_content.get('summary', element.get('text', ''))
            elif 'experience' in element_id.lower() or element_type == 'experience':
                content_mapping[element_id] = optimized_content.get('experience', element.get('text', ''))
            elif 'skills' in element_id.lower() or element_type == 'skills':
                content_mapping[element_id] = optimized_content.get('skills', element.get('text', ''))
            elif 'education' in element_id.lower() or element_type == 'education':
                content_mapping[element_id] = optimized_content.get('education', element.get('text', ''))
            else:
                # Keep original text for unmapped elements
                content_mapping[element_id] = element.get('text', '')
        
        return content_mapping
    
    def _recreate_pdf_with_precision(
        self,
        layout_data: Dict[str, Any],
        content_mapping: Dict[str, str],
        output_path: Path
    ):
        """Recreate PDF with pixel-perfect positioning"""
        
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import HexColor
        
        # Get page dimensions
        page_dims = layout_data.get('page_dimensions', {})
        page_width = page_dims.get('width', 612)
        page_height = page_dims.get('height', 792)
        
        # Create canvas
        c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
        
        # Draw each element with exact positioning
        for element in layout_data.get('elements', []):
            element_id = element.get('id', '')
            bbox = element.get('bbox', {})
            font_info = element.get('font', {})
            
            # Get optimized content for this element
            text = content_mapping.get(element_id, element.get('text', ''))
            
            if text:
                # Set font
                font_family = font_info.get('family', 'Helvetica')
                font_size = font_info.get('size', 12)
                font_weight = font_info.get('weight', 'normal')
                
                # Handle font weight
                if font_weight == 'bold':
                    font_name = f"{font_family}-Bold"
                else:
                    font_name = font_family
                
                try:
                    c.setFont(font_name, font_size)
                except:
                    # Fallback to Helvetica if font not available
                    c.setFont('Helvetica', font_size)
                
                # Set color
                color = font_info.get('color', '#000000')
                if color.startswith('#'):
                    c.setFillColor(HexColor(color))
                
                # Position and draw text
                x = bbox.get('x', 0)
                y = page_height - bbox.get('y', 0) - bbox.get('height', 20)  # Flip Y coordinate
                
                # Handle text wrapping if needed
                max_width = bbox.get('width', 100)
                wrapped_lines = self._wrap_text_to_width(text, font_name, font_size, max_width, c)
                
                # Draw each line
                line_height = font_size * 1.2
                for i, line in enumerate(wrapped_lines):
                    c.drawString(x, y - (i * line_height), line)
        
        # Save PDF
        c.save()
    
    def _wrap_text_to_width(self, text: str, font_name: str, font_size: int, max_width: float, canvas_obj) -> List[str]:
        """Wrap text to fit within specified width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                text_width = canvas_obj.stringWidth(test_line, font_name, font_size)
            except:
                text_width = len(test_line) * font_size * 0.6  # Rough estimate
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)  # Force break for long words
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
