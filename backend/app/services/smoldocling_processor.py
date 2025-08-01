"""
SmolDocling-256M Document Layout Analysis Service
Handles local document layout extraction using Hugging Face transformers
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import structlog
from PIL import Image
import torch

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoProcessor = None
    AutoModelForVision2Seq = None

logger = structlog.get_logger()


class SmolDoclingProcessor:
    """
    SmolDocling-256M processor for document layout analysis
    Extracts text bounding boxes and layout coordinates from document images
    """
    
    def __init__(self):
        """Initialize the SmolDocling processor"""
        self.processor = None
        self.model = None
        self.model_loaded = False
        self.model_name = "ds4sd/SmolDocling-256M-preview"
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, SmolDocling will be disabled")
    
    def _load_model(self):
        """Lazy load the model on first use"""
        if self.model_loaded or not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            logger.info("Loading SmolDocling-256M model", model=self.model_name)
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.model_loaded = True
            logger.info("SmolDocling-256M model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load SmolDocling model", error=str(e))
            raise
    
    def extract_layout_coordinates(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract layout coordinates from document image
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dict containing layout coordinates and text blocks
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available for SmolDocling")
        
        # Load model if not already loaded
        self._load_model()
        
        try:
            # Load and prepare image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create prompt for layout extraction with image placeholder
            prompt = """<image>Analyze this document image and extract the layout structure.
            Identify text blocks and their bounding box coordinates.
            Return the information in a structured format with coordinates for each text section."""

            # Process with model - images should be a list
            inputs = self.processor(images=[image], text=prompt, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    do_sample=False,
                    temperature=0.1
                )
            
            # Decode response
            result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the coordinates from the response
            coordinates = self._parse_layout_response(result, image.size)
            
            logger.info(
                "Layout extraction completed",
                image_size=image.size,
                blocks_found=len(coordinates.get('text_blocks', []))
            )
            
            return coordinates
            
        except Exception as e:
            logger.error("Failed to extract layout coordinates", error=str(e))
            raise
    
    def _parse_layout_response(self, response: str, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Parse the model response to extract coordinates
        
        Args:
            response: Raw model response
            image_size: (width, height) of the original image
            
        Returns:
            Structured layout data with coordinates
        """
        try:
            # Try to extract JSON if present
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group())
                    return self._normalize_coordinates(parsed_json, image_size)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: parse text-based coordinate descriptions
            text_blocks = self._extract_text_blocks_from_description(response, image_size)
            
            return {
                'text_blocks': text_blocks,
                'image_size': image_size,
                'extraction_method': 'text_parsing',
                'raw_response': response
            }
            
        except Exception as e:
            logger.warning("Failed to parse layout response", error=str(e))
            return self._create_fallback_layout(image_size)
    
    def _normalize_coordinates(self, data: Dict, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Normalize coordinates to consistent format"""
        normalized_blocks = []
        
        # Handle different possible response formats
        blocks = data.get('text_blocks', data.get('blocks', data.get('sections', [])))
        
        for block in blocks:
            normalized_block = {
                'text': block.get('text', ''),
                'bbox': self._normalize_bbox(block.get('bbox', block.get('coordinates', [])), image_size),
                'type': block.get('type', 'text'),
                'confidence': block.get('confidence', 1.0)
            }
            normalized_blocks.append(normalized_block)
        
        return {
            'text_blocks': normalized_blocks,
            'image_size': image_size,
            'extraction_method': 'json_parsing'
        }
    
    def _normalize_bbox(self, bbox: List, image_size: Tuple[int, int]) -> Dict[str, float]:
        """Normalize bounding box to consistent format"""
        if len(bbox) >= 4:
            return {
                'x': float(bbox[0]),
                'y': float(bbox[1]),
                'width': float(bbox[2] - bbox[0]) if len(bbox) == 4 else float(bbox[2]),
                'height': float(bbox[3] - bbox[1]) if len(bbox) == 4 else float(bbox[3])
            }
        
        # Fallback for invalid bbox
        return {'x': 0, 'y': 0, 'width': image_size[0], 'height': image_size[1]}
    
    def _extract_text_blocks_from_description(self, text: str, image_size: Tuple[int, int]) -> List[Dict]:
        """Extract coordinates from text description as fallback"""
        blocks = []
        
        # Look for coordinate patterns in text
        coord_patterns = [
            r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)',  # x,y,w,h format
            r'x:\s*(\d+).*?y:\s*(\d+).*?width:\s*(\d+).*?height:\s*(\d+)',  # named format
        ]
        
        for pattern in coord_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    x, y, w, h = map(int, match)
                    blocks.append({
                        'text': '',
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'type': 'text',
                        'confidence': 0.8
                    })
                except ValueError:
                    continue
        
        return blocks
    
    def _create_fallback_layout(self, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Create a simple fallback layout when parsing fails"""
        width, height = image_size
        
        # Create basic layout sections
        fallback_blocks = [
            {
                'text': '',
                'bbox': {'x': 50, 'y': 50, 'width': width - 100, 'height': height // 4},
                'type': 'header',
                'confidence': 0.5
            },
            {
                'text': '',
                'bbox': {'x': 50, 'y': height // 4 + 50, 'width': width - 100, 'height': height // 2},
                'type': 'body',
                'confidence': 0.5
            },
            {
                'text': '',
                'bbox': {'x': 50, 'y': 3 * height // 4, 'width': width - 100, 'height': height // 4 - 50},
                'type': 'footer',
                'confidence': 0.5
            }
        ]
        
        return {
            'text_blocks': fallback_blocks,
            'image_size': image_size,
            'extraction_method': 'fallback'
        }
    
    def is_available(self) -> bool:
        """Check if SmolDocling is available"""
        return TRANSFORMERS_AVAILABLE
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_loaded': self.model_loaded,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'cuda_available': torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False
        }
