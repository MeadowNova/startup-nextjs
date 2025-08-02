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
        Extract layout coordinates from document image using SmolDocling

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

            logger.info("Processing image with SmolDocling", image_size=image.size)

            # Create proper message format according to SmolDocling documentation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Convert this page to docling."}
                    ]
                },
            ]

            # Apply chat template
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            # Process inputs
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = inputs.to(device)

            # Generate DocTags output
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8192,  # Increased for full document processing
                    do_sample=False,
                    temperature=0.0
                )

            # Decode only the generated part (skip prompt)
            prompt_length = inputs.input_ids.shape[1]
            trimmed_generated_ids = generated_ids[:, prompt_length:]

            # Decode the DocTags response
            doctags = self.processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=False,
            )[0].lstrip()

            logger.info("SmolDocling generated DocTags", doctags_length=len(doctags))

            # Parse DocTags to extract layout coordinates
            coordinates = self._parse_doctags_to_coordinates(doctags, image.size)

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

    def _parse_doctags_to_coordinates(self, doctags: str, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Parse DocTags format to extract layout coordinates and text blocks

        Args:
            doctags: DocTags string from SmolDocling
            image_size: (width, height) of the image

        Returns:
            Dict with parsed layout information
        """
        try:
            import re

            text_blocks = []

            # DocTags format includes location tags like <loc_x><loc_y><loc_x2><loc_y2>
            # Extract location-based text blocks
            loc_pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>([^<]*)'
            loc_matches = re.findall(loc_pattern, doctags)

            for match in loc_matches:
                x1, y1, x2, y2, text = match
                if text.strip():  # Only add non-empty text
                    text_blocks.append({
                        'text': text.strip(),
                        'bbox': {
                            'x': int(x1),
                            'y': int(y1),
                            'width': int(x2) - int(x1),
                            'height': int(y2) - int(y1)
                        },
                        'type': 'text',
                        'confidence': 0.9
                    })

            # Extract text from common DocTags elements without location
            tag_patterns = [
                (r'<text[^>]*>([^<]+)</text>', 'text'),
                (r'<title[^>]*>([^<]+)</title>', 'title'),
                (r'<heading[^>]*>([^<]+)</heading>', 'heading'),
                (r'<paragraph[^>]*>([^<]+)</paragraph>', 'paragraph'),
            ]

            for pattern, element_type in tag_patterns:
                matches = re.findall(pattern, doctags, re.IGNORECASE)
                for i, text in enumerate(matches):
                    if text.strip():
                        # For non-located text, create approximate bounding boxes
                        text_blocks.append({
                            'text': text.strip(),
                            'bbox': {
                                'x': 50,
                                'y': 50 + (len(text_blocks) * 30),
                                'width': image_size[0] - 100,
                                'height': 25
                            },
                            'type': element_type,
                            'confidence': 0.7
                        })

            logger.info(
                "DocTags parsing completed",
                total_blocks=len(text_blocks),
                doctags_length=len(doctags)
            )

            return {
                'text_blocks': text_blocks,
                'image_size': image_size,
                'extraction_method': 'smoldocling_doctags',
                'raw_response': doctags,
                'doctags': doctags,
                'page_dimensions': {
                    'width': image_size[0],
                    'height': image_size[1]
                }
            }

        except Exception as e:
            logger.error("Failed to parse DocTags", error=str(e), doctags_preview=doctags[:200])
            return self._create_fallback_layout(image_size)

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
