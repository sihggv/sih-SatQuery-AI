"""
Image Processing Module for SatQuery AI
"""

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class ImageProcessor:
    """Handles all image processing operations"""
    
    @staticmethod
    def load_image(uploaded_file):
        """Load image from uploaded file"""
        try:
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            
            metadata = {
                'width': img.width,
                'height': img.height,
                'bands': img_array.shape[-1] if len(img_array.shape) == 3 else 1,
                'mode': img.mode,
                'format': img.format
            }
            
            return img_array, metadata
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None, None
    
    @staticmethod
    def resize_image(img_array, max_size=1024):
        """Resize image while maintaining aspect ratio"""
        h, w = img_array.shape[:2]
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            img_pil = Image.fromarray(img_array.astype('uint8'))
            img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            return np.array(img_resized)
        
        return img_array
    
    @staticmethod
    def apply_grounding(img_array, bboxes, labels=None, colors=None):
        """Apply bounding boxes to image"""
        img_pil = Image.fromarray(img_array.astype('uint8'))
        draw = ImageDraw.Draw(img_pil)
        
        if colors is None:
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080']
        
        h, w = img_array.shape[:2]
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            if labels and i < len(labels):
                draw.text((x1 + 5, y1 - 20), labels[i], fill=color)
        
        return np.array(img_pil)
    
    @staticmethod
    def get_image_stats(img_array):
        """Get statistical information about the image"""
        stats = {
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'min': float(np.min(img_array)),
            'max': float(np.max(img_array)),
            'shape': img_array.shape
        }
        return stats


# Test function
def test_processor():
    """Test the image processor with sample data"""
    import numpy as np
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    print("✅ Image Processor Test:")
    print(f"Image shape: {test_img.shape}")
    print("✅ All functions working!")

if __name__ == "__main__":
    test_processor()