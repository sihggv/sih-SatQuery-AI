"""
Query Handler Module for SatQuery AI
Processes user queries and manages API communication
"""

import streamlit as st
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

class QueryHandler:
    """Handles query processing and backend communication"""
    
    # Task classification patterns
    TASK_PATTERNS: Dict[str, List[str]] = {
        'vqa': [
            r'(what|which|who|where|when|why|how|is|are|does|do|did|has|have)',
            r'(tell me about|describe|explain|analyze)'
        ],
        'captioning': [
            r'(describe|summarize|caption|overview)',
            r'(what.*visible|what.*see|what.*show|scene description)'
        ],
        'grounding': [
            r'(highlight|locate|find|show|point|identify|where is|where are)',
            r'(water body|building|road|forest|agriculture|crop|object)'
        ],
        'change_detection': [
            r'(changed|change|difference|between|before|after|compare|temporal)',
            r'(increase|decrease|remained|same|different|transition)'
        ],
        'optical_sar_fusion': [
            r'(optical.*sar|sar.*optical|both|combined|together|fusion)',
            r'(multisensor|complementary|cross-modal|multi-modal)'
        ]
    }
    
    @staticmethod
    def classify_query(query: str) -> Dict[str, float]:
        """
        Classify the user query into task types with confidence scores
        """
        query_lower = query.lower()
        scores: Dict[str, float] = {}
        
        for task, patterns in QueryHandler.TASK_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                score += len(matches) * 0.25
            scores[task] = min(score, 1.0)
        
        # Ensure at least one task gets high score
        if max(scores.values()) < 0.3:  # Fixed: Using scores.values()
            scores['vqa'] = 0.8  # Default to VQA
        
        return scores
    
    @staticmethod
    def extract_entities(query: str) -> Dict[str, List[str]]:
        """
        Extract entities like locations, objects, dates from query
        """
        entities: Dict[str, List[str]] = {
            'locations': [],
            'objects': [],
            'dates': [],
            'attributes': []
        }
        
        query_lower = query.lower()
        
        # Common object patterns
        object_keywords = [
            'water', 'building', 'road', 'forest', 'agriculture', 'crop',
            'urban', 'rural', 'river', 'lake', 'mountain', 'desert',
            'vegetation', 'soil', 'cloud', 'shadow', 'structure'
        ]
        
        for obj in object_keywords:
            if obj in query_lower:
                entities['objects'].append(obj)
        
        # Date patterns (YYYY-MM-DD or similar)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}'
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, query)
            entities['dates'].extend(dates)
        
        # Location patterns (capitalized words)
        location_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        locations = re.findall(location_pattern, query)
        entities['locations'] = locations[:3]  # Limit to 3
        
        return entities
    
    @staticmethod
    def build_payload(query: str, image_type: str, images: List) -> Dict[str, Any]:
        """
        Build the payload to send to backend
        """
        # Classify the query
        task_scores = QueryHandler.classify_query(query)
        entities = QueryHandler.extract_entities(query)
        
        # Determine primary task - Fixed: using max() on dict values
        primary_task = max(task_scores, key=lambda k: task_scores[k])
        
        # Build payload
        payload = {
            'query': query,
            'image_type': image_type,
            'num_images': len(images),
            'primary_task': primary_task,
            'task_scores': task_scores,
            'entities': entities,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'enable_grounding': True,
                'confidence_threshold': 0.5,
                'max_tokens': 200,
                'temperature': 0.7
            }
        }
        
        return payload
    
    @staticmethod
    def send_to_backend(payload: Dict[str, Any], backend_url: str = "http://localhost:8000/api/analyze") -> Dict[str, Any]:
        """
        Send the query to the backend API
        """
        try:
            import requests
            
            response = requests.post(
                backend_url,
                json=payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': True,
                    'message': f"Backend error: {response.status_code}",
                    'status_code': response.status_code
                }
                
        except ImportError:
            return {
                'error': True,
                'message': "Requests module not installed. Using simulated response.",
                'simulated': True
            }
        except requests.exceptions.ConnectionError:
            return {
                'error': True,
                'message': "Cannot connect to backend. Using simulated response.",
                'simulated': True
            }
        except requests.exceptions.Timeout:
            return {
                'error': True,
                'message': "Backend request timed out. Using simulated response.",
                'simulated': True
            }
        except Exception as e:
            return {
                'error': True,
                'message': f"Error: {str(e)}",
                'simulated': True
            }
    
    @staticmethod
    def generate_simulated_response(query: str, image_type: str, task_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a simulated response when backend is not available
        """
        # Fixed: Using max() with key parameter
        if task_scores:
            primary_task = max(task_scores, key=lambda k: task_scores[k])
        else:
            primary_task = 'vqa'
        
        responses = {
            'vqa': {
                'answer': "Based on satellite image analysis, the area shows a mixed landscape with agricultural fields, urban structures, water bodies, and vegetation. The land cover distribution appears typical of a suburban-rural interface.",
                'confidence': 0.87
            },
            'captioning': {
                'answer': "This satellite image captures a diverse landscape featuring agricultural plots with regular patterns, scattered urban settlements with road networks, several water bodies with distinct boundaries, and patches of vegetation/forest cover. The scene represents a mixed land-use environment.",
                'confidence': 0.92
            },
            'grounding': {
                'answer': "The following regions have been identified: Water bodies in the north-west (3 regions), Built-up areas in the south-east (2 clusters), Agricultural fields in the central region (large contiguous area), Forest cover in the north-east (dense patch).",
                'confidence': 0.85,
                'visual_evidence': True
            },
            'change_detection': {
                'answer': "Significant changes detected between the two time periods: Urban expansion increased by 12.5%, Agricultural land decreased by 8.3%, New water body formation detected in the south-west, Forest cover remained relatively stable (±1.2%).",
                'confidence': 0.79,
                'change_map': True
            },
            'optical_sar_fusion': {
                'answer': "Combined optical and SAR analysis reveals: Built-up areas (Optical + SAR confirmed with high confidence), Water bodies (SAR shows low backscatter, Optical confirms dark regions), Agricultural fields (Optical shows vegetation patterns, SAR shows moderate backscatter), Urban infrastructure clearly visible in both modalities.",
                'confidence': 0.88
            }
        }
        
        response = responses.get(primary_task, responses['vqa'])
        
        return {
            'success': True,
            'primary_task': primary_task,
            'task_scores': task_scores,
            'result': response,
            'simulated': True,
            'message': "This is a simulated response. Connect to backend for actual results."
        }


# Test function
def test_query_handler():
    """Test the query handler"""
    test_queries = [
        "What is the land-cover in this image?",
        "Highlight the water bodies in this image.",
        "What changed between these two dates?",
        "Use optical and SAR images together.",
        "Has the built-up area increased?"
    ]
    
    print("✅ Query Handler Test:")
    for query in test_queries:
        scores = QueryHandler.classify_query(query)
        # Fixed: Using max() with key parameter
        primary = max(scores, key=lambda k: scores[k])
        print(f"Query: {query[:30]}... -> Primary Task: {primary}")

if __name__ == "__main__":
    test_query_handler()