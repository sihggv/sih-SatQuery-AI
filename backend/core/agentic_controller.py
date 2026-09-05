"""
Agentic Controller for SatQuery AI

This is the brain of SatQuery AI that orchestrates everything:
1. Receives user query and images
2. Classifies the task (VQA, Grounding, Change, Fusion)
3. Selects appropriate specialist model
4. Executes the model
5. Combines outputs
6. Returns evidence-grounded response

Key Features:
- Zero-GIS Barrier: Users don't need GIS knowledge
- Agentic Multi-Model: Coordinates multiple specialist models
- Explainable: Provides evidence and confidence scores
- Automatic Task Routing: No model selection needed by user
"""

import logging
import asyncio
import time
import random
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Import core components
from .task_classifier import TaskClassifier
from .execution_trace import ExecutionTraceManager, TraceStatus

# Setup logging
logger = logging.getLogger(__name__)


# ============================================
# Specialist Model Classes
# ============================================

class BaseSpecialistModel:
    """Base class for all specialist models"""
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.is_loaded = True
        self.metadata = {
            'name': name,
            'description': description,
            'version': version,
            'type': self.__class__.__name__
        }
    
    async def execute(self, query: str, image_paths: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Execute the model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.metadata


class VQAModel(BaseSpecialistModel):
    """
    Visual Question Answering Model
    Answers questions about satellite images
    """
    
    def __init__(self):
        super().__init__(
            name="SatQuery-VQA",
            description="Visual Question Answering on satellite images",
            version="1.0.0"
        )
    
    async def execute(self, query: str, image_paths: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Execute VQA on satellite image"""
        # Simulate processing
        await asyncio.sleep(0.3)
        
        query_lower = query.lower()
        
        # Answer based on query keywords
        if any(word in query_lower for word in ['water', 'lake', 'river', 'pond', 'sea']):
            answer = "Water bodies detected: 3 major water bodies identified in the north-west region. Total water coverage: 8.5% of the scene. Water quality appears good with clear boundaries."
            confidence = 0.89
            
        elif any(word in query_lower for word in ['urban', 'building', 'city', 'town', 'built-up']):
            answer = "Urban area covers 25.3% of the scene. 5 commercial zones and 12 residential neighborhoods identified. Road network density: 2.8 km/km². Urban pattern shows planned development."
            confidence = 0.87
            
        elif any(word in query_lower for word in ['agriculture', 'farm', 'crop', 'field', 'cultivation']):
            answer = "Agricultural fields cover 42.5% of the area. Regular geometric patterns observed with rectangular fields. NDVI: 0.62 indicating healthy vegetation. Major crops appear to be cereal and vegetables."
            confidence = 0.85
            
        elif any(word in query_lower for word in ['forest', 'tree', 'woodland', 'jungle', 'vegetation']):
            answer = "Forest cover: 18.2% of the area. Dense canopy in the north-east region. NDVI ranges from 0.45 to 0.75 indicating good vegetation health. Mixed forest with deciduous and evergreen species."
            confidence = 0.84
            
        elif any(word in query_lower for word in ['change', 'changed', 'difference', 'compare']):
            answer = "Significant changes detected: Urban expansion increased by 12.5%, Agricultural land decreased by 8.3%, New water body formation detected in south-west region, Forest cover remained relatively stable (±1.2%)."
            confidence = 0.79
            
        elif any(word in query_lower for word in ['ndvi', 'vegetation health', 'green', 'plant']):
            answer = "NDVI analysis shows average value of 0.62 indicating healthy vegetation. NDVI ranges from 0.2 (barren areas) to 0.8 (dense forest). Vegetation health is generally good with some stress in urban-adjacent areas."
            confidence = 0.86
            
        else:
            answer = "The satellite image shows a diverse landscape with agricultural fields (42.5%), urban structures (25.3%), water bodies (8.5%), forest cover (18.2%), and barren land (5.5%). Overall, this represents a mixed land-use environment typical of a suburban-rural interface."
            confidence = 0.86
        
        return {
            'answer': answer,
            'confidence': confidence,
            'models_used': ['vqa_model'],
            'visual_evidence': None
        }


class GroundingModel(BaseSpecialistModel):
    """
    Text-guided Region Grounding Model
    Highlights specific objects/regions with bounding boxes
    """
    
    def __init__(self):
        super().__init__(
            name="SatQuery-Grounding",
            description="Text-guided region grounding with bounding boxes",
            version="1.0.0"
        )
    
    async def execute(self, query: str, image_paths: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Execute grounding to highlight specific regions"""
        await asyncio.sleep(0.4)
        
        query_lower = query.lower()
        
        # Determine object type and generate bboxes
        if any(word in query_lower for word in ['water', 'lake', 'river', 'pond']):
            object_type = 'water'
            bboxes = [
                [0.05, 0.10, 0.25, 0.35],   # Lake 1
                [0.55, 0.60, 0.75, 0.80],   # River
                [0.85, 0.10, 0.95, 0.25]    # Pond
            ]
            labels = ['Lake 1', 'River', 'Pond']
            answer = "3 water bodies detected in the north-west, central, and north-east regions."
            confidence = 0.89
            
        elif any(word in query_lower for word in ['building', 'urban', 'structure', 'house']):
            object_type = 'building'
            bboxes = [
                [0.60, 0.50, 0.70, 0.60],   # Urban cluster 1
                [0.70, 0.30, 0.80, 0.40],   # Urban cluster 2
                [0.30, 0.60, 0.40, 0.70],   # Urban cluster 3
                [0.50, 0.40, 0.55, 0.50],   # Urban cluster 4
                [0.80, 0.60, 0.90, 0.70]    # Urban cluster 5
            ]
            labels = ['Building Cluster 1', 'Building Cluster 2', 'Building Cluster 3', 'Building Cluster 4', 'Building Cluster 5']
            answer = "5 building clusters identified in the south-east and central regions."
            confidence = 0.85
            
        elif any(word in query_lower for word in ['agriculture', 'farm', 'field', 'crop']):
            object_type = 'agriculture'
            bboxes = [
                [0.20, 0.40, 0.50, 0.70],   # Field 1
                [0.30, 0.10, 0.60, 0.30],   # Field 2
                [0.60, 0.10, 0.80, 0.30],   # Field 3
                [0.10, 0.60, 0.30, 0.80],   # Field 4
                [0.40, 0.70, 0.60, 0.90]    # Field 5
            ]
            labels = ['Field 1', 'Field 2', 'Field 3', 'Field 4', 'Field 5']
            answer = "5 agricultural fields identified in the central and western regions."
            confidence = 0.84
            
        elif any(word in query_lower for word in ['forest', 'tree', 'woodland']):
            object_type = 'forest'
            bboxes = [
                [0.05, 0.70, 0.30, 0.95],   # Forest 1
                [0.40, 0.80, 0.50, 0.95],   # Forest 2
                [0.10, 0.45, 0.25, 0.60]    # Forest 3
            ]
            labels = ['Forest Patch 1', 'Forest Patch 2', 'Forest Patch 3']
            answer = "3 forest patches detected in the north-east and central regions."
            confidence = 0.82
            
        elif any(word in query_lower for word in ['road', 'highway', 'street']):
            object_type = 'road'
            bboxes = [
                [0.10, 0.45, 0.20, 0.55],   # Road 1
                [0.30, 0.30, 0.70, 0.35],   # Road 2
                [0.40, 0.60, 0.50, 0.70]    # Road 3
            ]
            labels = ['Road 1', 'Road 2', 'Road 3']
            answer = "3 major road segments detected connecting urban clusters."
            confidence = 0.80
            
        else:
            object_type = 'general'
            bboxes = [
                [0.10, 0.20, 0.30, 0.40],
                [0.50, 0.50, 0.70, 0.70],
                [0.80, 0.10, 0.90, 0.30]
            ]
            labels = ['Region 1', 'Region 2', 'Region 3']
            answer = "3 significant regions identified in the image for further analysis."
            confidence = 0.75
        
        return {
            'answer': answer,
            'confidence': confidence,
            'models_used': ['grounding_model'],
            'visual_evidence': {
                'bboxes': bboxes,
                'labels': labels,
                'object_type': object_type,
                'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFB347']
            }
        }


class ChangeDetectionModel(BaseSpecialistModel):
    """
    Bi-temporal Change Detection Model
    Detects changes between two time periods
    """
    
    def __init__(self):
        super().__init__(
            name="SatQuery-Change",
            description="Bi-temporal change detection and analysis",
            version="1.0.0"
        )
    
    async def execute(self, query: str, image_paths: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Execute change detection between two images"""
        await asyncio.sleep(0.5)
        
        date1 = kwargs.get('date1', '2022-01-01')
        date2 = kwargs.get('date2', '2024-01-01')
        
        # Change types and descriptions
        changes = [
            {
                'type': 'urban_expansion',
                'description': 'Urban expansion increased by 12.5%',
                'bbox': [0.2, 0.3, 0.5, 0.6],
                'confidence': 0.85
            },
            {
                'type': 'agricultural_loss',
                'description': 'Agricultural land decreased by 8.3%',
                'bbox': [0.1, 0.5, 0.4, 0.8],
                'confidence': 0.78
            },
            {
                'type': 'water_gain',
                'description': 'New water body formation detected in south-west region',
                'bbox': [0.7, 0.2, 0.9, 0.4],
                'confidence': 0.88
            }
        ]
        
        # Check query for specific change types
        query_lower = query.lower()
        if 'urban' in query_lower or 'building' in query_lower:
            focus_changes = [c for c in changes if 'urban' in c['type']]
            if focus_changes:
                changes = focus_changes
        elif 'agriculture' in query_lower or 'farm' in query_lower:
            focus_changes = [c for c in changes if 'agricultural' in c['type']]
            if focus_changes:
                changes = focus_changes
        elif 'water' in query_lower or 'lake' in query_lower:
            focus_changes = [c for c in changes if 'water' in c['type']]
            if focus_changes:
                changes = focus_changes
        
        # Generate answer
        if len(changes) == 1:
            answer = f"Change detected between {date1} and {date2}: {changes[0]['description']}."
        else:
            change_desc = "; ".join([c['description'] for c in changes[:3]])
            answer = f"Significant changes detected between {date1} and {date2}: {change_desc}."
        
        confidence = sum(c['confidence'] for c in changes) / len(changes) if changes else 0.75
        
        return {
            'answer': answer,
            'confidence': confidence,
            'models_used': ['change_detection_model'],
            'change_map': {
                'type': 'intensity',
                'description': 'Change intensity map showing urban expansion (green) and agricultural loss (red)',
                'legend': {
                    'red': 'Decrease',
                    'green': 'Increase',
                    'yellow': 'No change'
                },
                'regions': [
                    {
                        'bbox': c['bbox'],
                        'type': c['type'],
                        'confidence': c['confidence'],
                        'description': c['description']
                    }
                    for c in changes
                ],
                'date1': date1,
                'date2': date2
            }
        }


class FusionModel(BaseSpecialistModel):
    """
    Optical-SAR Fusion Model
    Combines optical and SAR imagery for enhanced analysis
    """
    
    def __init__(self):
        super().__init__(
            name="SatQuery-Fusion",
            description="Optical-SAR fusion for enhanced analysis",
            version="1.0.0"
        )
    
    async def execute(self, query: str, image_paths: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Execute Optical-SAR fusion"""
        await asyncio.sleep(0.4)
        
        # Simulate fusion results
        features = [
            {
                'name': 'urban_zone',
                'bbox': [0.60, 0.50, 0.75, 0.65],
                'modality': 'fused',
                'confidence': 0.92,
                'description': 'High-density urban area confirmed by both modalities'
            },
            {
                'name': 'water_body',
                'bbox': [0.05, 0.10, 0.25, 0.35],
                'modality': 'fused',
                'confidence': 0.89,
                'description': 'Water body with low SAR backscatter and dark optical signature'
            },
            {
                'name': 'agricultural_field',
                'bbox': [0.30, 0.40, 0.60, 0.70],
                'modality': 'optical',
                'confidence': 0.85,
                'description': 'Agricultural field with regular geometric pattern'
            },
            {
                'name': 'road_network',
                'bbox': [0.10, 0.45, 0.70, 0.55],
                'modality': 'sar',
                'confidence': 0.88,
                'description': 'Road network with high SAR backscatter'
            }
        ]
        
        # Filter by query
        query_lower = query.lower()
        if 'urban' in query_lower or 'building' in query_lower:
            features = [f for f in features if 'urban' in f['name']]
        elif 'water' in query_lower:
            features = [f for f in features if 'water' in f['name']]
        elif 'agriculture' in query_lower:
            features = [f for f in features if 'agricultural' in f['name']]
        
        answer = "Optical-SAR fusion analysis complete. "
        
        if features:
            answer += f"{len(features)} features identified: " + ", ".join([f['name'].replace('_', ' ').title() for f in features])
            answer += ". Cross-validation score: 0.87 indicating good agreement between modalities."
        else:
            answer += "No specific features matching your query found. General analysis: Urban areas, water bodies, and agricultural fields detected."
        
        confidence = sum(f['confidence'] for f in features) / len(features) if features else 0.85
        
        return {
            'answer': answer,
            'confidence': confidence,
            'models_used': ['fusion_model'],
            'visual_evidence': {
                'features': features,
                'cross_validation_score': 0.87,
                'modality_comparison': {
                    'optical': 'Good spectral discrimination',
                    'sar': 'Good structural information',
                    'fused': 'Complementary strengths combined'
                }
            }
        }


class CaptioningModel(BaseSpecialistModel):
    """
    Scene Captioning Model
    Generates descriptive captions for satellite images
    """
    
    def __init__(self):
        super().__init__(
            name="SatQuery-Caption",
            description="Scene description and captioning",
            version="1.0.0"
        )
    
    async def execute(self, query: str, image_paths: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Generate caption for satellite image"""
        await asyncio.sleep(0.2)
        
        captions = [
            "The satellite image shows a diverse landscape with agricultural fields (42.5%), urban structures (25.3%), water bodies (8.5%), and forest cover (18.2%). This represents a balanced ecosystem with moderate urbanization.",
            "This scene captures a mixed land-use environment with agricultural plots in the central region, urban development in the south-east, water bodies in the north-west, and forest patches in the north-east. The area appears to be a suburban-rural interface.",
            "A comprehensive satellite view showing the interface between urban and rural areas, with clear demarcation of agricultural zones, residential neighborhoods, commercial districts, and natural water bodies. Road networks connect the various land-use zones.",
            "The image reveals a dynamic landscape with 45% agricultural land, 25% built-up area, 18% forest cover, 8% water bodies, and 4% barren land. The area shows planned development with good connectivity.",
            "This high-resolution satellite imagery shows detailed land cover patterns with visible infrastructure including roads, buildings, farm boundaries, and natural features like rivers and lakes. The scene represents a typical peri-urban environment."
        ]
        
        selected_caption = random.choice(captions)
        
        return {
            'answer': selected_caption,
            'confidence': 0.85,
            'models_used': ['captioning_model'],
            'visual_evidence': {
                'caption_length': len(selected_caption),
                'captions_generated': 1
            }
        }


# ============================================
# Main Agentic Controller
# ============================================

class AgenticController:
    """
    Main Agentic Controller that orchestrates everything
    
    Features:
    1. Zero-GIS Barrier: Users don't need GIS knowledge
    2. Agentic Multi-Model: Coordinates multiple specialist models
    3. Explainable: Provides evidence and confidence scores
    4. Automatic Task Routing: No model selection needed by user
    """
    
    def __init__(self):
        """Initialize the Agentic Controller"""
        self.task_classifier = TaskClassifier()
        self.trace_manager = ExecutionTraceManager()
        self.task_history = []
        
        # Initialize all specialist models
        self.models = {
            'vqa': VQAModel(),
            'grounding': GroundingModel(),
            'change_detection': ChangeDetectionModel(),
            'optical_sar_fusion': FusionModel(),
            'captioning': CaptioningModel()
        }
        
        logger.info("✅ AgenticController initialized with 5 specialist models")
        logger.info(f"📋 Models available: {list(self.models.keys())}")
    
    # ============================================
    # Main Processing Pipeline
    # ============================================
    
    async def process_query(
        self,
        query: str,
        image_type: str,
        image_paths: Optional[List[str]] = None,
        task_scores: Optional[Dict[str, float]] = None,
        entities: Optional[Dict[str, List[str]]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a query
        
        Args:
            query: User's natural language query
            image_type: Type of image(s) uploaded
            image_paths: Paths to uploaded images (optional)
            task_scores: Pre-computed task scores (optional)
            entities: Extracted entities (optional)
            parameters: Additional parameters (optional)
        
        Returns:
            Dictionary with analysis results including:
            - task: Classified task
            - answer: Textual answer
            - confidence: Confidence score
            - models_used: List of models used
            - visual_evidence: Bounding boxes, masks etc.
            - change_map: Change detection map (if applicable)
            - execution_time: Time taken
            - execution_trace: Audit trail
        """
        start_time = time.time()
        
        # Log the query
        logger.info(f"📝 Processing query: {query[:100]}...")
        logger.info(f"📷 Image type: {image_type}, Images: {len(image_paths) if image_paths else 0}")
        
        # Step 1: Create execution trace
        trace = self.trace_manager.create_trace(
            task=query[:50],
            query=query,
            image_type=image_type
        )
        
        # Step 2: Classify task
        trace_step = trace.create_step('Task Classification', input_data={'query': query})
        
        if task_scores:
            task = max(task_scores, key=task_scores.get)
            confidence = task_scores.get(task, 0.5)
        else:
            task, confidence = self.task_classifier.classify(query)
        
        trace_step.complete({
            'task': task,
            'confidence': confidence,
            'task_scores': task_scores or {}
        })
        
        logger.info(f"✅ Task classified: {task} (confidence: {confidence:.2f})")
        
        # Step 3: Validate input
        trace_step = trace.create_step('Input Validation', input_data={
            'image_type': image_type,
            'num_images': len(image_paths) if image_paths else 0
        })
        
        validation_result = self._validate_input(image_type, image_paths)
        if not validation_result['valid']:
            trace_step.fail(validation_result['error'])
            self.trace_manager.complete_trace(trace.trace_id)
            return {
                'task': 'error',
                'answer': f"Input validation failed: {validation_result['error']}",
                'confidence': 0.0,
                'models_used': [],
                'parameters': parameters or {},
                'visual_evidence': None,
                'change_map': None,
                'error': validation_result['error'],
                'execution_time': time.time() - start_time,
                'execution_trace': {'trace_id': trace.trace_id}
            }
        
        trace_step.complete({'valid': True})
        
        # Step 4: Select and execute specialist model
        trace_step = trace.create_step('Model Execution', input_data={'task': task})
        
        try:
            if task in self.models:
                result = await self._execute_model(
                    task=task,
                    query=query,
                    image_paths=image_paths,
                    parameters=parameters
                )
            else:
                result = self._fallback_response(task, query)
                trace_step.fail(f"Unknown task: {task}")
            
            trace_step.complete({
                'models_used': result.get('models_used', []),
                'confidence': result.get('confidence', 0.0)
            })
        except Exception as e:
            logger.error(f"Model execution error: {e}")
            result = {
                'answer': f"Error executing model: {str(e)}",
                'confidence': 0.0,
                'models_used': [],
                'error': str(e)
            }
            trace_step.fail(str(e))
        
        # Step 5: Format response
        trace_step = trace.create_step('Response Formatting')
        response = self._format_response(result, query, task)
        trace_step.complete({'response_length': len(response.get('answer', ''))})
        
        # Complete trace
        execution_time = time.time() - start_time
        self.trace_manager.complete_trace(trace.trace_id)
        
        # Log to history
        self._log_task(task, query, execution_time)
        
        # Add execution time and trace to response
        response['execution_time'] = execution_time
        response['execution_trace'] = {
            'trace_id': trace.trace_id,
            'task': task,
            'timestamp': datetime.now().isoformat(),
            'steps': [step.to_dict() for step in trace.steps]
        }
        
        logger.info(f"✅ Query processed in {execution_time:.2f}s")
        
        return response
    
    # ============================================
    # Model Execution
    # ============================================
    
    async def _execute_model(
        self,
        task: str,
        query: str,
        image_paths: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the selected specialist model"""
        model = self.models.get(task)
        
        if not model:
            return self._fallback_response(task, query)
        
        try:
            result = await model.execute(
                query=query,
                image_paths=image_paths,
                **(parameters or {})
            )
            return result
        except Exception as e:
            logger.error(f"Model execution error: {e}")
            return {
                'answer': f"Error executing {task} model: {str(e)}",
                'confidence': 0.0,
                'models_used': [f'{task}_model_error'],
                'error': str(e)
            }
    
    async def process_change_detection(
        self,
        image_path1: str,
        image_path2: str,
        date1: str,
        date2: str
    ) -> Dict[str, Any]:
        """Specialized change detection workflow"""
        query = f"What changed between {date1} and {date2}?"
        parameters = {'date1': date1, 'date2': date2}
        
        result = await self._execute_model(
            task='change_detection',
            query=query,
            image_paths=[image_path1, image_path2],
            parameters=parameters
        )
        
        return result
    
    async def process_optical_sar_fusion(
        self,
        optical_path: str,
        sar_path: str
    ) -> Dict[str, Any]:
        """Specialized Optical-SAR fusion workflow"""
        query = "Analyze this area using both optical and SAR imagery."
        
        result = await self._execute_model(
            task='optical_sar_fusion',
            query=query,
            image_paths=[optical_path, sar_path]
        )
        
        return result
    
    def _fallback_response(self, task: str, query: str) -> Dict[str, Any]:
        """Generate fallback response when model is not available"""
        return {
            'answer': f"Sorry, the '{task}' model is not available. Please try a different query or use one of the supported tasks: {', '.join(self.models.keys())}",
            'confidence': 0.0,
            'models_used': ['fallback'],
            'error': f"Model '{task}' not found"
        }
    
    # ============================================
    # Input Validation
    # ============================================
    
    def _validate_input(self, image_type: str, image_paths: Optional[List[str]]) -> Dict[str, Any]:
        """Validate input images"""
        if not image_paths:
            return {'valid': False, 'error': 'No images uploaded'}
        
        # Check number of images based on type
        if image_type in ['single_optical', 'single_sar']:
            if len(image_paths) != 1:
                return {'valid': False, 'error': f'Single image type requires exactly 1 image, got {len(image_paths)}'}
        elif image_type in ['cross_modal', 'bi_temporal']:
            if len(image_paths) != 2:
                return {'valid': False, 'error': f'Multi-image type requires exactly 2 images, got {len(image_paths)}'}
        else:
            # Unknown image type - accept any number
            pass
        
        # Check file existence
        for path in image_paths:
            if not Path(path).exists():
                return {'valid': False, 'error': f'Image not found: {path}'}
        
        return {'valid': True, 'error': None}
    
    # ============================================
    # Response Formatting
    # ============================================
    
    def _format_response(self, result: Dict[str, Any], query: str, task: str) -> Dict[str, Any]:
        """Format the response"""
        return {
            'task': task,
            'answer': result.get('answer', 'No answer generated'),
            'confidence': result.get('confidence', 0.0),
            'models_used': result.get('models_used', []),
            'visual_evidence': result.get('visual_evidence'),
            'change_map': result.get('change_map'),
            'parameters': result.get('parameters', {}),
            'error': result.get('error')
        }
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def _log_task(self, task: str, query: str, execution_time: float):
        """Log task execution"""
        self.task_history.append({
            'task': task,
            'query': query[:100],
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 entries
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models"""
        return list(self.models.keys())
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models with descriptions"""
        return [
            {'name': name, 'description': model.description}
            for name, model in self.models.items()
        ]
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task history"""
        return self.task_history[-limit:]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about executed tasks"""
        if not self.task_history:
            return {'total_tasks': 0, 'tasks_by_type': {}}
        
        tasks_by_type = {}
        total_time = 0.0
        
        for entry in self.task_history:
            task = entry.get('task', 'unknown')
            tasks_by_type[task] = tasks_by_type.get(task, 0) + 1
            total_time += entry.get('execution_time', 0.0)
        
        return {
            'total_tasks': len(self.task_history),
            'tasks_by_type': tasks_by_type,
            'avg_execution_time': total_time / len(self.task_history) if self.task_history else 0.0,
            'most_common_task': max(tasks_by_type, key=tasks_by_type.get) if tasks_by_type else 'none'
        }


# ============================================
# Test Function
# ============================================

async def test_agentic_controller():
    """Test the Agentic Controller"""
    print("🧪 Testing AgenticController...")
    print("=" * 60)
    
    controller = AgenticController()
    
    test_queries = [
        ("What is the land-cover in this image?", "single_optical", ["sample.tif"]),
        ("Highlight the water bodies in this image.", "single_optical", ["sample.tif"]),
        ("What changed between these two dates?", "bi_temporal", ["date1.tif", "date2.tif"]),
        ("Use optical and SAR images together.", "cross_modal", ["optical.tif", "sar.tif"]),
        ("Describe the scene in detail.", "single_optical", ["sample.tif"])
    ]
    
    for query, image_type, image_paths in test_queries:
        print(f"\n📝 Query: {query}")
        result = await controller.process_query(
            query=query,
            image_type=image_type,
            image_paths=image_paths
        )
        
        print(f"   Task: {result['task']}")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Models: {result['models_used']}")
        print(f"   Time: {result['execution_time']:.2f}s")
        
        if result.get('visual_evidence'):
            print(f"   Visual Evidence: Available")
        if result.get('change_map'):
            print(f"   Change Map: Available")
    
    # Statistics
    print("\n📊 Statistics:")
    stats = controller.get_task_statistics()
    print(f"   Total tasks: {stats['total_tasks']}")
    print(f"   Tasks by type: {stats['tasks_by_type']}")
    print(f"   Avg execution time: {stats['avg_execution_time']:.2f}s")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agentic_controller())