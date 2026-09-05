"""
Task Classifier for SatQuery AI

This module classifies user queries into appropriate AI tasks:
1. VQA (Visual Question Answering)
2. Grounding (Region localization)
3. Change Detection (Bi-temporal analysis)
4. Optical-SAR Fusion (Cross-modal analysis)
5. Captioning (Scene description)

The classifier uses:
- Pattern matching with weighted keywords
- Natural language processing
- Confidence scoring
- Fallback strategies
"""

import logging
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================
# Data Classes
# ============================================

@dataclass
class TaskPattern:
    """Pattern for task classification"""
    name: str
    keywords: List[str]
    patterns: List[str]
    weight: float = 1.0
    description: str = ""
    
    def matches(self, text: str) -> float:
        """Calculate match score for a text"""
        text_lower = text.lower()
        score = 0.0
        
        # Check keywords
        for keyword in self.keywords:
            if keyword in text_lower:
                score += 0.3
        
        # Check regex patterns
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.5
        
        # Apply weight
        score = min(score, 1.0) * self.weight
        
        return score


@dataclass
class ClassificationResult:
    """Result of classification"""
    task: str
    confidence: float
    scores: Dict[str, float]
    matched_patterns: List[str]
    top_matches: List[Tuple[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task,
            'confidence': self.confidence,
            'scores': self.scores,
            'matched_patterns': self.matched_patterns,
            'top_matches': self.top_matches
        }


# ============================================
# Task Classifier
# ============================================

class TaskClassifier:
    """
    Classifies user queries into appropriate AI tasks
    
    Features:
    - Pattern-based matching with weighted keywords
    - Multi-task scoring
    - Confidence estimation
    - Fallback to VQA for unknown queries
    - Explainable results
    """
    
    def __init__(self):
        """Initialize the Task Classifier with patterns"""
        self.tasks = self._initialize_tasks()
        self._compile_patterns()
        logger.info(f"✅ TaskClassifier initialized with {len(self.tasks)} tasks")
    
    def _initialize_tasks(self) -> List[TaskPattern]:
        """Initialize task patterns"""
        return [
            # VQA - Visual Question Answering
            TaskPattern(
                name='vqa',
                keywords=[
                    'what', 'which', 'who', 'where', 'when', 'why', 'how',
                    'is', 'are', 'does', 'do', 'did', 'has', 'have',
                    'tell me about', 'describe', 'explain', 'analyze',
                    'land-cover', 'land use', 'vegetation', 'urban',
                    'agriculture', 'water', 'forest', 'type', 'class',
                    'how many', 'how much', 'percentage', 'area', 'size',
                    'ndvi', 'vegetation health', 'green', 'plant'
                ],
                patterns=[
                    r'(what|which|who|where|when|why|how)\s+(is|are|does|do|did|has|have)',
                    r'(tell me about|describe|explain|analyze)\s+',
                    r'(land-cover|land use|vegetation|urban|agriculture|water|forest)',
                    r'(how many|how much|percentage|area|size)',
                    r'(ndvi|vegetation health|green|plant)'
                ],
                weight=0.8,
                description='Visual Question Answering - Answer questions about satellite images'
            ),
            
            # Grounding - Text-guided Region Grounding
            TaskPattern(
                name='grounding',
                keywords=[
                    'highlight', 'locate', 'find', 'show', 'point',
                    'identify', 'where is', 'where are', 'detect',
                    'water body', 'building', 'road', 'forest',
                    'agriculture', 'crop', 'object', 'feature',
                    'bounding box', 'region', 'area', 'location', 'position',
                    'mark', 'indicate', 'outline', 'boundary'
                ],
                patterns=[
                    r'(highlight|locate|find|show|point|identify|where is|where are)',
                    r'(water body|building|road|forest|agriculture|crop|object|feature)',
                    r'(bounding box|region|area|location|position)',
                    r'(mark|indicate|outline|boundary)'
                ],
                weight=0.9,
                description='Visual Grounding - Highlight specific objects with bounding boxes'
            ),
            
            # Change Detection - Bi-temporal Change Detection
            TaskPattern(
                name='change_detection',
                keywords=[
                    'change', 'changed', 'difference', 'between',
                    'before', 'after', 'compare', 'temporal',
                    'increase', 'decrease', 'remained', 'same',
                    'different', 'transition', 'evolution',
                    'over time', 'through time', 'year', 'month', 'decade',
                    'growth', 'decline', 'expansion', 'loss', 'gain'
                ],
                patterns=[
                    r'(change|changed|difference|between|before|after|compare|temporal)',
                    r'(increase|decrease|remained|same|different|transition|evolution)',
                    r'(over time|through time|year|month|decade)',
                    r'(growth|decline|expansion|loss|gain)'
                ],
                weight=0.9,
                description='Change Detection - Detect changes between two time periods'
            ),
            
            # Optical-SAR Fusion - Cross-modal Analysis
            TaskPattern(
                name='optical_sar_fusion',
                keywords=[
                    'optical', 'sar', 'radar', 'both', 'combined',
                    'together', 'fusion', 'multisensor',
                    'complementary', 'cross-modal', 'multi-modal',
                    'sentinel', 'landsat', 'modis', 'radar',
                    'joint analysis', 'integrated', 'synergistic'
                ],
                patterns=[
                    r'(optical.*sar|sar.*optical|both|combined|together|fusion)',
                    r'(multisensor|complementary|cross-modal|multi-modal)',
                    r'(radar|sentinel|optical|sar)',
                    r'(joint analysis|integrated|synergistic)'
                ],
                weight=1.0,
                description='Optical-SAR Fusion - Combine optical and radar imagery'
            ),
            
            # Captioning - Scene Description
            TaskPattern(
                name='captioning',
                keywords=[
                    'describe', 'summarize', 'caption', 'overview',
                    'general', 'overall', 'broad', 'comprehensive',
                    'what is visible', 'what do you see', 'tell me about',
                    'scene description', 'landscape', 'depict'
                ],
                patterns=[
                    r'(describe|summarize|caption|overview|general)',
                    r'(what.*visible|what.*see|what.*show|scene description)',
                    r'(overall|general|broad|comprehensive)',
                    r'(landscape|depict)'
                ],
                weight=0.7,
                description='Captioning - Generate scene descriptions'
            )
        ]
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        for task in self.tasks:
            task.patterns = [re.compile(p, re.IGNORECASE) for p in task.patterns]
    
    # ============================================
    # Main Classification Methods
    # ============================================
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query into a task
        
        Args:
            query: User's natural language query
        
        Returns:
            Tuple of (task_name, confidence_score)
        """
        if not query or not query.strip():
            logger.warning("Empty query received, defaulting to VQA")
            return 'vqa', 0.5
        
        # Get scores for all tasks
        scores = self.get_scores(query)
        
        # Get best task
        if not scores:
            logger.warning("No scores computed, defaulting to VQA")
            return 'vqa', 0.5
        
        best_task = max(scores, key=scores.get)
        confidence = scores[best_task]
        
        # Normalize confidence
        max_score = max(scores.values())
        if max_score > 0:
            confidence = scores[best_task] / max_score
        else:
            confidence = 0.5
        
        # If confidence is too low, default to VQA
        if confidence < 0.3:
            logger.info(f"Low confidence ({confidence:.2f}), defaulting to VQA")
            return 'vqa', 0.5
        
        logger.info(f"Classified: {best_task} (confidence: {confidence:.2f})")
        return best_task, confidence
    
    def get_scores(self, query: str) -> Dict[str, float]:
        """
        Get scores for all tasks
        
        Args:
            query: User's natural language query
        
        Returns:
            Dictionary with task scores
        """
        scores = {}
        for task in self.tasks:
            score = task.matches(query)
            scores[task.name] = score
        
        return scores
    
    def classify_with_details(self, query: str) -> ClassificationResult:
        """
        Classify with detailed results
        
        Args:
            query: User's natural language query
        
        Returns:
            ClassificationResult with detailed information
        """
        scores = self.get_scores(query)
        
        if not scores:
            return ClassificationResult(
                task='vqa',
                confidence=0.5,
                scores={},
                matched_patterns=[],
                top_matches=[]
            )
        
        # Get top matches
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_matches = sorted_scores[:3]
        
        # Get best task
        best_task, confidence = self.classify(query)
        
        # Get matched patterns
        matched_patterns = self._get_matched_patterns(query, best_task)
        
        return ClassificationResult(
            task=best_task,
            confidence=confidence,
            scores=scores,
            matched_patterns=matched_patterns,
            top_matches=top_matches
        )
    
    def _get_matched_patterns(self, query: str, task_name: str) -> List[str]:
        """Get patterns that matched for a specific task"""
        matched = []
        query_lower = query.lower()
        
        for task in self.tasks:
            if task.name == task_name:
                # Check keywords
                for keyword in task.keywords:
                    if keyword in query_lower:
                        matched.append(f"keyword: {keyword}")
                
                # Check patterns
                for pattern in task.patterns:
                    if isinstance(pattern, re.Pattern):
                        if pattern.search(query_lower):
                            matched.append(f"pattern: {pattern.pattern}")
        
        return matched[:10]  # Limit to top 10 matches
    
    # ============================================
    # Multi-Query Support
    # ============================================
    
    def classify_batch(self, queries: List[str]) -> List[Tuple[str, float]]:
        """
        Classify multiple queries
        
        Args:
            queries: List of queries
        
        Returns:
            List of (task_name, confidence) tuples
        """
        return [self.classify(q) for q in queries]
    
    def get_scores_batch(self, queries: List[str]) -> List[Dict[str, float]]:
        """
        Get scores for multiple queries
        
        Args:
            queries: List of queries
        
        Returns:
            List of score dictionaries
        """
        return [self.get_scores(q) for q in queries]
    
    # ============================================
    # Explainability Methods
    # ============================================
    
    def explain_classification(self, query: str) -> Dict[str, Any]:
        """
        Get explanation for classification
        
        Args:
            query: User's natural language query
        
        Returns:
            Dictionary with explanation
        """
        result = self.classify_with_details(query)
        
        explanation = {
            'query': query,
            'classified_task': result.task,
            'confidence': result.confidence,
            'scores': result.scores,
            'top_matches': result.top_matches,
            'matched_patterns': result.matched_patterns,
            'reasoning': f"Classified as '{result.task}' with {result.confidence:.2f} confidence based on {len(result.matched_patterns)} pattern matches."
        }
        
        return explanation
    
    # ============================================
    # Task Management
    # ============================================
    
    def get_task_names(self) -> List[str]:
        """Get list of task names"""
        return [task.name for task in self.tasks]
    
    def get_task_info(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific task"""
        for task in self.tasks:
            if task.name == task_name:
                return {
                    'name': task.name,
                    'description': task.description,
                    'keywords': task.keywords[:10],  # Top 10 keywords
                    'weight': task.weight
                }
        return None
    
    def get_all_task_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tasks"""
        return {
            task.name: {
                'name': task.name,
                'description': task.description,
                'keywords': task.keywords[:10],
                'weight': task.weight
            }
            for task in self.tasks
        }
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def add_task(self, task_pattern: TaskPattern):
        """Add a new task pattern"""
        self.tasks.append(task_pattern)
        # Recompile patterns
        if isinstance(task_pattern.patterns[0], str):
            task_pattern.patterns = [re.compile(p, re.IGNORECASE) for p in task_pattern.patterns]
        logger.info(f"Added task: {task_pattern.name}")
    
    def remove_task(self, task_name: str) -> bool:
        """Remove a task pattern"""
        for i, task in enumerate(self.tasks):
            if task.name == task_name:
                self.tasks.pop(i)
                logger.info(f"Removed task: {task_name}")
                return True
        return False
    
    def update_task_weight(self, task_name: str, weight: float) -> bool:
        """Update weight of a task"""
        for task in self.tasks:
            if task.name == task_name:
                task.weight = weight
                logger.info(f"Updated weight for {task_name}: {weight}")
                return True
        return False


# ============================================
# Test Functions
# ============================================

def test_classifier():
    """Test the task classifier"""
    print("🧪 Testing TaskClassifier...")
    print("=" * 60)
    
    classifier = TaskClassifier()
    
    test_queries = [
        ("What is the land-cover in this image?", "vqa"),
        ("Highlight the water bodies in this image.", "grounding"),
        ("What changed between these two dates?", "change_detection"),
        ("Use optical and SAR images together.", "optical_sar_fusion"),
        ("Describe the scene in detail.", "captioning"),
        ("How many buildings are visible?", "vqa"),
        ("Where are the agricultural fields?", "grounding"),
        ("Compare the urban areas in 2022 and 2024.", "change_detection"),
        ("Combine optical and radar data for analysis.", "optical_sar_fusion"),
        ("Summarize what this image shows.", "captioning")
    ]
    
    print("\n📋 Classification Results:")
    correct = 0
    total = len(test_queries)
    
    for query, expected in test_queries:
        task, confidence = classifier.classify(query)
        matched = task == expected
        if matched:
            correct += 1
        
        status = "✅" if matched else "❌"
        print(f"  {status} Query: {query[:40]}...")
        print(f"     Task: {task} (expected: {expected}) - Confidence: {confidence:.2f}")
    
    accuracy = correct / total * 100
    print(f"\n📊 Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Test with details
    print("\n📋 Detailed Classification Example:")
    query = "What is the land-cover and are there water bodies?"
    result = classifier.classify_with_details(query)
    print(f"  Query: {query}")
    print(f"  Task: {result.task}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Scores: {result.scores}")
    print(f"  Top Matches: {result.top_matches}")
    
    # Test task info
    print("\n📋 Task Information:")
    info = classifier.get_all_task_info()
    for task_name, task_info in info.items():
        print(f"  ✅ {task_name}: {task_info['description']}")
        print(f"     Keywords: {', '.join(task_info['keywords'][:5])}...")
    
    print("\n✅ Test complete!")
    return accuracy


def test_classifier_batch():
    """Test batch classification"""
    print("\n🧪 Testing Batch Classification...")
    print("=" * 60)
    
    classifier = TaskClassifier()
    
    queries = [
        "What is the land-cover?",
        "Highlight water bodies.",
        "What changed?",
        "Use both optical and SAR.",
        "Describe the scene."
    ]
    
    results = classifier.classify_batch(queries)
    
    for query, (task, confidence) in zip(queries, results):
        print(f"  Query: {query}")
        print(f"    Task: {task} (conf: {confidence:.2f})")
    
    print("\n✅ Batch test complete!")


if __name__ == "__main__":
    test_classifier()
    test_classifier_batch()