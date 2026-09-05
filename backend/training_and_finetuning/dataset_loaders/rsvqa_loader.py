"""
RSVQA Dataset Loader for SatQuery AI

This module provides the RSVQA (Remote Sensing Visual Question Answering) dataset loader:
1. RSVQA - Visual Question Answering on satellite images
2. Multiple question types (yes/no, number, other)
3. Image-Text pairs
4. Answer vocabulary management
5. Question tokenization
6. Answer encoding

Features:
- VQA data loading
- Question type filtering
- Answer vocabulary management
- Tokenization
- Batch processing
- Metadata extraction
"""

import logging
import json
import random
import re
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

from .base_loader import BaseDatasetLoader

logger = logging.getLogger(__name__)


# ============================================
# Constants
# ============================================

# Common question types in RSVQA
QUESTION_TYPES = ['yesno', 'number', 'other']

# Default question and answer vocabulary
DEFAULT_WORD_VOCAB = {
    '<pad>': 0,
    '<unk>': 1,
    '<sos>': 2,
    '<eos>': 3,
    'what': 4,
    'is': 5,
    'the': 6,
    'in': 7,
    'this': 8,
    'image': 9,
    'are': 10,
    'there': 11,
    'how': 12,
    'many': 13,
    'which': 14,
    'of': 15,
    'land': 16,
    'cover': 17,
    'type': 18,
    'water': 19,
    'building': 20,
    'forest': 21,
    'agriculture': 22,
    'urban': 23,
    'area': 24,
    'vegetation': 25,
    'river': 26,
    'lake': 27,
    'road': 28,
    'bridge': 29,
    'reservoir': 30
}

DEFAULT_ANSWER_VOCAB = {
    '<pad>': 0,
    '<unk>': 1,
    'yes': 2,
    'no': 3,
    'agriculture': 4,
    'urban': 5,
    'forest': 6,
    'water': 7,
    'built-up': 8,
    'vegetation': 9,
    'barren': 10,
    '1': 11,
    '2': 12,
    '3': 13,
    '4': 14,
    '5': 15,
    '6': 16,
    '7': 17,
    '8': 18,
    '9': 19,
    '10': 20
}


# ============================================
# RSVQA Dataset
# ============================================

class RSVQADataset(Dataset):
    """
    RSVQA Dataset for Visual Question Answering
    
    Features:
    - VQA data loading
    - Question type filtering
    - Answer vocabulary management
    - Tokenization
    - Batch processing
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        question_type: str = 'all',
        max_question_len: int = 50,
        max_answer_len: int = 20,
        word_vocab: Optional[Dict[str, int]] = None,
        answer_vocab: Optional[Dict[str, int]] = None,
        use_pretrained_vocab: bool = True
    ):
        """
        Initialize RSVQA dataset
        
        Args:
            data_dir: Directory containing RSVQA data
            split: 'train', 'val', 'test'
            transform: Transform for images
            question_type: 'yesno', 'number', 'other', 'all'
            max_question_len: Maximum question length
            max_answer_len: Maximum answer length
            word_vocab: Word vocabulary (if None, builds from data)
            answer_vocab: Answer vocabulary (if None, builds from data)
            use_pretrained_vocab: Use pretrained vocabulary
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.question_type = question_type
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        
        # Vocabulary
        if use_pretrained_vocab:
            self.word_vocab = DEFAULT_WORD_VOCAB
            self.answer_vocab = DEFAULT_ANSWER_VOCAB
        else:
            self.word_vocab = word_vocab or {}
            self.answer_vocab = answer_vocab or {}
        
        self.samples = []
        self._load_data()
        
        logger.info(f"✅ Loaded {len(self.samples)} samples from RSVQA ({split})")
    
    def _load_data(self):
        """Load RSVQA data"""
        # In production, load actual RSVQA data from files
        # Here using dummy data with realistic patterns
        
        question_samples = {
            'yesno': [
                ('Is there water in this image?', 'yes'),
                ('Are there buildings in this image?', 'yes'),
                ('Is this image of a forest?', 'no'),
                ('Is the area urban?', 'yes'),
                ('Are there any roads visible?', 'yes'),
                ('Is there agriculture in this image?', 'no'),
                ('Is the land cover mostly vegetation?', 'yes'),
                ('Are there any lakes in this image?', 'no'),
                ('Is the image from a coastal area?', 'yes'),
                ('Is there cloud cover in this image?', 'no')
            ],
            'number': [
                ('How many water bodies are visible?', '3'),
                ('How many buildings are in this image?', '5'),
                ('What is the area of this region?', '2.5'),
                ('How many roads are visible?', '4'),
                ('How many bridges are in this image?', '1'),
                ('What is the NDVI value?', '0.62'),
                ('How many lakes are visible?', '2'),
                ('What is the urban percentage?', '25'),
                ('How many rivers are shown?', '1'),
                ('What is the average elevation?', '150')
            ],
            'other': [
                ('What is the land cover in this image?', 'Agriculture'),
                ('What type of vegetation is present?', 'Forest'),
                ('Describe the urban area in this image.', 'Built-up'),
                ('What is the dominant land cover type?', 'Urban'),
                ('What water bodies are visible?', 'River'),
                ('What is the main land use?', 'Agricultural'),
                ('What features are visible?', 'Buildings and roads'),
                ('What is the vegetation type?', 'Mixed forest'),
                ('What is the landscape type?', 'Urban-rural interface'),
                ('What natural features are present?', 'River and forest')
            ]
        }
        
        # Select question type
        if self.question_type == 'all':
            all_samples = []
            for qt in QUESTION_TYPES:
                all_samples.extend(question_samples[qt])
            question_samples = all_samples
        else:
            question_samples = question_samples.get(self.question_type, [])
        
        # Generate samples
        num_samples = 100 if self.split == 'train' else 30
        
        for i in range(num_samples):
            if question_samples:
                q, a = random.choice(question_samples)
            else:
                q, a = 'What is this image?', 'Unknown'
            
            self.samples.append({
                'image_id': f'img_{i:06d}',
                'question': q,
                'answer': a,
                'question_type': self._classify_question_type(q),
                'image_path': self.data_dir / f'img_{i:06d}.jpg'
            })
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['is', 'are', 'does', 'do']):
            return 'yesno'
        elif any(word in question_lower for word in ['how many', 'how much', 'what is the', 'number']):
            return 'number'
        else:
            return 'other'
    
    def _build_vocab_from_data(self):
        """Build vocabulary from data"""
        word_counter = Counter()
        answer_counter = Counter()
        
        for sample in self.samples:
            # Tokenize question
            words = sample['question'].lower().split()
            word_counter.update(words)
            
            # Tokenize answer
            ans_words = sample['answer'].lower().split()
            answer_counter.update(ans_words)
        
        # Build word vocabulary
        if not self.word_vocab:
            self.word_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
            for i, (word, _) in enumerate(word_counter.most_common(1000), start=len(self.word_vocab)):
                self.word_vocab[word] = i
        
        # Build answer vocabulary
        if not self.answer_vocab:
            self.answer_vocab = {'<pad>': 0, '<unk>': 1}
            for i, (word, _) in enumerate(answer_counter.most_common(500), start=len(self.answer_vocab)):
                self.answer_vocab[word] = i
    
    def _tokenize_question(self, question: str) -> List[int]:
        """Tokenize question"""
        tokens = ['<sos>'] + question.lower().split() + ['<eos>']
        token_ids = [self.word_vocab.get(t, self.word_vocab['<unk>']) for t in tokens]
        
        # Pad or truncate
        if len(token_ids) > self.max_question_len:
            token_ids = token_ids[:self.max_question_len]
        else:
            token_ids += [self.word_vocab['<pad>']] * (self.max_question_len - len(token_ids))
        
        return token_ids
    
    def _tokenize_answer(self, answer: str) -> List[int]:
        """Tokenize answer"""
        tokens = answer.lower().split()
        token_ids = [self.answer_vocab.get(t, self.answer_vocab['<unk>']) for t in tokens]
        
        # Pad or truncate
        if len(token_ids) > self.max_answer_len:
            token_ids = token_ids[:self.max_answer_len]
        else:
            token_ids += [self.answer_vocab['<pad>']] * (self.max_answer_len - len(token_ids))
        
        return token_ids
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample"""
        sample = self.samples[idx]
        
        # Load image (dummy)
        image = np.random.rand(3, 224, 224).astype(np.float32)
        
        # Tokenize
        question_tokens = self._tokenize_question(sample['question'])
        answer_tokens = self._tokenize_answer(sample['answer'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': torch.from_numpy(image).float(),
            'question': torch.tensor(question_tokens, dtype=torch.long),
            'answer': torch.tensor(answer_tokens, dtype=torch.long),
            'answer_text': sample['answer'],
            'question_type': sample['question_type'],
            'metadata': {
                'image_id': sample['image_id'],
                'question': sample['question']
            }
        }


# ============================================
# RSVQA Loader Class
# ============================================

class RSVQALoader(BaseDatasetLoader):
    """
    RSVQA Data Loader
    
    Features:
    - VQA data loading
    - Question type filtering
    - Answer vocabulary management
    - Tokenization
    - Batch processing
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        question_type: str = 'all',
        max_question_len: int = 50,
        max_answer_len: int = 20,
        word_vocab: Optional[Dict[str, int]] = None,
        answer_vocab: Optional[Dict[str, int]] = None,
        use_pretrained_vocab: bool = True,
        **kwargs
    ):
        """
        Initialize RSVQA loader
        
        Args:
            data_dir: Directory containing RSVQA data
            split: 'train', 'val', 'test'
            transform: Transform for images
            target_transform: Transform for targets
            question_type: 'yesno', 'number', 'other', 'all'
            max_question_len: Maximum question length
            max_answer_len: Maximum answer length
            word_vocab: Word vocabulary
            answer_vocab: Answer vocabulary
            use_pretrained_vocab: Use pretrained vocabulary
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, split, transform, target_transform, **kwargs)
        self.question_type = question_type
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.word_vocab = word_vocab
        self.answer_vocab = answer_vocab
        self.use_pretrained_vocab = use_pretrained_vocab
        
        self.num_classes = 0  # VQA doesn't have fixed classes
        self.class_names = []
        
        logger.info(f"✅ RSVQALoader initialized (question_type: {question_type})")
    
    def load_dataset(self) -> Dataset:
        """Load RSVQA dataset"""
        self.dataset = RSVQADataset(
            data_dir=self.data_dir,
            split=self.split,
            transform=self.transform,
            question_type=self.question_type,
            max_question_len=self.max_question_len,
            max_answer_len=self.max_answer_len,
            word_vocab=self.word_vocab,
            answer_vocab=self.answer_vocab,
            use_pretrained_vocab=self.use_pretrained_vocab
        )
        return self.dataset
    
    def get_class_names(self) -> List[str]:
        """Get class names (not applicable for VQA)"""
        return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'name': 'RSVQA',
            'split': self.split,
            'question_type': self.question_type,
            'max_question_len': self.max_question_len,
            'max_answer_len': self.max_answer_len,
            'dataset_size': len(self.dataset) if self.dataset else 0,
            'vocab_size': len(self.word_vocab) if self.word_vocab else 0,
            'answer_vocab_size': len(self.answer_vocab) if self.answer_vocab else 0
        }
    
    def get_vocab_info(self) -> Dict[str, Any]:
        """
        Get vocabulary information
        
        Returns:
            Dictionary with vocabulary info
        """
        if self.dataset is None:
            self.dataset = self.load_dataset()
        
        return {
            'word_vocab_size': len(self.dataset.word_vocab),
            'answer_vocab_size': len(self.dataset.answer_vocab),
            'word_vocab_sample': dict(list(self.dataset.word_vocab.items())[:10]),
            'answer_vocab_sample': dict(list(self.dataset.answer_vocab.items())[:10])
        }


# ============================================
# Test Function
# ============================================

def test_rsvqa_loader():
    """Test the RSVQA loader"""
    print("🧪 Testing RSVQALoader...")
    print("=" * 60)
    
    # Create loader
    loader = RSVQALoader(
        data_dir="./data/RSVQA",
        split='train',
        question_type='all'
    )
    
    print(f"✅ Loader created")
    print(f"   Question type: {loader.question_type}")
    
    # Load dataset
    dataset = loader.load_dataset()
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Get a sample
    sample = dataset[0]
    print(f"\n📋 Sample:")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Question tokens: {sample['question']}")
    print(f"   Answer tokens: {sample['answer']}")
    print(f"   Answer text: {sample['answer_text']}")
    print(f"   Question type: {sample['question_type']}")
    
    # Get dataloader
    dataloader = loader.get_dataloader(batch_size=8)
    batch = next(iter(dataloader))
    print(f"\n📦 Batch:")
    print(f"   Image shape: {batch['image'].shape}")
    print(f"   Question shape: {batch['question'].shape}")
    print(f"   Answer shape: {batch['answer'].shape}")
    
    # Get info
    info = loader.get_info()
    print(f"\n📋 Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Get vocab info
    try:
        vocab_info = loader.get_vocab_info()
        print(f"\n📋 Vocabulary:")
        for key, value in vocab_info.items():
            print(f"   {key}: {value}")
    except:
        pass
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_rsvqa_loader()