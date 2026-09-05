"""
Evaluation Module for SatQuery AI Models

This module provides comprehensive evaluation capabilities:
1. Classification Metrics - Accuracy, Precision, Recall, F1, AUC
2. VQA Metrics - Exact match, WUPS, BLEU, ROUGE
3. Change Detection Metrics - IoU, F1, Pixel Accuracy
4. Grounding Metrics - IoU, Average Precision
5. Visualization - Confusion matrix, ROC curves
6. Report Generation - Comprehensive evaluation reports

Features:
- Multiple evaluation metrics
- Visualization support
- Report generation
- Batch evaluation
- Model comparison
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    jaccard_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================
# Evaluator Class
# ============================================

class Evaluator:
    """
    Evaluation utilities for SatQuery AI models
    
    Features:
    1. Compute classification metrics
    2. Compute VQA metrics
    3. Compute change detection metrics
    4. Compute grounding metrics
    5. Generate visualizations
    6. Generate evaluation reports
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Evaluator initialized (output_dir: {self.output_dir})")
    
    # ============================================
    # Classification Metrics
    # ============================================
    
    @staticmethod
    def compute_classification_metrics(
        predictions: List[int],
        targets: List[int],
        num_classes: Optional[int] = None,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            predictions: List of predictions
            targets: List of targets
            num_classes: Number of classes
            average: Averaging method for multi-class
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(targets, predictions)
        }
        
        # Determine if binary or multi-class
        unique_classes = len(set(targets))
        is_binary = unique_classes == 2
        
        if is_binary:
            metrics['precision'] = precision_score(targets, predictions, average='binary')
            metrics['recall'] = recall_score(targets, predictions, average='binary')
            metrics['f1_score'] = f1_score(targets, predictions, average='binary')
            
            try:
                metrics['auc'] = roc_auc_score(targets, predictions)
            except:
                pass
        else:
            metrics['precision'] = precision_score(targets, predictions, average=average, zero_division=0)
            metrics['recall'] = recall_score(targets, predictions, average=average, zero_division=0)
            metrics['f1_score'] = f1_score(targets, predictions, average=average, zero_division=0)
            
            # Weighted versions
            metrics['precision_weighted'] = precision_score(targets, predictions, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(targets, predictions, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(
        predictions: List[int],
        targets: List[int],
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute confusion matrix
        
        Args:
            predictions: List of predictions
            targets: List of targets
            num_classes: Number of classes
        
        Returns:
            Dictionary with confusion matrix and per-class metrics
        """
        if num_classes is None:
            num_classes = len(set(targets))
        
        cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
        
        # Per-class metrics
        per_class = {}
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[i] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': cm[i, :].sum()
            }
        
        return {
            'matrix': cm.tolist(),
            'per_class': per_class,
            'num_classes': num_classes
        }
    
    @staticmethod
    def compute_classification_report(
        predictions: List[int],
        targets: List[int],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute classification report
        
        Args:
            predictions: List of predictions
            targets: List of targets
            class_names: List of class names
        
        Returns:
            Dictionary with classification report
        """
        report = classification_report(
            targets,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    # ============================================
    # VQA Metrics
    # ============================================
    
    @staticmethod
    def compute_vqa_metrics(
        predictions: List[str],
        targets: List[str],
        multiple_answers: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Compute VQA metrics
        
        Args:
            predictions: List of predicted answers
            targets: List of target answers
            multiple_answers: Optional list of multiple target answers
        
        Returns:
            Dictionary with VQA metrics
        """
        metrics = {}
        
        # Exact match
        exact_match = sum(1 for p, t in zip(predictions, targets) if p == t) / len(predictions)
        metrics['exact_match'] = exact_match
        
        # Partial match (if answer is in target or vice versa)
        partial_match = sum(
            1 for p, t in zip(predictions, targets) 
            if p in t or t in p
        ) / len(predictions)
        metrics['partial_match'] = partial_match
        
        # Token-based metrics (BLEU, etc.)
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            bleu_scores = []
            smoothing = SmoothingFunction().method1
            
            for pred, target in zip(predictions, targets):
                pred_tokens = pred.split()
                target_tokens = [target.split()]
                bleu = sentence_bleu(target_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(bleu)
            
            metrics['bleu'] = sum(bleu_scores) / len(bleu_scores)
        except:
            pass
        
        # Multiple answers support
        if multiple_answers:
            # Check if any answer matches
            any_match = sum(
                1 for p, targets_list in zip(predictions, multiple_answers)
                if any(p == t for t in targets_list)
            ) / len(predictions)
            metrics['any_match'] = any_match
        
        return metrics
    
    @staticmethod
    def compute_wups_score(
        predictions: List[str],
        targets: List[str],
        threshold: float = 0.5
    ) -> float:
        """
        Compute WUPS (Word Usage Similarity) score for VQA
        
        Args:
            predictions: List of predicted answers
            targets: List of target answers
            threshold: WUPS threshold
        
        Returns:
            WUPS score
        """
        try:
            from nltk.corpus import wordnet
            from nltk.tokenize import word_tokenize
            
            scores = []
            for pred, target in zip(predictions, targets):
                pred_words = set(word_tokenize(pred.lower()))
                target_words = set(word_tokenize(target.lower()))
                
                if not pred_words and not target_words:
                    scores.append(1.0)
                    continue
                
                # WUPS calculation
                common = pred_words.intersection(target_words)
                union = pred_words.union(target_words)
                
                if not union:
                    scores.append(0.0)
                    continue
                
                # Simple overlap score
                score = len(common) / len(union)
                
                # Check wordnet synonyms
                for pw in pred_words:
                    for tw in target_words:
                        try:
                            synsets_pw = wordnet.synsets(pw)
                            synsets_tw = wordnet.synsets(tw)
                            if synsets_pw and synsets_tw:
                                # Check if any synset is the same
                                for sp in synsets_pw:
                                    for st in synsets_tw:
                                        if sp == st:
                                            score = 1.0
                                            break
                                    if score == 1.0:
                                        break
                        except:
                            pass
                
                scores.append(score)
            
            return sum(scores) / len(scores)
            
        except:
            return 0.0
    
    # ============================================
    # Change Detection Metrics
    # ============================================
    
    @staticmethod
    def compute_change_detection_metrics(
        predictions: List[int],
        targets: List[int],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute change detection metrics
        
        Args:
            predictions: List of binary predictions
            targets: List of binary targets
            iou_threshold: IoU threshold for positive detection
        
        Returns:
            Dictionary with change detection metrics
        """
        # Convert to numpy arrays
        pred = np.array(predictions)
        target = np.array(targets)
        
        # Compute confusion matrix
        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        fn = np.sum((pred == 0) & (target == 1))
        tn = np.sum((pred == 0) & (target == 0))
        
        # IoU
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    @staticmethod
    def compute_iou_score(
        predicted_boxes: List[List[float]],
        target_boxes: List[List[float]],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute IoU score for bounding boxes
        
        Args:
            predicted_boxes: List of predicted boxes [x1, y1, x2, y2]
            target_boxes: List of target boxes [x1, y1, x2, y2]
            iou_threshold: IoU threshold for positive detection
        
        Returns:
            Dictionary with IoU metrics
        """
        if not predicted_boxes or not target_boxes:
            return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Calculate IoU for each pair
        ious = []
        for pbox in predicted_boxes:
            for tbox in target_boxes:
                iou = Evaluator._calculate_iou(pbox, tbox)
                ious.append(iou)
        
        # Average IoU
        avg_iou = sum(ious) / len(ious) if ious else 0.0
        
        # Precision at IoU threshold
        correct = sum(1 for i in ious if i >= iou_threshold)
        precision = correct / len(predicted_boxes) if predicted_boxes else 0.0
        recall = correct / len(target_boxes) if target_boxes else 0.0
        
        return {
            'iou': avg_iou,
            'precision': precision,
            'recall': recall,
            'correct_detections': correct,
            'total_predictions': len(predicted_boxes),
            'total_targets': len(target_boxes)
        }
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # ============================================
    # Visualization Methods
    # ============================================
    
    def plot_confusion_matrix(
        self,
        predictions: List[int],
        targets: List[int],
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            predictions: List of predictions
            targets: List of targets
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(targets, predictions)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / save_path if self.output_dir else save_path
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: List[int],
        y_pred: List[float],
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        try:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc="lower right")
            
            plt.tight_layout()
            
            if save_path:
                save_path = self.output_dir / save_path if self.output_dir else save_path
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"✅ ROC curve saved to {save_path}")
            
            return fig
            
        except:
            logger.warning("ROC curve plotting failed")
            return None
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        if 'train_loss' in history and 'val_loss' in history:
            ax1.plot(history['train_loss'], label='Train Loss', color='blue')
            ax1.plot(history['val_loss'], label='Validation Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_acc' in history and 'val_acc' in history:
            ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
            ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.output_dir / save_path if self.output_dir else save_path
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Training history saved to {save_path}")
        
        return fig
    
    # ============================================
    # Report Generation
    # ============================================
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        model_name: str = "SatQuery AI Model",
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report
        
        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
            save_path: Path to save the report
        
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 60,
            f"Evaluation Report: {model_name}",
            "=" * 60,
            "",
            f"Generated: {pd.Timestamp.now()}",
            "",
            "-" * 60,
            "Overall Metrics",
            "-" * 60,
        ]
        
        # Add metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                report_lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        report_lines.append(f"  {sub_key}: {sub_value:.4f}")
                    else:
                        report_lines.append(f"  {sub_key}: {sub_value}")
            else:
                report_lines.append(f"{key}: {value}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("End of Report")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            save_path = self.output_dir / save_path if self.output_dir else save_path
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"✅ Report saved to {save_path}")
        
        return report_text
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"✅ Metrics saved to {filepath}")


# ============================================
# Test Function
# ============================================

def test_evaluator():
    """Test the evaluator module"""
    print("🧪 Testing Evaluator...")
    print("=" * 60)
    
    # Create evaluator
    evaluator = Evaluator(output_dir="./test_evaluation")
    
    # Test classification metrics
    print("\n📋 Testing Classification Metrics:")
    predictions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    targets = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0]
    
    metrics = evaluator.compute_classification_metrics(predictions, targets)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test confusion matrix
    print("\n📋 Testing Confusion Matrix:")
    cm = evaluator.compute_confusion_matrix(predictions, targets, num_classes=2)
    print(f"  Matrix: {cm['matrix']}")
    
    # Test change detection metrics
    print("\n📋 Testing Change Detection Metrics:")
    change_pred = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    change_target = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    
    change_metrics = evaluator.compute_change_detection_metrics(change_pred, change_target)
    for key, value in change_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test report generation
    print("\n📋 Testing Report Generation:")
    report = evaluator.generate_report(metrics, model_name="Test Model")
    print(report[:500] + "...")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_evaluator()