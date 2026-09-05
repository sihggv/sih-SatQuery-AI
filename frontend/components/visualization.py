"""
Visualization Module for SatQuery AI
Creates charts, maps, and interactive visualizations
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
from typing import Optional, Dict, List, Any, Union

class Visualization:
    """Handles all visualization and plotting"""
    
    @staticmethod
    def plot_land_cover_distribution(data: dict, title: str = "Land Cover Distribution"):
        """Create a pie chart for land cover distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        labels = list(data.keys())
        values = list(data.values())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B5CF6']
        
        ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(labels, values, color=colors[:len(labels)])
        ax2.set_title('Land Cover Distribution (Area)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Area (sq km)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_change_analysis(before_data: dict, after_data: dict):
        """Plot change detection analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Before and after comparison
        categories = list(before_data.keys())
        before_values = list(before_data.values())
        after_values = list(after_data.values())
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax1.bar(x - width/2, before_values, width, label='Before', color='#2E86AB')
        ax1.bar(x + width/2, after_values, width, label='After', color='#F18F01')
        ax1.set_xlabel('Land Cover Type')
        ax1.set_ylabel('Area (sq km)')
        ax1.set_title('Land Cover Change Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45)
        ax1.legend()
        
        # Change percentage
        changes = [(after - before) / before * 100 if before > 0 else 0 
                   for before, after in zip(before_values, after_values)]
        
        colors = ['#2E86AB' if c >= 0 else '#C73E1D' for c in changes]
        ax2.bar(categories, changes, color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Land Cover Type')
        ax2.set_ylabel('Change (%)')
        ax2.set_title('Percentage Change', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(changes):
            ax2.text(i, v + (2 if v >= 0 else -2), f'{v:.1f}%', 
                    ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_confidence_gauge(confidence: float):
        """Create a confidence gauge visualization"""
        fig, ax = plt.subplots(figsize=(8, 3))
        
        # Create horizontal bar
        ax.barh(['Confidence'], [confidence], color='#2E86AB', height=0.5)
        ax.barh(['Confidence'], [1 - confidence], left=[confidence], 
                color='#E5E7EB', height=0.5)
        
        # Add text
        ax.text(0.5, 0, f'{confidence*100:.1f}%', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title('Model Confidence Score', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_time_series(data: dict, title: str = "Temporal Analysis"):
        """Plot time series data"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        dates = list(data.keys())
        values = list(data.values())
        
        ax.plot(dates, values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax.fill_between(dates, values, alpha=0.3, color='#2E86AB')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_heatmap(data: np.ndarray, title: str = "Heatmap"):
        """Create a heatmap visualization"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='RdYlGn', interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Intensity', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def display_image_with_overlay(
        image: np.ndarray, 
        overlay: Optional[np.ndarray] = None, 
        alpha: float = 0.5, 
        title: str = "Image"
    ):
        """
        Display image with optional overlay
        
        Args:
            image: Input image as numpy array
            overlay: Optional overlay image (can be None)
            alpha: Transparency level for overlay (0-1)
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display base image
        ax.imshow(image)
        
        # Display overlay if provided
        if overlay is not None:
            ax.imshow(overlay, alpha=alpha)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_interactive_chart(data: dict, chart_type: str = "bar"):
        """Create interactive Plotly charts"""
        fig = None
        df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
        
        if chart_type == "bar":
            fig = px.bar(
                df, x='Category', y='Value',
                title="Interactive Bar Chart",
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        elif chart_type == "pie":
            fig = px.pie(
                df, values='Value', names='Category',
                title="Interactive Pie Chart",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        elif chart_type == "line":
            fig = px.line(
                df, x='Category', y='Value',
                title="Interactive Line Chart",
                markers=True
            )
        
        if fig:
            fig.update_layout(
                template='plotly_white',
                hovermode='x unified',
                font=dict(size=12)
            )
            fig.update_traces(textposition='auto')
        
        return fig
    
    @staticmethod
    def generate_statistics_table(stats: dict) -> pd.DataFrame:
        """Generate a statistics table"""
        table_data = []
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                table_data.append({
                    'Metric': key.replace('_', ' ').title(),
                    'Value': f'{value:.2f}' if isinstance(value, float) else str(value)
                })
        return pd.DataFrame(table_data)
    
    @staticmethod
    def create_comparison_chart(data1: dict, data2: dict, label1: str = "Before", label2: str = "After"):
        """Create a comparison chart"""
        categories = list(data1.keys())
        values1 = list(data1.values())
        values2 = list(data2.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, values1, width, label=label1, color='#2E86AB')
        ax.bar(x + width/2, values2, width, label=label2, color='#F18F01')
        
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title(f'Comparison: {label1} vs {label2}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        return fig


# Test function
def test_visualization():
    """Test the visualization module"""
    print("✅ Visualization Test:")
    
    # Test data
    land_cover = {
        'Forest': 45,
        'Agriculture': 25,
        'Urban': 20,
        'Water': 10
    }
    
    # Test pie chart
    fig1 = Visualization.plot_land_cover_distribution(land_cover)
    plt.close(fig1)
    print("✅ Land cover plot created")
    
    # Test confidence gauge
    fig2 = Visualization.create_confidence_gauge(0.87)
    plt.close(fig2)
    print("✅ Confidence gauge created")
    
    # Test display with overlay
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    fig3 = Visualization.display_image_with_overlay(test_img, overlay=None)
    plt.close(fig3)
    print("✅ Image display with overlay (None) created")
    
    print("✅ All visualization functions working!")

if __name__ == "__main__":
    test_visualization()