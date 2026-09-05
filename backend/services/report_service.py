"""
Report Service for SatQuery AI Backend

This service handles report generation for analysis results:
1. PDF Report Generation
2. JSON Report Generation
3. Visual Evidence Inclusion
4. Confidence Scores and Metrics
5. Downloadable Reports

Features:
- Professional PDF reports with branding
- Visual evidence (bounding boxes, change maps)
- Execution trace and audit trail
- Confidence scores with visual indicators
- Downloadable report formats (PDF, JSON)
"""

import logging
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not available. PDF generation will be limited.")

logger = logging.getLogger(__name__)


# ============================================
# Report Service Class
# ============================================

class ReportService:
    """
    Report Service for generating analysis reports
    
    Features:
    1. PDF reports with professional formatting
    2. JSON reports for data exchange
    3. Visual evidence inclusion
    4. Confidence metrics
    5. Execution audit trail
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        company_name: str = "SatQuery AI",
        company_logo: Optional[str] = None
    ):
        """
        Initialize Report Service
        
        Args:
            output_dir: Directory to save reports
            company_name: Company name for branding
            company_logo: Path to company logo image
        """
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), 'satquery_reports')
        self.company_name = company_name
        self.company_logo = company_logo
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ ReportService initialized (output_dir: {self.output_dir})")
    
    # ============================================
    # Main Report Generation Methods
    # ============================================
    
    async def generate_report(
        self,
        analysis_data: Dict[str, Any],
        format: str = "pdf",
        include_visuals: bool = True,
        include_trace: bool = True
    ) -> str:
        """
        Generate a report from analysis data
        
        Args:
            analysis_data: Analysis results dictionary
            format: Report format ('pdf' or 'json')
            include_visuals: Whether to include visual evidence
            include_trace: Whether to include execution trace
        
        Returns:
            Path to generated report file
        """
        if format.lower() == "pdf":
            return await self._generate_pdf_report(
                analysis_data,
                include_visuals,
                include_trace
            )
        elif format.lower() == "json":
            return await self._generate_json_report(
                analysis_data,
                include_trace
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # ============================================
    # PDF Report Generation
    # ============================================
    
    async def _generate_pdf_report(
        self,
        analysis_data: Dict[str, Any],
        include_visuals: bool = True,
        include_trace: bool = True
    ) -> str:
        """
        Generate PDF report
        
        Args:
            analysis_data: Analysis results
            include_visuals: Include visual evidence
            include_trace: Include execution trace
        
        Returns:
            Path to PDF file
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("PDF generation with reportlab not available. Creating simple PDF.")
            return await self._generate_simple_pdf(analysis_data)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"SatQuery_Report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1E3A8A'),
                alignment=TA_CENTER,
                spaceAfter=30
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#1E3A8A'),
                spaceAfter=12
            )
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#333333'),
                spaceAfter=6
            )
            
            # Build content
            story = []
            
            # Title
            story.append(Paragraph(f"SatQuery AI Analysis Report", title_style))
            story.append(Spacer(1, 12))
            
            # Metadata
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
            if 'task' in analysis_data:
                story.append(Paragraph(f"Task: {analysis_data['task'].upper()}", body_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph("-" * 80, body_style))
            story.append(Spacer(1, 12))
            
            # Query
            if 'query' in analysis_data:
                story.append(Paragraph("Query", heading_style))
                story.append(Paragraph(analysis_data['query'][:500], body_style))
                story.append(Spacer(1, 12))
            
            # Answer
            if 'answer' in analysis_data:
                story.append(Paragraph("Analysis Result", heading_style))
                story.append(Paragraph(analysis_data['answer'][:1000], body_style))
                story.append(Spacer(1, 12))
            
            # Confidence
            if 'confidence' in analysis_data:
                story.append(Paragraph("Confidence Score", heading_style))
                confidence = analysis_data['confidence'] * 100
                confidence_text = f"{confidence:.1f}%"
                story.append(Paragraph(confidence_text, body_style))
                
                # Simple confidence bar (using table)
                confidence_data = [
                    ['', f'{confidence:.1f}%']
                ]
                confidence_table = Table(confidence_data, colWidths=[400, 50])
                confidence_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#1E3A8A')),
                    ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#F3F4F6')),
                    ('TEXTCOLOR', (0, 0), (0, 0), colors.whitesmoke),
                    ('TEXTCOLOR', (1, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('ALIGN', (1, 0), (1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (0, 0), 12),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(confidence_table)
                story.append(Spacer(1, 12))
            
            # Models Used
            if 'models_used' in analysis_data and analysis_data['models_used']:
                story.append(Paragraph("Models Used", heading_style))
                for model in analysis_data['models_used']:
                    story.append(Paragraph(f"• {model}", body_style))
                story.append(Spacer(1, 12))
            
            # Execution Trace
            if include_trace and 'execution_trace' in analysis_data:
                story.append(Paragraph("Execution Trace", heading_style))
                trace = analysis_data['execution_trace']
                if isinstance(trace, dict):
                    for key, value in trace.items():
                        if key != 'steps':
                            story.append(Paragraph(f"{key}: {value}", body_style))
                    
                    if 'steps' in trace:
                        story.append(Paragraph("Execution Steps:", body_style))
                        for step in trace['steps'][:10]:
                            if isinstance(step, dict):
                                step_name = step.get('name', 'Unknown')
                                step_status = step.get('status', 'pending')
                                step_duration = step.get('duration', 0)
                                story.append(Paragraph(
                                    f"• {step_name} - {step_status} ({step_duration:.3f}s)",
                                    body_style
                                ))
                story.append(Spacer(1, 12))
            
            # Visual Evidence
            if include_visuals and 'visual_evidence' in analysis_data:
                story.append(Paragraph("Visual Evidence", heading_style))
                visual = analysis_data['visual_evidence']
                if isinstance(visual, dict):
                    for key, value in visual.items():
                        if key != 'bboxes' and key != 'change_regions':
                            story.append(Paragraph(f"{key}: {value}", body_style))
                    
                    if 'bboxes' in visual:
                        story.append(Paragraph(f"Number of bounding boxes: {len(visual['bboxes'])}", body_style))
                    if 'change_regions' in visual:
                        story.append(Paragraph(f"Number of change regions: {len(visual['change_regions'])}", body_style))
                story.append(Spacer(1, 12))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph(
                f"Generated by {self.company_name} AI Platform",
                body_style
            ))
            story.append(Paragraph(
                f"© {datetime.now().year} {self.company_name}. All rights reserved.",
                body_style
            ))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"✅ PDF report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return await self._generate_simple_pdf(analysis_data)
    
    async def _generate_simple_pdf(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate simple PDF when reportlab is not available
        
        Args:
            analysis_data: Analysis results
        
        Returns:
            Path to PDF file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"SatQuery_Report_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SatQuery AI Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if 'task' in analysis_data:
                f.write(f"Task: {analysis_data['task'].upper()}\n")
            f.write("\n")
            
            if 'query' in analysis_data:
                f.write("Query:\n")
                f.write(analysis_data['query'] + "\n\n")
            
            if 'answer' in analysis_data:
                f.write("Analysis Result:\n")
                f.write(analysis_data['answer'] + "\n\n")
            
            if 'confidence' in analysis_data:
                f.write(f"Confidence: {analysis_data['confidence'] * 100:.1f}%\n\n")
            
            if 'models_used' in analysis_data:
                f.write("Models Used:\n")
                for model in analysis_data['models_used']:
                    f.write(f"  • {model}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write(f"Generated by {self.company_name} AI Platform\n")
        
        logger.info(f"✅ Simple report generated: {filepath}")
        return filepath
    
    # ============================================
    # JSON Report Generation
    # ============================================
    
    async def _generate_json_report(
        self,
        analysis_data: Dict[str, Any],
        include_trace: bool = True
    ) -> str:
        """
        Generate JSON report
        
        Args:
            analysis_data: Analysis results
            include_trace: Include execution trace
        
        Returns:
            Path to JSON file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"SatQuery_Report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare JSON data
        report_data = {
            'report_type': 'SatQuery AI Analysis Report',
            'generated_at': datetime.now().isoformat(),
            'company': self.company_name,
            'analysis_data': analysis_data
        }
        
        if not include_trace:
            report_data['analysis_data'].pop('execution_trace', None)
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"✅ JSON report generated: {filepath}")
        return filepath
    
    # ============================================
    # Utility Methods
    # ============================================
    
    def get_report_path(self, report_id: str) -> Optional[str]:
        """
        Get report file path by ID
        
        Args:
            report_id: Report ID
        
        Returns:
            File path or None
        """
        # Check for report files
        for ext in ['.pdf', '.json', '.txt']:
            filepath = os.path.join(self.output_dir, f"SatQuery_Report_{report_id}{ext}")
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def list_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent reports
        
        Args:
            limit: Maximum number of reports
        
        Returns:
            List of report information
        """
        reports = []
        for filename in sorted(os.listdir(self.output_dir), reverse=True)[:limit]:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.isfile(filepath):
                reports.append({
                    'filename': filename,
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
        
        return reports
    
    def cleanup_old_reports(self, older_than_days: int = 30) -> int:
        """
        Cleanup old reports
        
        Args:
            older_than_days: Age threshold in days
        
        Returns:
            Number of files deleted
        """
        cutoff = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        deleted = 0
        
        for filename in os.listdir(self.output_dir):
            filepath = os.path.join(self.output_dir, filename)
            if os.path.isfile(filepath):
                mtime = os.path.getmtime(filepath)
                if mtime < cutoff:
                    try:
                        os.remove(filepath)
                        deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {filepath}: {e}")
        
        logger.info(f"Cleaned up {deleted} old reports")
        return deleted


# ============================================
# Test Functions
# ============================================

async def test_report_service():
    """Test the report service"""
    print("🧪 Testing ReportService...")
    print("=" * 60)
    
    # Create service
    service = ReportService()
    print(f"✅ Service initialized (output_dir: {service.output_dir})")
    
    # Test data
    analysis_data = {
        'task': 'vqa',
        'query': 'What is the land-cover in this image?',
        'answer': 'The image shows agricultural fields (42.5%), urban structures (25.3%), water bodies (8.5%), and forest cover (18.2%).',
        'confidence': 0.87,
        'models_used': ['vqa_model', 'grounding_model'],
        'visual_evidence': {'bboxes': [[0.1, 0.2, 0.3, 0.4], [0.6, 0.5, 0.8, 0.7]]},
        'execution_trace': {
            'trace_id': 'abc123',
            'steps': [
                {'name': 'Task Classification', 'status': 'success', 'duration': 0.05},
                {'name': 'Model Execution', 'status': 'success', 'duration': 0.35}
            ]
        }
    }
    
    # Generate reports
    print("\n📋 Generating reports...")
    
    # PDF report
    try:
        pdf_path = await service.generate_report(
            analysis_data,
            format='pdf',
            include_visuals=True,
            include_trace=True
        )
        print(f"✅ PDF Report: {pdf_path}")
    except Exception as e:
        print(f"❌ PDF Report failed: {e}")
    
    # JSON report
    json_path = await service.generate_report(
        analysis_data,
        format='json',
        include_trace=True
    )
    print(f"✅ JSON Report: {json_path}")
    
    # List reports
    print("\n📋 Recent Reports:")
    for report in service.list_reports(limit=5):
        print(f"  • {report['filename']} ({report['size']} bytes)")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_report_service())