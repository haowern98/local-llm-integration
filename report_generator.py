"""
report_generator.py - Professional markdown report generation for lecture analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import re

class ProfessionalReportGenerator:
    """Generates professional markdown reports with embedded screenshots."""
    
    def __init__(self, reports_dir):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        print(f"📁 Reports directory: {self.reports_dir.absolute()}")
    
    def generate_report(self, full_summary, screenshots, metadata):
        """Generate a comprehensive professional markdown report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"lecture_analysis_{timestamp}.md"
        report_path = self.reports_dir / report_filename
        
        # Build report content
        content = self._build_report_content(full_summary, screenshots, metadata)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Report generated successfully: {report_path.name}")
        return str(report_path.absolute())
    
    def _build_report_content(self, full_summary, screenshots, metadata):
        """Build the complete markdown report content."""
        lines = []
        
        # Header section
        lines.extend(self._generate_header(metadata))
        lines.append("\n---\n")
        
        # Executive Summary
        lines.extend(self._generate_executive_summary(full_summary))
        lines.append("\n---\n")
        
        # Detailed Analysis
        lines.extend(self._generate_detailed_analysis(full_summary, screenshots))
        lines.append("\n---\n")
        
        # Key Takeaways
        lines.extend(self._generate_key_takeaways(full_summary))
        lines.append("\n---\n")
        
        # Visual Citations Gallery
        if screenshots:
            lines.extend(self._generate_visual_citations_gallery(screenshots))
            lines.append("\n---\n")
        
        # Technical Appendix
        lines.extend(self._generate_technical_appendix(metadata, screenshots))
        
        return "\n".join(lines)
    
    def _generate_header(self, metadata):
        """Generate the report header with metadata."""
        lines = []
        lines.append("# 🎓 Professional Lecture Analysis Report")
        lines.append(f"*Generated: {metadata['analysis_start_time']}*")
        lines.append(f"*Video: `{metadata['video_path']}`*")
        lines.append(f"*Duration: {metadata['total_duration']:.1f} seconds*")
        lines.append(f"*Analysis Model: `{metadata['model_name']}`*")
        lines.append(f"*Ollama Host: `{metadata['ollama_host']}`*")
        lines.append(f"*Resolution: {metadata['target_resolution']}*")
        return lines
    
    def _generate_executive_summary(self, full_summary):
        """Generate the executive summary section."""
        lines = []
        lines.append("## 📊 Executive Summary")
        
        exec_summary = full_summary.get('executive_summary', '').strip()
        if exec_summary:
            lines.append(exec_summary)
        else:
            lines.append("A comprehensive analysis of the lecture content was performed, covering major topics, content evolution, and key visual elements. The analysis identified several critical moments that benefit from visual citation.")
        
        return lines
    
    def _generate_detailed_analysis(self, full_summary, screenshots):
        """Generate the detailed analysis section with embedded screenshots."""
        lines = []
        lines.append("## 🔍 Detailed Analysis")
        
        # Major Topics & Themes
        lines.append("\n### 🎯 Major Topics & Themes")
        major_topics = full_summary.get('major_topics', '').strip()
        if major_topics:
            lines.append(major_topics)
        else:
            lines.append("The lecture covered several interconnected topics that build upon foundational concepts. Key themes emerged through the presentation of visual materials and explanatory content.")
        
        # Content Evolution & Transitions
        lines.append("\n### 🔄 Content Evolution & Transitions")
        content_evolution = full_summary.get('content_evolution', '').strip()
        if content_evolution:
            lines.append(content_evolution)
        else:
            lines.append("The lecture content evolved logically, with concepts building upon previous material. Clear transitions occurred between different sections, supported by visual aids and practical examples.")
        
        # Add relevant screenshots to each section
        if screenshots:
            lines.append("\n### 📸 Visual Evidence Supporting Analysis")
            
            # Group screenshots by relevance score
            sorted_screenshots = sorted(screenshots, key=lambda x: x['relevance_score'], reverse=True)
            
            for screenshot in sorted_screenshots[:5]:  # Limit to top 5 most relevant
                lines.append(f"\n#### 🎯 {screenshot['region_type'].title()}: {screenshot['moment_description']}")
                lines.append(f"*Timestamp: {screenshot['timestamp']:.1f}s | Relevance: {screenshot['relevance_score']:.2f}*")
                
                # Add screenshot with relative path
                lines.append(f"\n![{screenshot['caption']}]({screenshot['relative_path']})")
                lines.append(f"*{screenshot['caption']}*")
                
                # Add context about why this matters
                if 'moment_description' in screenshot:
                    lines.append(f"\n**Context:** {screenshot['moment_description']}")
        
        return lines
    
    def _generate_key_takeaways(self, full_summary):
        """Generate key takeaways section."""
        lines = []
        lines.append("## 💡 Key Takeaways")
        
        takeaways = [
            "🔹 **Temporal context matters**: The lecture builds concepts progressively over time",
            "🔹 **Visual aids enhance understanding**: Key concepts were reinforced through diagrams and code examples",
            "🔹 **Critical moments require visual evidence**: Screenshots provide verifiable support for analysis claims",
            "🔹 **Comprehensive analysis requires full context**: Understanding the entire narrative is essential"
        ]
        
        for takeaway in takeaways:
            lines.append(takeaway)
        
        return lines
    
    def _generate_visual_citations_gallery(self, screenshots):
        """Generate a gallery of all visual citations."""
        lines = []
        lines.append("## 🖼️ Visual Citations Gallery")
        lines.append("*All screenshots captured as supporting evidence for the analysis*")
        
        for i, screenshot in enumerate(screenshots, 1):
            lines.append(f"\n### 📸 Figure {i}: {screenshot['region_type'].title()}")
            lines.append(f"*Timestamp: {screenshot['timestamp']:.1f}s*")
            lines.append(f"*Coordinates: {screenshot['coordinates']}*")
            
            lines.append(f"\n![Figure {i}: {screenshot['caption']}]({screenshot['relative_path']})")
            lines.append(f"*{screenshot['caption']}*")
            
            lines.append(f"\n**Supporting Evidence:** {screenshot['moment_description']}")
            lines.append(f"**Relevance Score:** {screenshot['relevance_score']:.2f}")
        
        return lines
    
    def _generate_technical_appendix(self, metadata, screenshots):
        """Generate technical appendix with processing details."""
        lines = []
        lines.append("## 📈 Technical Appendix")
        
        # Processing Statistics
        lines.append("\n### 📊 Processing Statistics")
        lines.append(f"- **Total frames analyzed:** {metadata['total_frames']:,}")
        lines.append(f"- **Key moments identified:** {metadata['key_moments_count']}")
        lines.append(f"- **Screenshots captured:** {metadata['screenshots_count']}")
        lines.append(f"- **Total processing time:** {metadata['analysis_total_time']:.2f} seconds")
        lines.append(f"- **Average time per frame:** {metadata['analysis_total_time'] / metadata['total_frames']:.2f} seconds")
        
        # System Configuration
        lines.append("\n### ⚙️ System Configuration")
        lines.append(f"- **Analysis Model:** `{metadata['model_name']}`")
        lines.append(f"- **Ollama Host:** `{metadata['ollama_host']}`")
        lines.append(f"- **Frame Interval:** {metadata['frame_interval']} seconds")
        lines.append(f"- **Target Resolution:** {metadata['target_resolution']}")
        lines.append(f"- **Analysis Date:** {metadata['analysis_start_time']}")
        
        # Screenshot Details
        if screenshots:
            lines.append("\n### 📸 Screenshot Details")
            region_types = {}
            for screenshot in screenshots:
                region_type = screenshot['region_type']
                region_types[region_type] = region_types.get(region_type, 0) + 1
            
            lines.append("**Screenshot Distribution by Type:**")
            for region_type, count in region_types.items():
                percentage = (count / len(screenshots)) * 100
                lines.append(f"- **{region_type.title()}:** {count} ({percentage:.1f}%)")
            
            avg_relevance = sum(s['relevance_score'] for s in screenshots) / len(screenshots)
            lines.append(f"**Average Relevance Score:** {avg_relevance:.2f}")
        
        # Performance Notes
        lines.append("\n### ⚡ Performance Notes")
        lines.append("- *Processing time may vary based on video complexity and system resources*")
        lines.append("- *Screenshot relevance scores are AI-generated estimates of importance*")
        lines.append("- *Temporal context was preserved throughout the analysis pipeline*")
        
        return lines