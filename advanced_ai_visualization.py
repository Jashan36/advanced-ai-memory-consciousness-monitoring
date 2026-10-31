#!/usr/bin/env python3
"""
Advanced AI Memory and Consciousness Monitoring System - Visualization
=====================================================================

This system creates comprehensive visualizations of:
- Consciousness monitoring metrics
- Memory system performance
- External knowledge integration
- Parameter-efficient fine-tuning progress
- System health and safety metrics
- Real-time cognitive monitoring

Designed specifically for the Advanced AI Memory and Consciousness Monitoring System.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import threading
from collections import deque

# Import the Advanced AI system
from advanced_ai_memory_system import (
    AdvancedAIMemorySystem, 
    AIConfig, 
    ConsciousnessMonitor,
    ExternalKnowledgeSystem,
    ParameterEfficientFineTuning,
    MultiThreadedMemorySystem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedAIVisualizer:
    """Advanced AI Memory and Consciousness Monitoring System Visualizer."""
    
    def __init__(self, output_dir: str = "advanced_ai_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color schemes for different components
        self.colors = {
            'consciousness': '#FF6B6B',
            'memory': '#4ECDC4', 
            'knowledge': '#45B7D1',
            'fine_tuning': '#96CEB4',
            'safety': '#FFEAA7',
            'performance': '#DDA0DD',
            'system_health': '#98D8C8',
            'attention': '#F7DC6F',
            'learning': '#BB8FCE',
            'awareness': '#FF6B6B',
            'focus': '#4ECDC4',
            'plasticity': '#45B7D1'
        }
        
        # Data storage for real-time visualization
        self.consciousness_data = deque(maxlen=1000)
        self.memory_data = deque(maxlen=1000)
        self.knowledge_data = deque(maxlen=1000)
        self.performance_data = deque(maxlen=1000)
        
        logger.info(f"Advanced AI Visualizer initialized. Output directory: {output_dir}")
    
    def create_comprehensive_dashboard(self, system: AdvancedAIMemorySystem, save_path: str = None):
        """Create a comprehensive dashboard for the Advanced AI system."""
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"advanced_ai_dashboard_{timestamp}.png")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('Advanced AI Memory and Consciousness Monitoring System Dashboard', 
                    fontsize=28, fontweight='bold', y=0.98)
        
        # Get system data
        system_report = system.get_system_report()
        consciousness_report = system.consciousness_monitor.get_consciousness_report()
        memory_stats = system.memory_system.get_memory_stats()
        knowledge_stats = system.external_knowledge.get_knowledge_stats()
        
        # 1. Consciousness Monitoring Radar
        ax1 = plt.subplot(3, 4, 1, projection='polar')
        self._plot_consciousness_radar(consciousness_report, ax1)
        
        # 2. System Health Gauge
        ax2 = plt.subplot(3, 4, 2)
        self._plot_system_health_gauge(system_report, ax2)
        
        # 3. Memory System Performance
        ax3 = plt.subplot(3, 4, 3)
        self._plot_memory_performance(memory_stats, ax3)
        
        # 4. Knowledge Base Statistics
        ax4 = plt.subplot(3, 4, 4)
        self._plot_knowledge_statistics(knowledge_stats, ax4)
        
        # 5. Consciousness Timeline
        ax5 = plt.subplot(3, 4, 5)
        self._plot_consciousness_timeline(consciousness_report, ax5)
        
        # 6. Memory Processing Queue
        ax6 = plt.subplot(3, 4, 6)
        self._plot_memory_queue(memory_stats, ax6)
        
        # 7. Knowledge Categories Distribution
        ax7 = plt.subplot(3, 4, 7)
        self._plot_knowledge_categories(knowledge_stats, ax7)
        
        # 8. Performance Metrics
        ax8 = plt.subplot(3, 4, 8)
        self._plot_performance_metrics(system_report, ax8)
        
        # 9. Fine-tuning Adapters
        ax9 = plt.subplot(3, 4, 9)
        self._plot_fine_tuning_adapters(system.parameter_efficient_tuning, ax9)
        
        # 10. Safety Monitoring
        ax10 = plt.subplot(3, 4, 10)
        self._plot_safety_monitoring(system_report, ax10)
        
        # 11. Interaction History
        ax11 = plt.subplot(3, 4, 11)
        self._plot_interaction_history(system_report, ax11)
        
        # 12. System Architecture Overview
        ax12 = plt.subplot(3, 4, 12)
        self._plot_system_architecture(ax12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive dashboard saved to: {save_path}")
        return save_path
    
    def _plot_consciousness_radar(self, consciousness_report: Dict[str, Any], ax):
        """Plot consciousness metrics as a radar chart."""
        if 'metrics_history' not in consciousness_report:
            ax.text(0.5, 0.5, 'No consciousness data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consciousness Metrics', fontsize=14, fontweight='bold')
            return
        
        metrics = consciousness_report['metrics_history']
        if not metrics:
            ax.text(0.5, 0.5, 'No consciousness data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consciousness Metrics', fontsize=14, fontweight='bold')
            return
        
        # Get latest metrics
        latest_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                latest_metrics[metric_name] = values[-1]
        
        if not latest_metrics:
            ax.text(0.5, 0.5, 'No consciousness data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consciousness Metrics', fontsize=14, fontweight='bold')
            return
        
        # Prepare radar chart data
        categories = list(latest_metrics.keys())
        values = list(latest_metrics.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add the first value at the end to close the plot
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['consciousness'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['consciousness'])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        
        # Add title
        ax.set_title('Consciousness Metrics', fontsize=14, fontweight='bold', pad=20)
        
        # Add current consciousness score
        if 'current_consciousness_score' in consciousness_report:
            score = consciousness_report['current_consciousness_score']
            ax.text(0.5, 1.1, f'Overall Score: {score:.3f}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color=self.colors['consciousness'])
    
    def _plot_system_health_gauge(self, system_report: Dict[str, Any], ax):
        """Plot system health as a gauge chart."""
        health = system_report.get('system_health', {})
        
        # Create gauge
        fig = ax.figure
        ax.remove()
        ax = fig.add_subplot(3, 4, 2, projection='polar')
        
        # Gauge parameters
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Health score (normalize to 0-1)
        uptime = health.get('uptime_seconds', 0)
        interactions = health.get('interaction_count', 0)
        memory_health = health.get('memory_system_health', {}).get('total_memories', 0)
        
        # Calculate health score
        health_score = min(1.0, (interactions * 0.3 + memory_health * 0.2 + min(uptime/3600, 1.0) * 0.5))
        
        # Plot gauge background
        ax.plot(theta, [r] * len(theta), color='lightgray', linewidth=10)
        
        # Plot health indicator
        health_angle = health_score * np.pi
        ax.plot(np.linspace(0, health_angle, 50), [r] * 50, 
               color=self.colors['system_health'], linewidth=10)
        
        # Add health score text
        ax.text(0, 0.5, f'{health_score:.1%}', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['system_health'])
        
        ax.set_ylim(0, 1.2)
        ax.set_title('System Health', fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
    
    def _plot_memory_performance(self, memory_stats: Dict[str, Any], ax):
        """Plot memory system performance metrics."""
        total_memories = memory_stats.get('total_memories', 0)
        queued_memories = memory_stats.get('queued_memories', 0)
        active_threads = memory_stats.get('active_threads', 0)
        processing_active = memory_stats.get('processing_active', False)
        
        # Create bar chart
        categories = ['Total Memories', 'Queued Memories', 'Active Threads']
        values = [total_memories, queued_memories, active_threads]
        colors = [self.colors['memory'], self.colors['performance'], self.colors['learning']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Memory System Performance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add processing status
        status_color = 'green' if processing_active else 'red'
        status_text = 'Active' if processing_active else 'Inactive'
        ax.text(0.5, 0.95, f'Processing: {status_text}', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, fontweight='bold', color=status_color)
    
    def _plot_knowledge_statistics(self, knowledge_stats: Dict[str, Any], ax):
        """Plot knowledge base statistics."""
        total_items = knowledge_stats.get('total_knowledge_items', 0)
        categories = knowledge_stats.get('categories', [])
        sources = knowledge_stats.get('sources', [])
        embeddings_available = knowledge_stats.get('embeddings_available', 0)
        total_usage = knowledge_stats.get('total_usage', 0)
        
        # Create metrics display
        metrics = [
            ('Total Items', total_items, self.colors['knowledge']),
            ('Categories', len(categories), self.colors['performance']),
            ('Sources', len(sources), self.colors['learning']),
            ('Embeddings', embeddings_available, self.colors['fine_tuning']),
            ('Total Usage', total_usage, self.colors['consciousness'])
        ]
        
        y_pos = np.arange(len(metrics))
        values = [metric[1] for metric in metrics]
        colors = [metric[2] for metric in metrics]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([metric[0] for metric in metrics])
        ax.set_xlabel('Count')
        ax.set_title('Knowledge Base Statistics', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value}', ha='left', va='center', fontweight='bold')
    
    def _plot_consciousness_timeline(self, consciousness_report: Dict[str, Any], ax):
        """Plot consciousness metrics over time."""
        if 'metrics_history' not in consciousness_report:
            ax.text(0.5, 0.5, 'No consciousness timeline data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consciousness Timeline', fontsize=14, fontweight='bold')
            return
        
        metrics = consciousness_report['metrics_history']
        if not metrics:
            ax.text(0.5, 0.5, 'No consciousness timeline data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consciousness Timeline', fontsize=14, fontweight='bold')
            return
        
        # Plot each metric over time
        for metric_name, values in metrics.items():
            if values:
                x = range(len(values))
                ax.plot(x, values, label=metric_name, alpha=0.8, linewidth=2)
        
        ax.set_title('Consciousness Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_memory_queue(self, memory_stats: Dict[str, Any], ax):
        """Plot memory processing queue status."""
        total_memories = memory_stats.get('total_memories', 0)
        queued_memories = memory_stats.get('queued_memories', 0)
        processed_memories = total_memories - queued_memories
        
        # Create pie chart
        sizes = [processed_memories, queued_memories]
        labels = ['Processed', 'Queued']
        colors = [self.colors['memory'], self.colors['performance']]
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax.text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Memory Processing Queue', fontsize=14, fontweight='bold')
    
    def _plot_knowledge_categories(self, knowledge_stats: Dict[str, Any], ax):
        """Plot knowledge categories distribution."""
        categories = knowledge_stats.get('categories', [])
        
        if not categories:
            ax.text(0.5, 0.5, 'No knowledge categories', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Knowledge Categories', fontsize=14, fontweight='bold')
            return
        
        # Count categories (simplified - in real system you'd track this)
        category_counts = [1] * len(categories)  # Placeholder
        
        bars = ax.bar(categories, category_counts, color=self.colors['knowledge'], alpha=0.8)
        ax.set_title('Knowledge Categories', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_performance_metrics(self, system_report: Dict[str, Any], ax):
        """Plot system performance metrics."""
        health = system_report.get('system_health', {})
        
        uptime = health.get('uptime_seconds', 0)
        interactions = health.get('interaction_count', 0)
        
        # Calculate performance metrics
        interactions_per_hour = (interactions / max(uptime, 1)) * 3600
        uptime_hours = uptime / 3600
        
        metrics = [
            ('Uptime (hrs)', uptime_hours, self.colors['system_health']),
            ('Interactions', interactions, self.colors['performance']),
            ('Interactions/hr', interactions_per_hour, self.colors['learning'])
        ]
        
        y_pos = np.arange(len(metrics))
        values = [metric[1] for metric in metrics]
        colors = [metric[2] for metric in metrics]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([metric[0] for metric in metrics])
        ax.set_xlabel('Value')
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}', ha='left', va='center', fontweight='bold')
    
    def _plot_fine_tuning_adapters(self, pft_system: ParameterEfficientFineTuning, ax):
        """Plot fine-tuning adapters information."""
        adapters = pft_system.list_adapters()
        
        if not adapters:
            ax.text(0.5, 0.5, 'No adapters available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fine-tuning Adapters', fontsize=14, fontweight='bold')
            return
        
        # Extract adapter information
        adapter_names = [adapter['adapter_id'] for adapter in adapters]
        training_steps = [adapter['training_steps'] for adapter in adapters]
        
        bars = ax.bar(adapter_names, training_steps, color=self.colors['fine_tuning'], alpha=0.8)
        ax.set_title('Fine-tuning Adapters', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Steps')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, training_steps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_safety_monitoring(self, system_report: Dict[str, Any], ax):
        """Plot safety monitoring metrics."""
        consciousness_report = system_report.get('consciousness_report', {})
        
        # Extract safety metrics
        consciousness_score = consciousness_report.get('current_consciousness_score', 0)
        consciousness_trend = consciousness_report.get('consciousness_trend', 'stable')
        
        # Create safety indicators
        indicators = [
            ('Consciousness Score', consciousness_score, self.colors['consciousness']),
            ('System Health', 0.8, self.colors['system_health']),  # Placeholder
            ('Memory Integrity', 0.9, self.colors['memory']),      # Placeholder
            ('Knowledge Safety', 0.95, self.colors['knowledge'])   # Placeholder
        ]
        
        y_pos = np.arange(len(indicators))
        values = [indicator[1] for indicator in indicators]
        colors = [indicator[2] for indicator in indicators]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([indicator[0] for indicator in indicators])
        ax.set_xlabel('Safety Score')
        ax.set_xlim(0, 1)
        ax.set_title('Safety Monitoring', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        # Add trend indicator
        trend_color = 'green' if consciousness_trend == 'increasing' else 'red' if consciousness_trend == 'decreasing' else 'orange'
        ax.text(0.5, 0.95, f'Trend: {consciousness_trend.title()}', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, fontweight='bold', color=trend_color)
    
    def _plot_interaction_history(self, system_report: Dict[str, Any], ax):
        """Plot interaction history."""
        health = system_report.get('system_health', {})
        interactions = health.get('interaction_count', 0)
        uptime = health.get('uptime_seconds', 0)
        
        # Create simple interaction timeline
        if uptime > 0:
            # Simulate interaction times (in real system you'd track this)
            interaction_times = np.linspace(0, uptime, max(interactions, 1))
            interaction_counts = np.arange(1, interactions + 1)
            
            ax.plot(interaction_times, interaction_counts, 
                   color=self.colors['performance'], linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Cumulative Interactions')
        else:
            ax.text(0.5, 0.5, 'No interactions yet', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Interaction History', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_system_architecture(self, ax):
        """Plot system architecture overview."""
        ax.axis('off')
        
        # Define component positions
        components = {
            'Consciousness\nMonitor': (0.2, 0.8),
            'Memory\nSystem': (0.8, 0.8),
            'External\nKnowledge': (0.2, 0.4),
            'Fine-tuning': (0.8, 0.4),
            'Safety\nMonitor': (0.5, 0.1)
        }
        
        # Draw components
        for name, pos in components.items():
            circle = Circle(pos, 0.08, color=self.colors['system_health'], alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], name, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Draw connections
        connections = [
            ((0.2, 0.8), (0.8, 0.8)),  # Consciousness to Memory
            ((0.2, 0.8), (0.2, 0.4)),  # Consciousness to Knowledge
            ((0.8, 0.8), (0.8, 0.4)),  # Memory to Fine-tuning
            ((0.2, 0.4), (0.5, 0.1)),  # Knowledge to Safety
            ((0.8, 0.4), (0.5, 0.1)),  # Fine-tuning to Safety
        ]
        
        for start, end in connections:
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color='gray', alpha=0.5, linewidth=2)
        
        ax.set_title('System Architecture', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def create_real_time_monitor(self, system: AdvancedAIMemorySystem, duration: int = 300):
        """Create a real-time monitoring visualization."""
        logger.info(f"Starting real-time monitoring for {duration} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Get current system data
                system_report = system.get_system_report()
                consciousness_report = system.consciousness_monitor.get_consciousness_report()
                
                # Store data for visualization
                timestamp = time.time() - start_time
                
                # Store consciousness data
                if 'current_consciousness_score' in consciousness_report:
                    self.consciousness_data.append({
                        'timestamp': timestamp,
                        'score': consciousness_report['current_consciousness_score']
                    })
                
                # Store memory data
                memory_stats = system.memory_system.get_memory_stats()
                self.memory_data.append({
                    'timestamp': timestamp,
                    'total_memories': memory_stats.get('total_memories', 0),
                    'queued_memories': memory_stats.get('queued_memories', 0)
                })
                
                # Store performance data
                health = system_report.get('system_health', {})
                self.performance_data.append({
                    'timestamp': timestamp,
                    'interactions': health.get('interaction_count', 0),
                    'uptime': health.get('uptime_seconds', 0)
                })
                
                # Create real-time visualization every 10 seconds
                if int(timestamp) % 10 == 0:
                    self._create_real_time_plot(timestamp)
                
                time.sleep(1)  # Update every second
                
            except KeyboardInterrupt:
                logger.info("Real-time monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
                time.sleep(5)
        
        logger.info("Real-time monitoring completed")
    
    def _create_real_time_plot(self, timestamp: float):
        """Create a real-time plot of system metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Advanced AI System - Real-time Monitor (Time: {timestamp:.0f}s)', 
                    fontsize=20, fontweight='bold')
        
        # Consciousness timeline
        if self.consciousness_data:
            times = [d['timestamp'] for d in self.consciousness_data]
            scores = [d['score'] for d in self.consciousness_data]
            ax1.plot(times, scores, color=self.colors['consciousness'], linewidth=2)
            ax1.set_title('Consciousness Score Over Time', fontweight='bold')
            ax1.set_ylabel('Consciousness Score')
            ax1.grid(True, alpha=0.3)
        
        # Memory processing
        if self.memory_data:
            times = [d['timestamp'] for d in self.memory_data]
            total = [d['total_memories'] for d in self.memory_data]
            queued = [d['queued_memories'] for d in self.memory_data]
            ax2.plot(times, total, label='Total Memories', color=self.colors['memory'])
            ax2.plot(times, queued, label='Queued Memories', color=self.colors['performance'])
            ax2.set_title('Memory Processing', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Performance metrics
        if self.performance_data:
            times = [d['timestamp'] for d in self.performance_data]
            interactions = [d['interactions'] for d in self.performance_data]
            ax3.plot(times, interactions, color=self.colors['learning'], linewidth=2)
            ax3.set_title('Cumulative Interactions', fontweight='bold')
            ax3.set_ylabel('Interactions')
            ax3.grid(True, alpha=0.3)
        
        # System health gauge
        if self.consciousness_data:
            latest_score = self.consciousness_data[-1]['score']
            ax4.remove()
            ax4 = fig.add_subplot(2, 2, 4, projection='polar')
            
            theta = np.linspace(0, np.pi, 100)
            r = 1
            ax4.plot(theta, [r] * len(theta), color='lightgray', linewidth=10)
            health_angle = latest_score * np.pi
            ax4.plot(np.linspace(0, health_angle, 50), [r] * 50, 
                    color=self.colors['system_health'], linewidth=10)
            ax4.text(0, 0.5, f'{latest_score:.1%}', ha='center', va='center', 
                    fontsize=20, fontweight='bold', color=self.colors['system_health'])
            ax4.set_ylim(0, 1.2)
            ax4.set_title('Current Consciousness Score', fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"realtime_monitor_{timestamp_str}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Real-time plot saved: {save_path}")


def main():
    """Main function to demonstrate the visualization system."""
    print("🎨 Advanced AI Memory and Consciousness Monitoring System - Visualization")
    print("=" * 80)
    
    # Initialize system
    config = AIConfig()
    system = AdvancedAIMemorySystem(config)
    
    # Initialize visualizer
    visualizer = AdvancedAIVisualizer()
    
    try:
        # Start system
        system.start_system()
        print("✅ System started successfully")
        
        # Process some interactions to generate data
        test_interactions = [
            "What is AI consciousness?",
            "How does memory work in AI systems?",
            "Tell me about parameter-efficient fine-tuning",
            "What are AI safety considerations?"
        ]
        
        for i, interaction in enumerate(test_interactions, 1):
            print(f"Processing interaction {i}/{len(test_interactions)}...")
            result = system.process_interaction(interaction)
            time.sleep(2)  # Wait between interactions
        
        # Create comprehensive dashboard
        print("Creating comprehensive dashboard...")
        dashboard_path = visualizer.create_comprehensive_dashboard(system)
        print(f"✅ Dashboard saved to: {dashboard_path}")
        
        # Create real-time monitor (run for 30 seconds)
        print("Starting real-time monitoring (30 seconds)...")
        visualizer.create_real_time_monitor(system, duration=30)
        
        print("🎉 Visualization demonstration completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Visualization stopped by user")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        print(f"❌ Visualization failed: {e}")
    finally:
        system.stop_system()
        print("✅ System shutdown complete")


if __name__ == "__main__":
    main() 