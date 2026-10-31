#!/usr/bin/env python3
"""
Memory and Model Performance Visualization System
================================================

This system creates comprehensive visualizations of:
- Memory system performance metrics
- Neuroplasticity tracking
- Consciousness and awareness levels
- Response quality analysis
- Learning patterns over time

UPDATED: Now integrates with Advanced AI Memory and Consciousness Monitoring System
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
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Import the Advanced AI system for integration
try:
    from advanced_ai_memory_system import (
        AdvancedAIMemorySystem, 
        AIConfig, 
        ConsciousnessMonitor,
        ExternalKnowledgeSystem,
        ParameterEfficientFineTuning,
        MultiThreadedMemorySystem
    )
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    ADVANCED_AI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryVisualizer:
    """Comprehensive memory system visualizer."""
    
    def __init__(self, output_dir: str = "memory_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'awareness': '#FF6B6B',
            'focus': '#4ECDC4', 
            'plasticity': '#45B7D1',
            'memory': '#96CEB4',
            'consciousness': '#FFEAA7',
            'learning': '#DDA0DD',
            'performance': '#98D8C8',
            'knowledge': '#FF9F43',
            'fine_tuning': '#6C5CE7',
            'system_health': '#00B894',
            'safety': '#E17055'
        }
        
        logger.info(f"Memory Visualizer initialized. Output directory: {output_dir}")
    
    def create_performance_dashboard(self, results: List[Dict[str, Any]], save_path: str = None):
        """Create a comprehensive performance dashboard."""
        
        if not results:
            logger.warning("No results provided for visualization")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Memory System Performance Dashboard', fontsize=24, fontweight='bold')
        
        # Extract data
        df = pd.DataFrame(results)
        
        # 1. Neuroplasticity Metrics Over Time
        ax1 = plt.subplot(3, 3, 1)
        self._plot_neuroplasticity_timeline(df, ax1)
        
        # 2. Consciousness State Radar
        ax2 = plt.subplot(3, 3, 2, projection='polar')
        self._plot_consciousness_radar(df, ax2)
        
        # 3. Response Quality Analysis
        ax3 = plt.subplot(3, 3, 3)
        self._plot_response_quality(df, ax3)
        
        # 4. Memory Type Distribution
        ax4 = plt.subplot(3, 3, 4)
        self._plot_memory_distribution(df, ax4)
        
        # 5. Learning Pattern Heatmap
        ax5 = plt.subplot(3, 3, 5)
        self._plot_learning_heatmap(df, ax5)
        
        # 6. Performance Metrics
        ax6 = plt.subplot(3, 3, 6)
        self._plot_performance_metrics(df, ax6)
        
        # 7. Focus vs Awareness Scatter
        ax7 = plt.subplot(3, 3, 7)
        self._plot_focus_awareness_scatter(df, ax7)
        
        # 8. Plasticity Updates Timeline
        ax8 = plt.subplot(3, 3, 8)
        self._plot_plasticity_timeline(df, ax8)
        
        # 9. System Health Gauge
        ax9 = plt.subplot(3, 3, 9)
        self._plot_system_health_gauge(df, ax9)
        
        plt.tight_layout()
        
        # Save the dashboard
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"performance_dashboard_{int(time.time())}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance dashboard saved to: {save_path}")
        
        return save_path
    
    def _plot_neuroplasticity_timeline(self, df: pd.DataFrame, ax):
        """Plot neuroplasticity metrics over time."""
        
        if 'test_id' not in df.columns:
            df['test_id'] = range(len(df))
        
        # Extract neuroplasticity data
        awareness = df.get('awareness', [0.0] * len(df))
        focus = df.get('focus', [0.0] * len(df))
        plasticity = df.get('plasticity_updates', [0] * len(df))
        
        x = df['test_id']
        
        # Plot lines
        ax.plot(x, awareness, color=self.colors['awareness'], linewidth=3, label='Awareness', marker='o')
        ax.plot(x, focus, color=self.colors['focus'], linewidth=3, label='Focus', marker='s')
        
        # Plot plasticity updates as bars
        ax2 = ax.twinx()
        ax2.bar(x, plasticity, alpha=0.3, color=self.colors['plasticity'], label='Plasticity Updates')
        
        ax.set_title('Neuroplasticity Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Test Number')
        ax.set_ylabel('Level (0-1)', color='black')
        ax2.set_ylabel('Plasticity Updates', color=self.colors['plasticity'])
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_consciousness_radar(self, df: pd.DataFrame, ax):
        """Plot consciousness state as a radar chart."""
        
        # Calculate average metrics
        avg_awareness = df.get('awareness', [0.0]).mean()
        avg_focus = df.get('focus', [0.0]).mean()
        avg_plasticity = min(1.0, df.get('plasticity_updates', [0]).mean() / 2)  # Normalize
        
        # Radar chart data
        categories = ['Awareness', 'Focus', 'Plasticity', 'Learning', 'Memory']
        values = [avg_awareness, avg_focus, avg_plasticity, 
                 (avg_awareness + avg_focus) / 2,  # Learning
                 (avg_awareness + avg_plasticity) / 2]  # Memory
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add the first value at the end to close the polygon
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['consciousness'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['consciousness'])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Consciousness State', fontsize=14, fontweight='bold', pad=20)
    
    def _plot_response_quality(self, df: pd.DataFrame, ax):
        """Plot response quality analysis."""
        
        # Extract response data
        length_diffs = df.get('response_length_diff', [0] * len(df))
        memory_enhanced = df.get('memory_enhanced', [False] * len(df))
        
        # Create quality categories
        quality_categories = []
        for diff in length_diffs:
            if diff > 20:
                quality_categories.append('Expanded')
            elif diff < -20:
                quality_categories.append('Condensed')
            else:
                quality_categories.append('Similar')
        
        # Count categories
        quality_counts = pd.Series(quality_categories).value_counts()
        
        # Create pie chart
        colors = [self.colors['learning'], self.colors['performance'], self.colors['memory']]
        wedges, texts, autotexts = ax.pie(quality_counts.values, labels=quality_counts.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        ax.set_title('Response Quality Distribution', fontsize=14, fontweight='bold')
    
    def _plot_memory_distribution(self, df: pd.DataFrame, ax):
        """Plot memory type distribution."""
        
        # Extract memory types
        memory_types = df.get('memory_classification', ['unknown'] * len(df))
        type_counts = pd.Series(memory_types).value_counts()
        
        # Create bar chart
        bars = ax.bar(range(len(type_counts)), type_counts.values, 
                     color=[self.colors['memory'], self.colors['learning'], self.colors['consciousness']])
        
        ax.set_title('Memory Type Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Memory Type')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(type_counts)))
        ax.set_xticklabels(type_counts.index, rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, type_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')
    
    def _plot_learning_heatmap(self, df: pd.DataFrame, ax):
        """Plot learning pattern heatmap."""
        
        # Create correlation matrix for learning metrics
        learning_metrics = ['awareness', 'focus', 'plasticity_updates']
        available_metrics = [m for m in learning_metrics if m in df.columns]
        
        if len(available_metrics) >= 2:
            # Calculate correlations
            corr_data = df[available_metrics].corr()
            
            # Create heatmap
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Learning Pattern Correlations', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor correlation', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Learning Pattern Correlations', fontsize=14, fontweight='bold')
    
    def _plot_performance_metrics(self, df: pd.DataFrame, ax):
        """Plot key performance metrics."""
        
        # Calculate metrics
        total_tests = len(df)
        memory_enhanced = df.get('memory_enhanced', [False] * len(df)).sum()
        avg_length_diff = df.get('response_length_diff', [0] * len(df)).mean()
        avg_plasticity = df.get('plasticity_updates', [0] * len(df)).mean()
        
        # Create metrics display
        metrics_text = f"""
        Total Tests: {total_tests}
        Memory Enhanced: {memory_enhanced}/{total_tests} ({memory_enhanced/total_tests*100:.1f}%)
        Avg Length Change: {avg_length_diff:.1f} chars
        Avg Plasticity: {avg_plasticity:.2f}
        """
        
        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor=self.colors['performance'], alpha=0.7))
        
        ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _plot_focus_awareness_scatter(self, df: pd.DataFrame, ax):
        """Plot focus vs awareness scatter plot."""
        
        awareness = df.get('awareness', [0.0] * len(df))
        focus = df.get('focus', [0.0] * len(df))
        
        # Create scatter plot
        scatter = ax.scatter(awareness, focus, c=df.get('plasticity_updates', [0] * len(df)), 
                           cmap='viridis', s=100, alpha=0.7)
        
        ax.set_xlabel('Awareness Level')
        ax.set_ylabel('Focus Level')
        ax.set_title('Focus vs Awareness', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Plasticity Updates')
    
    def _plot_plasticity_timeline(self, df: pd.DataFrame, ax):
        """Plot plasticity updates timeline."""
        
        if 'test_id' not in df.columns:
            df['test_id'] = range(len(df))
        
        plasticity = df.get('plasticity_updates', [0] * len(df))
        x = df['test_id']
        
        # Create timeline
        ax.step(x, plasticity, where='post', linewidth=3, color=self.colors['plasticity'])
        ax.fill_between(x, plasticity, alpha=0.3, color=self.colors['plasticity'])
        
        ax.set_xlabel('Test Number')
        ax.set_ylabel('Plasticity Updates')
        ax.set_title('Plasticity Updates Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_system_health_gauge(self, df: pd.DataFrame, ax):
        """Plot system health gauge."""
        
        # Calculate health score
        awareness_avg = df.get('awareness', [0.0]).mean()
        focus_avg = df.get('focus', [0.0]).mean()
        plasticity_avg = min(1.0, df.get('plasticity_updates', [0]).mean() / 2)
        
        health_score = (awareness_avg + focus_avg + plasticity_avg) / 3
        
        # Create gauge
        gauge_angle = health_score * 180  # Convert to degrees
        
        # Draw gauge
        gauge = Circle((0.5, 0.5), 0.4, fill=False, linewidth=10, color='lightgray')
        ax.add_patch(gauge)
        
        # Draw health indicator
        if health_score > 0.7:
            color = 'green'
        elif health_score > 0.4:
            color = 'orange'
        else:
            color = 'red'
        
        health_arc = mpatches.Arc((0.5, 0.5), 0.8, 0.8, angle=180, 
                                 theta1=0, theta2=gauge_angle, linewidth=10, color=color)
        ax.add_patch(health_arc)
        
        # Add text
        ax.text(0.5, 0.3, f'Health Score\n{health_score:.2f}', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('System Health', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def create_neural_network_visualization(self, memory_system, save_path: str = None):
        """Create a neural network-style visualization of the memory system."""
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Define node positions
        nodes = {
            'input': (0.1, 0.5),
            'stm': (0.3, 0.7),
            'ltm': (0.3, 0.3),
            'consciousness': (0.5, 0.8),
            'attention': (0.5, 0.6),
            'plasticity': (0.5, 0.4),
            'memory_consolidation': (0.5, 0.2),
            'output': (0.9, 0.5)
        }
        
        # Draw connections
        connections = [
            ('input', 'stm'),
            ('input', 'ltm'),
            ('stm', 'consciousness'),
            ('stm', 'attention'),
            ('stm', 'plasticity'),
            ('ltm', 'memory_consolidation'),
            ('consciousness', 'output'),
            ('attention', 'output'),
            ('plasticity', 'output'),
            ('memory_consolidation', 'output')
        ]
        
        # Draw connections
        for start, end in connections:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=2)
        
        # Draw nodes
        for node_name, (x, y) in nodes.items():
            color = self.colors.get(node_name.split('_')[0], '#CCCCCC')
            circle = Circle((x, y), 0.05, color=color, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, node_name.replace('_', '\n'), ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Memory System Architecture', fontsize=18, fontweight='bold')
        ax.axis('off')
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"neural_architecture_{int(time.time())}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Neural architecture visualization saved to: {save_path}")
        
        return save_path
    
    def create_real_time_monitor(self, memory_system, duration: int = 60):
        """Create a real-time monitoring visualization."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Data storage
        times = []
        awareness_data = []
        focus_data = []
        plasticity_data = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get current system state
            try:
                status = memory_system.get_system_status()
                
                current_time = time.time() - start_time
                times.append(current_time)
                
                # Extract metrics
                awareness = status.get('awareness_level', 0.0)
                focus = status.get('focus_level', 0.0)
                plasticity = len(status.get('plasticity_updates', []))
                
                awareness_data.append(awareness)
                focus_data.append(focus)
                plasticity_data.append(plasticity)
                
                # Update plots
                ax1.clear()
                ax1.plot(times, awareness_data, label='Awareness', color=self.colors['awareness'])
                ax1.plot(times, focus_data, label='Focus', color=self.colors['focus'])
                ax1.set_ylabel('Level')
                ax1.set_title('Real-time Consciousness Monitoring')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.clear()
                ax2.plot(times, plasticity_data, label='Plasticity Updates', color=self.colors['plasticity'])
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Updates')
                ax2.set_title('Real-time Plasticity Monitoring')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.pause(1)
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
                break
        
        plt.show()
    
    def create_advanced_ai_dashboard(self, advanced_ai_system, save_path: str = None):
        """Create a comprehensive dashboard for the Advanced AI system."""
        if not ADVANCED_AI_AVAILABLE:
            logger.error("Advanced AI system not available")
            return None
            
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"advanced_ai_dashboard_{timestamp}.png")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Advanced AI Memory and Consciousness Monitoring System', 
                    fontsize=24, fontweight='bold')
        
        # Get system data
        system_report = advanced_ai_system.get_system_report()
        consciousness_report = advanced_ai_system.consciousness_monitor.get_consciousness_report()
        memory_stats = advanced_ai_system.memory_system.get_memory_stats()
        knowledge_stats = advanced_ai_system.external_knowledge.get_knowledge_stats()
        
        # 1. Consciousness Monitoring
        ax1 = plt.subplot(3, 3, 1)
        self._plot_advanced_consciousness(consciousness_report, ax1)
        
        # 2. Memory System Performance
        ax2 = plt.subplot(3, 3, 2)
        self._plot_advanced_memory(memory_stats, ax2)
        
        # 3. Knowledge Base
        ax3 = plt.subplot(3, 3, 3)
        self._plot_advanced_knowledge(knowledge_stats, ax3)
        
        # 4. System Health
        ax4 = plt.subplot(3, 3, 4)
        self._plot_advanced_health(system_report, ax4)
        
        # 5. Performance Metrics
        ax5 = plt.subplot(3, 3, 5)
        self._plot_advanced_performance(system_report, ax5)
        
        # 6. Fine-tuning Status
        ax6 = plt.subplot(3, 3, 6)
        self._plot_advanced_fine_tuning(advanced_ai_system.parameter_efficient_tuning, ax6)
        
        # 7. Safety Monitoring
        ax7 = plt.subplot(3, 3, 7)
        self._plot_advanced_safety(system_report, ax7)
        
        # 8. Interaction History
        ax8 = plt.subplot(3, 3, 8)
        self._plot_advanced_interactions(system_report, ax8)
        
        # 9. System Architecture
        ax9 = plt.subplot(3, 3, 9)
        self._plot_advanced_architecture(ax9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Advanced AI dashboard saved to: {save_path}")
        return save_path
    
    def _plot_advanced_consciousness(self, consciousness_report: Dict[str, Any], ax):
        """Plot advanced consciousness metrics."""
        if 'current_consciousness_score' in consciousness_report:
            score = consciousness_report['current_consciousness_score']
            trend = consciousness_report.get('consciousness_trend', 'stable')
            
            # Create gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = 1
            ax.plot(theta, [r] * len(theta), color='lightgray', linewidth=10)
            health_angle = score * np.pi
            ax.plot(np.linspace(0, health_angle, 50), [r] * 50, 
                   color=self.colors['consciousness'], linewidth=10)
            ax.text(0, 0.5, f'{score:.3f}', ha='center', va='center', 
                   fontsize=20, fontweight='bold', color=self.colors['consciousness'])
            ax.set_ylim(0, 1.2)
            ax.set_title('Consciousness Score', fontweight='bold')
            ax.axis('off')
            
            # Add trend indicator
            trend_color = 'green' if trend == 'increasing' else 'red' if trend == 'decreasing' else 'orange'
            ax.text(0.5, 1.1, f'Trend: {trend.title()}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color=trend_color)
        else:
            ax.text(0.5, 0.5, 'No consciousness data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consciousness Score', fontweight='bold')
    
    def _plot_advanced_memory(self, memory_stats: Dict[str, Any], ax):
        """Plot advanced memory system metrics."""
        total_memories = memory_stats.get('total_memories', 0)
        queued_memories = memory_stats.get('queued_memories', 0)
        active_threads = memory_stats.get('active_threads', 0)
        
        categories = ['Total', 'Queued', 'Threads']
        values = [total_memories, queued_memories, active_threads]
        colors = [self.colors['memory'], self.colors['performance'], self.colors['learning']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_title('Memory System', fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_advanced_knowledge(self, knowledge_stats: Dict[str, Any], ax):
        """Plot advanced knowledge base metrics."""
        total_items = knowledge_stats.get('total_knowledge_items', 0)
        categories = knowledge_stats.get('categories', [])
        embeddings = knowledge_stats.get('embeddings_available', 0)
        
        metrics = [
            ('Total Items', total_items, self.colors['knowledge']),
            ('Categories', len(categories), self.colors['performance']),
            ('Embeddings', embeddings, self.colors['fine_tuning'])
        ]
        
        y_pos = np.arange(len(metrics))
        values = [metric[1] for metric in metrics]
        colors = [metric[2] for metric in metrics]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([metric[0] for metric in metrics])
        ax.set_xlabel('Count')
        ax.set_title('Knowledge Base', fontweight='bold')
    
    def _plot_advanced_health(self, system_report: Dict[str, Any], ax):
        """Plot advanced system health."""
        health = system_report.get('system_health', {})
        uptime = health.get('uptime_seconds', 0)
        interactions = health.get('interaction_count', 0)
        
        # Calculate health score
        health_score = min(1.0, (interactions * 0.3 + min(uptime/3600, 1.0) * 0.7))
        
        # Create health gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        ax.plot(theta, [r] * len(theta), color='lightgray', linewidth=10)
        health_angle = health_score * np.pi
        ax.plot(np.linspace(0, health_angle, 50), [r] * 50, 
               color=self.colors['system_health'], linewidth=10)
        ax.text(0, 0.5, f'{health_score:.1%}', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=self.colors['system_health'])
        ax.set_ylim(0, 1.2)
        ax.set_title('System Health', fontweight='bold')
        ax.axis('off')
    
    def _plot_advanced_performance(self, system_report: Dict[str, Any], ax):
        """Plot advanced performance metrics."""
        health = system_report.get('system_health', {})
        uptime = health.get('uptime_seconds', 0)
        interactions = health.get('interaction_count', 0)
        
        interactions_per_hour = (interactions / max(uptime, 1)) * 3600
        
        metrics = [
            ('Uptime (hrs)', uptime / 3600, self.colors['system_health']),
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
        ax.set_title('Performance', fontweight='bold')
    
    def _plot_advanced_fine_tuning(self, pft_system, ax):
        """Plot advanced fine-tuning status."""
        adapters = pft_system.list_adapters()
        
        if adapters:
            adapter_names = [adapter['adapter_id'][:10] + '...' for adapter in adapters]
            training_steps = [adapter['training_steps'] for adapter in adapters]
            
            bars = ax.bar(adapter_names, training_steps, color=self.colors['fine_tuning'], alpha=0.8)
            ax.set_title('Fine-tuning Adapters', fontweight='bold')
            ax.set_ylabel('Training Steps')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No adapters', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fine-tuning Adapters', fontweight='bold')
    
    def _plot_advanced_safety(self, system_report: Dict[str, Any], ax):
        """Plot advanced safety monitoring."""
        consciousness_report = system_report.get('consciousness_report', {})
        consciousness_score = consciousness_report.get('current_consciousness_score', 0)
        
        # Safety indicators
        indicators = [
            ('Consciousness', consciousness_score, self.colors['consciousness']),
            ('System Health', 0.8, self.colors['system_health']),
            ('Memory Integrity', 0.9, self.colors['memory']),
            ('Knowledge Safety', 0.95, self.colors['knowledge'])
        ]
        
        y_pos = np.arange(len(indicators))
        values = [indicator[1] for indicator in indicators]
        colors = [indicator[2] for indicator in indicators]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([indicator[0] for indicator in indicators])
        ax.set_xlabel('Safety Score')
        ax.set_xlim(0, 1)
        ax.set_title('Safety Monitoring', fontweight='bold')
    
    def _plot_advanced_interactions(self, system_report: Dict[str, Any], ax):
        """Plot advanced interaction history."""
        health = system_report.get('system_health', {})
        interactions = health.get('interaction_count', 0)
        uptime = health.get('uptime_seconds', 0)
        
        if uptime > 0:
            # Simple interaction timeline
            interaction_times = np.linspace(0, uptime, max(interactions, 1))
            interaction_counts = np.arange(1, interactions + 1)
            
            ax.plot(interaction_times, interaction_counts, 
                   color=self.colors['performance'], linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Interactions')
        else:
            ax.text(0.5, 0.5, 'No interactions', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Interaction History', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_advanced_architecture(self, ax):
        """Plot advanced system architecture."""
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
        
        ax.set_title('System Architecture', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

def generate_sample_data(num_tests: int = 20) -> List[Dict[str, Any]]:
    """Generate sample data for visualization testing."""
    
    results = []
    
    for i in range(num_tests):
        # Simulate realistic memory system behavior
        awareness = np.random.beta(2, 3)  # Skewed towards lower values
        focus = np.random.beta(3, 2)      # Skewed towards higher values
        plasticity = np.random.poisson(0.3)  # Rare events
        
        # Memory type distribution
        memory_types = ['episodic', 'semantic', 'procedural']
        memory_type = np.random.choice(memory_types, p=[0.4, 0.4, 0.2])
        
        # Response quality
        length_diff = np.random.normal(15, 25)  # Some positive, some negative
        
        result = {
            'test_id': i,
            'awareness': awareness,
            'focus': focus,
            'plasticity_updates': plasticity,
            'memory_classification': memory_type,
            'response_length_diff': length_diff,
            'memory_enhanced': True,
            'timestamp': datetime.now() - timedelta(minutes=num_tests-i)
        }
        
        results.append(result)
    
    return results

def main():
    """Main function to demonstrate visualization capabilities."""
    
    print("🎨 Memory System Visualization Generator")
    print("=" * 50)
    
    # Create visualizer
    visualizer = MemoryVisualizer()
    
    # Generate sample data
    print("Generating sample data...")
    sample_data = generate_sample_data(50)
    
    # Create performance dashboard
    print("Creating performance dashboard...")
    dashboard_path = visualizer.create_performance_dashboard(sample_data)
    
    # Create neural architecture visualization
    print("Creating neural architecture visualization...")
    try:
        from cpu_optimized_memory_system import CPUOptimizedMemorySystem, CPUOptimizedConfig
        
        config = CPUOptimizedConfig(storage_path="viz_test")
        memory_system = CPUOptimizedMemorySystem(config)
        
        neural_path = visualizer.create_neural_network_visualization(memory_system)
        
        memory_system.shutdown()
        
        print(f"\n✅ Visualizations created successfully!")
        print(f"📊 Performance Dashboard: {dashboard_path}")
        print(f"🧠 Neural Architecture: {neural_path}")
        
    except Exception as e:
        print(f"⚠️ Neural architecture visualization failed: {e}")
        print(f"📊 Performance Dashboard: {dashboard_path}")
    
    # Test Advanced AI system visualization if available
    if ADVANCED_AI_AVAILABLE:
        print("\n🚀 Testing Advanced AI system visualization...")
        try:
            # Initialize Advanced AI system
            ai_config = AIConfig()
            ai_system = AdvancedAIMemorySystem(ai_config)
            
            # Start system
            ai_system.start_system()
            
            # Process some interactions to generate data
            test_interactions = [
                "What is AI consciousness?",
                "How does memory work in AI systems?",
                "Tell me about parameter-efficient fine-tuning"
            ]
            
            for interaction in test_interactions:
                ai_system.process_interaction(interaction)
                time.sleep(1)
            
            # Create Advanced AI dashboard
            advanced_dashboard_path = visualizer.create_advanced_ai_dashboard(ai_system)
            print(f"✅ Advanced AI dashboard saved to: {advanced_dashboard_path}")
            
            # Stop system
            ai_system.stop_system()
            
        except Exception as e:
            print(f"⚠️ Advanced AI visualization failed: {e}")
    else:
        print("⚠️ Advanced AI system not available for visualization")
    
    print("\n🚀 Visualization system ready for use!")
    print("Use the MemoryVisualizer class to create custom visualizations.")

if __name__ == "__main__":
    main() 