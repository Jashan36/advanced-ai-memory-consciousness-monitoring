#!/usr/bin/env python3
"""
Test Script for Advanced AI Visualization System
===============================================

This script demonstrates the visualization capabilities for the Advanced AI Memory and Consciousness Monitoring System.
"""

import time
import logging
from memory_visualization import MemoryVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_advanced_ai_visualization():
    """Test the Advanced AI system visualization."""
    print("🎨 Testing Advanced AI Visualization System")
    print("=" * 60)
    
    try:
        # Import the Advanced AI system
        from advanced_ai_memory_system import AdvancedAIMemorySystem, AIConfig
        
        # Initialize visualizer
        visualizer = MemoryVisualizer(output_dir="advanced_ai_visualizations")
        
        # Initialize Advanced AI system
        print("🚀 Initializing Advanced AI system...")
        config = AIConfig()
        system = AdvancedAIMemorySystem(config)
        
        # Start system
        system.start_system()
        print("✅ System started successfully")
        
        # Process interactions to generate data
        test_interactions = [
            "What is AI consciousness?",
            "How does memory work in AI systems?",
            "Tell me about parameter-efficient fine-tuning",
            "What are AI safety considerations?",
            "How does external knowledge integration work?",
            "What is multi-threaded memory processing?",
            "Explain consciousness monitoring in AI",
            "How does the safety monitoring system work?"
        ]
        
        print(f"📝 Processing {len(test_interactions)} interactions...")
        for i, interaction in enumerate(test_interactions, 1):
            print(f"  {i}/{len(test_interactions)}: {interaction[:50]}...")
            result = system.process_interaction(interaction)
            
            # Show some metrics
            consciousness_score = result['consciousness_metrics'].get('current_consciousness_score', 'N/A')
            knowledge_used = len(result['knowledge_used'])
            print(f"    Consciousness: {consciousness_score}, Knowledge: {knowledge_used} items")
            
            time.sleep(0.5)  # Brief pause between interactions
        
        # Create comprehensive dashboard
        print("\n📊 Creating Advanced AI dashboard...")
        dashboard_path = visualizer.create_advanced_ai_dashboard(system)
        print(f"✅ Dashboard saved to: {dashboard_path}")
        
        # Get system report
        print("\n📋 System Report:")
        report = system.get_system_report()
        
        # Print key metrics
        health = report['system_health']
        consciousness = report['consciousness_report']
        
        print(f"  System Name: {report['system_name']}")
        print(f"  Version: {report['version']}")
        print(f"  Uptime: {health['uptime_seconds']:.1f} seconds")
        print(f"  Interactions: {health['interaction_count']}")
        print(f"  Consciousness Score: {consciousness.get('current_consciousness_score', 'N/A')}")
        print(f"  Memory Processed: {report['memory_stats']['total_memories']}")
        print(f"  Knowledge Items: {report['knowledge_stats']['total_knowledge_items']}")
        
        # Stop system
        system.stop_system()
        print("\n✅ System stopped successfully")
        
        print(f"\n🎉 Advanced AI visualization test completed!")
        print(f"📁 Check the 'advanced_ai_visualizations' directory for generated images")
        
        return True
        
    except ImportError as e:
        print(f"❌ Advanced AI system not available: {e}")
        print("Make sure you have the advanced_ai_memory_system.py file in your directory")
        return False
        
    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

def test_basic_visualization():
    """Test basic visualization capabilities."""
    print("\n🎨 Testing Basic Visualization System")
    print("=" * 60)
    
    try:
        # Create visualizer
        visualizer = MemoryVisualizer()
        
        # Generate sample data
        print("📊 Generating sample data...")
        from memory_visualization import generate_sample_data
        sample_data = generate_sample_data(30)
        
        # Create performance dashboard
        print("📈 Creating performance dashboard...")
        dashboard_path = visualizer.create_performance_dashboard(sample_data)
        print(f"✅ Dashboard saved to: {dashboard_path}")
        
        print("✅ Basic visualization test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Basic visualization test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function to run visualization tests."""
    print("🧪 Advanced AI Visualization Test Suite")
    print("=" * 80)
    
    # Test basic visualization
    basic_success = test_basic_visualization()
    
    # Test Advanced AI visualization
    advanced_success = test_advanced_ai_visualization()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary:")
    print(f"  Basic Visualization: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"  Advanced AI Visualization: {'✅ PASSED' if advanced_success else '❌ FAILED'}")
    
    if basic_success and advanced_success:
        print("\n🎉 All tests passed! Visualization system is working correctly.")
    elif basic_success:
        print("\n⚠️ Basic visualization works, but Advanced AI system needs setup.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    print("\n📁 Generated visualizations are saved in:")
    print("  - memory_visualizations/ (basic visualizations)")
    print("  - advanced_ai_visualizations/ (Advanced AI system visualizations)")

if __name__ == "__main__":
    main() 