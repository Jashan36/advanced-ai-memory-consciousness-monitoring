#!/usr/bin/env python3
"""
Test Script for Advanced AI Memory and Consciousness Monitoring System
=====================================================================

This script demonstrates the system's capabilities and provides a comprehensive test suite.
"""

import json
import time
import logging
from datetime import datetime
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


def test_consciousness_monitoring():
    """Test consciousness monitoring capabilities."""
    print("\n🧠 Testing Consciousness Monitoring")
    print("=" * 50)
    
    config = AIConfig()
    monitor = ConsciousnessMonitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Let it run for a few seconds
    print("Starting consciousness monitoring...")
    time.sleep(5)
    
    # Get consciousness report
    report = monitor.get_consciousness_report()
    
    print(f"Current consciousness score: {report.get('current_consciousness_score', 'N/A')}")
    print(f"Average consciousness score: {report.get('average_consciousness_score', 'N/A')}")
    print(f"Consciousness trend: {report.get('consciousness_trend', 'N/A')}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("✅ Consciousness monitoring test completed")
    return report


def test_external_knowledge_system():
    """Test external knowledge integration."""
    print("\n📚 Testing External Knowledge System")
    print("=" * 50)
    
    config = AIConfig()
    knowledge_system = ExternalKnowledgeSystem(config)
    
    # Add some test knowledge
    test_knowledge = [
        {
            "text": "AI consciousness is a complex topic that involves understanding how AI systems process information and maintain awareness.",
            "source": "AI Research Paper",
            "category": "consciousness"
        },
        {
            "text": "Parameter-efficient fine-tuning allows AI models to adapt to new tasks without full retraining.",
            "source": "Machine Learning Journal",
            "category": "technology"
        },
        {
            "text": "Memory-augmented neural networks can improve AI system performance by maintaining context across interactions.",
            "source": "Neural Networks Research",
            "category": "memory"
        },
        {
            "text": "AI safety research focuses on ensuring AI systems behave safely and align with human values.",
            "source": "AI Safety Foundation",
            "category": "safety"
        }
    ]
    
    for knowledge in test_knowledge:
        knowledge_system.add_knowledge(
            knowledge["text"],
            knowledge["source"],
            knowledge["category"]
        )
    
    # Test knowledge search
    search_queries = [
        "AI consciousness",
        "fine-tuning",
        "memory systems",
        "AI safety"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = knowledge_system.search_knowledge(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['text'][:100]}... (similarity: {result['similarity']:.3f})")
        else:
            print("  No results found")
    
    # Get knowledge statistics
    stats = knowledge_system.get_knowledge_stats()
    print(f"\nKnowledge base statistics:")
    print(f"  Total items: {stats['total_knowledge_items']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Sources: {stats['sources']}")
    
    print("✅ External knowledge system test completed")
    return knowledge_system


def test_parameter_efficient_fine_tuning():
    """Test parameter-efficient fine-tuning capabilities."""
    print("\n🔧 Testing Parameter-Efficient Fine-Tuning")
    print("=" * 50)
    
    config = AIConfig()
    pft = ParameterEfficientFineTuning(config)
    
    # Create a LoRA adapter
    adapter_id = pft.create_lora_adapter("gpt2")
    print(f"Created LoRA adapter: {adapter_id}")
    
    # Prepare training data
    training_data = [
        {
            "input": "What is artificial intelligence?",
            "output": "Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."
        },
        {
            "input": "How does machine learning work?",
            "output": "Machine learning works by training algorithms on data to recognize patterns and make predictions or decisions without being explicitly programmed for specific tasks."
        },
        {
            "input": "What is deep learning?",
            "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
        }
    ]
    
    # Train the adapter
    print("Training LoRA adapter...")
    pft.train_adapter(adapter_id, training_data, epochs=2, learning_rate=1e-4)
    
    # Get adapter information
    adapter_info = pft.get_adapter_info(adapter_id)
    print(f"Training completed. Total steps: {adapter_info['adapter_config']['training_steps']}")
    
    # List all adapters
    adapters = pft.list_adapters()
    print(f"Available adapters: {len(adapters)}")
    for adapter in adapters:
        print(f"  - {adapter['adapter_id']}: {adapter['model_name']} ({adapter['training_steps']} steps)")
    
    print("✅ Parameter-efficient fine-tuning test completed")
    return pft


def test_multi_threaded_memory_system():
    """Test multi-threaded memory processing."""
    print("\n🧵 Testing Multi-Threaded Memory System")
    print("=" * 50)
    
    config = AIConfig()
    memory_system = MultiThreadedMemorySystem(config)
    
    # Start memory processing
    memory_system.start_processing(num_threads=4)
    print("Started memory processing with 4 threads")
    
    # Add memories with different priorities
    test_memories = [
        ("high_priority", {"data": "Critical information", "type": "urgent"}, 10),
        ("medium_priority", {"data": "Important information", "type": "normal"}, 5),
        ("low_priority", {"data": "General information", "type": "background"}, 1),
        ("research_data", {"data": "Research findings", "type": "analysis"}, 8),
        ("user_preference", {"data": "User preference data", "type": "personal"}, 6)
    ]
    
    for memory_id, data, priority in test_memories:
        memory_system.add_memory(memory_id, data, priority)
        print(f"Added memory: {memory_id} (priority: {priority})")
    
    # Wait for processing
    print("Waiting for memory processing...")
    time.sleep(3)
    
    # Retrieve memories
    for memory_id, _, _ in test_memories:
        memory = memory_system.get_memory(memory_id)
        if memory:
            print(f"Retrieved memory: {memory_id} - {memory['original_data']['data']}")
        else:
            print(f"Memory not found: {memory_id}")
    
    # Get memory statistics
    stats = memory_system.get_memory_stats()
    print(f"\nMemory system statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Queued memories: {stats['queued_memories']}")
    print(f"  Active threads: {stats['active_threads']}")
    
    # Stop processing
    memory_system.stop_processing()
    print("✅ Multi-threaded memory system test completed")
    return memory_system


def test_full_system_integration():
    """Test the complete system integration."""
    print("\n🚀 Testing Full System Integration")
    print("=" * 50)
    
    # Initialize system
    config = AIConfig()
    system = AdvancedAIMemorySystem(config)
    
    # Start system
    system.start_system()
    print("System started successfully")
    
    # Test interactions
    test_interactions = [
        {
            "input": "What is AI consciousness?",
            "context": {"user_id": "test_user", "session_id": "test_session_1"}
        },
        {
            "input": "How does parameter-efficient fine-tuning work?",
            "context": {"user_id": "test_user", "session_id": "test_session_1"}
        },
        {
            "input": "Tell me about memory systems in AI",
            "context": {"user_id": "test_user", "session_id": "test_session_1"}
        },
        {
            "input": "What are the safety considerations for AI systems?",
            "context": {"user_id": "test_user", "session_id": "test_session_2"}
        }
    ]
    
    for i, interaction in enumerate(test_interactions, 1):
        print(f"\n--- Interaction {i} ---")
        print(f"User: {interaction['input']}")
        
        result = system.process_interaction(
            interaction['input'],
            interaction['context']
        )
        
        print(f"Response: {result['response']}")
        print(f"Knowledge used: {len(result['knowledge_used'])} items")
        
        consciousness_score = result['consciousness_metrics'].get('current_consciousness_score', 'N/A')
        print(f"Consciousness score: {consciousness_score}")
        
        # Wait between interactions
        time.sleep(1)
    
    # Get comprehensive system report
    print("\n--- System Report ---")
    report = system.get_system_report()
    
    # Print key metrics
    print(f"System Name: {report['system_name']}")
    print(f"Version: {report['version']}")
    print(f"Uptime: {report['system_health']['uptime_seconds']:.1f} seconds")
    print(f"Interactions: {report['system_health']['interaction_count']}")
    
    consciousness_report = report['consciousness_report']
    if 'current_consciousness_score' in consciousness_report:
        print(f"Current Consciousness Score: {consciousness_report['current_consciousness_score']:.3f}")
    
    print(f"Memory System Health: {report['memory_stats']['total_memories']} memories processed")
    print(f"Knowledge Base: {report['knowledge_stats']['total_knowledge_items']} items")
    
    # Stop system
    system.stop_system()
    print("✅ Full system integration test completed")
    return system


def test_safety_features():
    """Test safety features and monitoring."""
    print("\n🛡️ Testing Safety Features")
    print("=" * 50)
    
    config = AIConfig(
        enable_safety_checks=True,
        safety_threshold=0.7,
        monitoring_interval=30
    )
    
    system = AdvancedAIMemorySystem(config)
    system.start_system()
    
    # Test system health monitoring
    health = system._get_system_health()
    
    print("System Health Metrics:")
    print(f"  Uptime: {health['uptime_seconds']:.1f} seconds")
    print(f"  Interaction count: {health['interaction_count']}")
    print(f"  Memory system active: {health['memory_system_health']['processing_active']}")
    print(f"  Consciousness monitoring active: {health['consciousness_monitoring_active']}")
    
    # Test safety threshold
    if health.get('overall_health', 1.0) < config.safety_threshold:
        print("⚠️ WARNING: System health below safety threshold")
    else:
        print("✅ System health above safety threshold")
    
    # Test consciousness monitoring
    consciousness_report = system.consciousness_monitor.get_consciousness_report()
    if 'current_consciousness_score' in consciousness_report:
        score = consciousness_report['current_consciousness_score']
        if score < 0.3:
            print("⚠️ WARNING: Low consciousness score detected")
        elif score > 0.8:
            print("✅ High consciousness score - system performing well")
        else:
            print(f"ℹ️ Normal consciousness score: {score:.3f}")
    
    system.stop_system()
    print("✅ Safety features test completed")


def run_performance_benchmark():
    """Run a performance benchmark."""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    config = AIConfig()
    system = AdvancedAIMemorySystem(config)
    system.start_system()
    
    # Benchmark parameters
    num_interactions = 50
    start_time = time.time()
    
    print(f"Running benchmark with {num_interactions} interactions...")
    
    for i in range(num_interactions):
        system.process_interaction(f"Benchmark question {i}")
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1}/{num_interactions} interactions ({rate:.2f} interactions/sec)")
    
    total_time = time.time() - start_time
    avg_time = total_time / num_interactions
    
    print(f"\nBenchmark Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per interaction: {avg_time:.3f} seconds")
    print(f"  Interactions per second: {num_interactions / total_time:.2f}")
    
    system.stop_system()
    print("✅ Performance benchmark completed")


def main():
    """Run all tests."""
    print("🧪 Advanced AI Memory and Consciousness Monitoring System - Test Suite")
    print("=" * 80)
    
    try:
        # Run individual component tests
        consciousness_report = test_consciousness_monitoring()
        knowledge_system = test_external_knowledge_system()
        pft_system = test_parameter_efficient_fine_tuning()
        memory_system = test_multi_threaded_memory_system()
        
        # Run integration tests
        full_system = test_full_system_integration()
        test_safety_features()
        
        # Run performance benchmark
        run_performance_benchmark()
        
        print("\n🎉 All tests completed successfully!")
        
        # Save test results
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_completed": [
                "consciousness_monitoring",
                "external_knowledge_system", 
                "parameter_efficient_fine_tuning",
                "multi_threaded_memory_system",
                "full_system_integration",
                "safety_features",
                "performance_benchmark"
            ],
            "consciousness_report": consciousness_report,
            "system_health": full_system._get_system_health() if full_system else None
        }
        
        with open("test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print("📊 Test results saved to test_results.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 