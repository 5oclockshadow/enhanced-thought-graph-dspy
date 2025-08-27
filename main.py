#!/usr/bin/env python3
"""
Enhanced main runner for the advanced thought graph generator.
Supports multiple modes: demo, web, and MCP server.
"""

import sys
import asyncio
import uvicorn
from models import ExpandedInput, GraphParameters, GenerationMode, ClusteringMethod
from dspy_program import AdvancedThoughtReactor

def run_demo():
    """Run enhanced demonstration with multiple scenarios"""
    print("🧠 Enhanced Thought Graph Generator - Demo Mode")
    print("=" * 60)
    
    # Create different scenarios to showcase capabilities
    scenarios = {
        "Creative Innovation": {
            "input": "sustainable urban transportation for smart cities",
            "params": GraphParameters(
                max_nodes=25,
                max_depth=5,
                generation_mode=GenerationMode.CREATIVE,
                creativity_level=0.9,
                divergence_factor=0.8,
                depth_vs_breadth=0.3,
                clustering_method=ClusteringMethod.THEMATIC,
                enable_emotional_layer=True,
                enable_meta_thinking=True,
                domain_focus="urban_planning"
            )
        },
        "Business Analysis": {
            "input": "remote work productivity challenges and solutions",
            "params": GraphParameters(
                max_nodes=20,
                max_depth=6,
                generation_mode=GenerationMode.ANALYTICAL,
                creativity_level=0.5,
                divergence_factor=0.4,
                depth_vs_breadth=0.7,
                clustering_method=ClusteringMethod.HIERARCHICAL,
                enable_practical_layer=True,
                enable_risk_analysis=True,
                domain_focus="business"
            )
        },
        "Technical Planning": {
            "input": "AI-powered healthcare diagnostic system implementation",
            "params": GraphParameters(
                max_nodes=18,
                max_depth=4,
                generation_mode=GenerationMode.PRACTICAL,
                creativity_level=0.6,
                divergence_factor=0.5,
                depth_vs_breadth=0.6,
                clustering_method=ClusteringMethod.FUNCTIONAL,
                enable_practical_layer=True,
                min_confidence=0.6,
                domain_focus="healthcare_technology"
            )
        }
    }
    
    reactor = AdvancedThoughtReactor()
    
    for scenario_name, config in scenarios.items():
        print(f"\n🎯 Scenario: {scenario_name}")
        print("-" * 40)
        
        # Create enhanced input
        expanded_input = ExpandedInput(
            original_text=config["input"],
            parameters=config["params"],
            context={"demo_mode": True, "scenario": scenario_name},
            focus_areas=["innovation", "feasibility", "impact"],
            requirements=["scalable", "sustainable", "user-centric"]
        )
        
        print(f"🔍 Processing: '{config['input']}'")
        print(f"📊 Mode: {config['params'].generation_mode.value}")
        print(f"🎨 Creativity: {config['params'].creativity_level}")
        print(f"🌐 Max Nodes: {config['params'].max_nodes}")
        print(f"📏 Max Depth: {config['params'].max_depth}")
        
        # Process with enhanced reactor
        result = reactor(expanded_input, use_fake_data=True)
        
        # Display comprehensive results
        print(f"\n✅ Success: {result.success}")
        print(f"🔄 Iterations: {result.iterations_used}")
        
        if result.graph:
            print(f"🧠 Thoughts Generated: {len(result.graph.thoughts)}")
            print(f"🔗 Connections: {len(result.graph.connections)}")
            print(f"📁 Clusters: {len(result.graph.clusters)}")
            print(f"📏 Max Depth Achieved: {result.graph.max_depth_achieved}")
            print(f"⭐ Overall Quality: {result.graph.overall_quality:.1%}")
            print(f"🎨 Creativity Score: {result.graph.creativity_score:.1%}")
            print(f"🔄 Coherence Score: {result.graph.coherence_score:.1%}")
            print(f"📈 Completeness: {result.graph.completeness_score:.1%}")
        
        if result.context:
            print(f"🎯 Final State: {result.context.current_state.value}")
            print(f"📊 Coverage: {result.context.coverage_score:.1%}")
            print(f"🌈 Diversity: {result.context.diversity_score:.1%}")
        
        if result.insights:
            print(f"\n💡 Key Insights:")
            for insight in result.insights[:3]:
                print(f"   • {insight}")
        
        if result.recommendations:
            print(f"\n📋 Recommendations:")
            for rec in result.recommendations[:2]:
                print(f"   • {rec}")
        
        if result.summary:
            print(f"\n📝 Summary: {result.summary}")
        
        print("\n" + "="*60)

def run_web():
    """Run enhanced web server"""
    print("🌐 Starting Enhanced Thought Graph Web Server...")
    print("🔗 Access at: http://localhost:52336")
    
    uvicorn.run(
        "web_ui:app",
        host="0.0.0.0",
        port=52336,
        reload=True,
        log_level="info"
    )

def run_mcp():
    """Run enhanced MCP server"""
    print("🔧 Starting Enhanced MCP Server...")
    
    try:
        from mcp_server import run_mcp_server
        asyncio.run(run_mcp_server())
    except ImportError:
        print("❌ MCP server dependencies not available")
        print("💡 Install with: pip install mcp")
        sys.exit(1)

def show_help():
    """Show enhanced help information"""
    print("""
🧠 Enhanced Thought Graph Generator

USAGE:
    python main.py [MODE]

MODES:
    demo    - Run comprehensive demonstration with multiple scenarios
    web     - Start interactive web interface (default)
    mcp     - Start MCP server for tool integration
    help    - Show this help message

FEATURES:
    ✨ Advanced parameter configuration
    🎯 Multiple generation modes (creative, analytical, practical, etc.)
    📊 Intelligent clustering and organization
    🔗 Rich relationship mapping
    🌐 Professional web interface
    🔧 MCP server integration
    📈 Comprehensive analytics and insights

EXAMPLES:
    python main.py demo     # Run demonstration scenarios
    python main.py web      # Start web server at localhost:52336
    python main.py mcp      # Start MCP server for integration

For more information, visit the web interface or check the README.md
    """)

def main():
    """Enhanced main function with comprehensive mode support"""
    if len(sys.argv) < 2:
        mode = "web"  # Default to web mode
    else:
        mode = sys.argv[1].lower()
    
    try:
        if mode == "demo":
            run_demo()
        elif mode == "web":
            run_web()
        elif mode == "mcp":
            run_mcp()
        elif mode in ["help", "-h", "--help"]:
            show_help()
        else:
            print(f"❌ Unknown mode: {mode}")
            print("💡 Use 'python main.py help' for available modes")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()