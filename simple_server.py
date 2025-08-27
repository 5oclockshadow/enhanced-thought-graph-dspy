#!/usr/bin/env python3
"""
Simple API server for the Enhanced Thought Graph Generator
Provides REST API endpoints without template dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import uvicorn
import json
from typing import Optional, List, Dict, Any
import logging

from models import (
    ExpandedInput, GraphParameters, GenerationMode, ClusteringMethod,
    ThoughtType, ConnectionType, UnifiedOutput
)
from dspy_program import AutonomousThoughtReactor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Thought Graph API",
    description="Professional-grade API for complex idea generation and mind mapping",
    version="2.0.0"
)

# Add CORS middleware to allow external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the thought reactor
reactor = AutonomousThoughtReactor()

class GenerationRequest(BaseModel):
    """Request model for thought generation"""
    input_text: str
    max_nodes: int = 20
    max_depth: int = 4
    generation_mode: str = "balanced"
    creativity_level: float = 0.7
    divergence_factor: float = 0.6
    depth_vs_breadth: float = 0.5
    clustering_method: str = "semantic"
    max_clusters: int = 5
    min_confidence: float = 0.3
    domain_focus: Optional[str] = None
    perspective: Optional[str] = None
    enable_meta_thinking: bool = False
    enable_emotional_layer: bool = True
    enable_practical_layer: bool = True
    enable_risk_analysis: bool = False
    focus_areas: List[str] = []
    constraints: List[str] = []
    requirements: List[str] = []

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Enhanced Thought Graph API",
        "version": "2.0.0",
        "description": "Professional-grade API for complex idea generation and mind mapping",
        "features": [
            "Advanced parameter configuration",
            "Multiple generation modes (creative, analytical, practical, etc.)",
            "3D spatial thought representation",
            "Intelligent clustering and organization",
            "Real-time processing with state management",
            "Comprehensive analytics and insights"
        ],
        "endpoints": {
            "POST /generate": "Generate thought graph with full parameters",
            "GET /generate/{input_text}": "Quick generation with query parameters",
            "GET /presets": "Get predefined parameter presets",
            "POST /validate": "Validate parameter configuration",
            "GET /health": "Health check endpoint"
        },
        "examples": {
            "quick_generate": "/generate/sustainable%20energy%20solutions?max_nodes=15&generation_mode=creative",
            "full_generate": "POST /generate with JSON body containing all parameters"
        }
    }

@app.post("/generate")
async def generate_thoughts(request: GenerationRequest):
    """Generate thought graph with comprehensive parameters"""
    try:
        # Create enhanced parameters
        parameters = GraphParameters(
            max_nodes=request.max_nodes,
            max_depth=request.max_depth,
            generation_mode=GenerationMode(request.generation_mode),
            creativity_level=request.creativity_level,
            divergence_factor=request.divergence_factor,
            depth_vs_breadth=request.depth_vs_breadth,
            clustering_method=ClusteringMethod(request.clustering_method),
            max_clusters=request.max_clusters,
            min_confidence=request.min_confidence,
            domain_focus=request.domain_focus,
            perspective=request.perspective,
            enable_meta_thinking=request.enable_meta_thinking,
            enable_emotional_layer=request.enable_emotional_layer,
            enable_practical_layer=request.enable_practical_layer,
            enable_risk_analysis=request.enable_risk_analysis
        )
        
        # Create enhanced input
        expanded_input = ExpandedInput(
            original_text=request.input_text,
            parameters=parameters,
            focus_areas=request.focus_areas,
            constraints=request.constraints,
            requirements=request.requirements,
            context={"api_request": True}
        )
        
        # Generate thought graph
        logger.info(f"Generating thought graph for: {request.input_text}")
        result = reactor(expanded_input, use_fake_data=True)
        
        # Prepare response
        response = {
            "success": result.success,
            "input_text": request.input_text,
            "parameters": parameters.dict(),
            "processing_info": {
                "iterations": result.iterations_used,
                "processing_time": result.processing_time,
                "final_state": result.context.current_state.value if result.context else "unknown"
            }
        }
        
        if result.graph:
            response["graph"] = {
                "thoughts": [
                    {
                        "id": t.id,
                        "content": t.content,
                        "type": t.thought_type.value,
                        "confidence": t.confidence,
                        "depth": t.depth,
                        "relevance": t.relevance_score,
                        "novelty": t.novelty_score,
                        "practicality": t.practicality_score,
                        "impact": t.potential_impact,
                        "complexity": t.complexity_level,
                        "tags": t.tags,
                        "keywords": t.keywords,
                        "domain": t.domain
                    } for t in result.graph.thoughts
                ],
                "connections": [
                    {
                        "id": c.id,
                        "source": c.source_id,
                        "target": c.target_id,
                        "type": c.connection_type.value,
                        "strength": c.strength,
                        "confidence": c.confidence,
                        "description": c.description
                    } for c in result.graph.connections
                ],
                "clusters": [
                    {
                        "id": cl.id,
                        "name": cl.name,
                        "description": cl.description,
                        "thought_ids": cl.thought_ids,
                        "coherence": cl.coherence_score,
                        "importance": cl.importance_score,
                        "type": cl.cluster_type
                    } for cl in result.graph.clusters
                ],
                "metrics": {
                    "total_nodes": result.graph.total_nodes,
                    "total_edges": result.graph.total_edges,
                    "max_depth": result.graph.max_depth_achieved,
                    "quality_score": result.graph.overall_quality,
                    "creativity_score": result.graph.creativity_score,
                    "coherence_score": result.graph.coherence_score,
                    "completeness_score": result.graph.completeness_score
                }
            }
        
        if result.insights:
            response["insights"] = result.insights
        
        if result.recommendations:
            response["recommendations"] = result.recommendations
        
        if result.summary:
            response["summary"] = result.summary
        
        if result.errors:
            response["errors"] = result.errors
        
        return response
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {e}")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/generate/{input_text}")
async def quick_generate(
    input_text: str,
    max_nodes: int = 20,
    max_depth: int = 4,
    generation_mode: str = "balanced",
    creativity_level: float = 0.7,
    clustering_method: str = "semantic",
    domain_focus: Optional[str] = None
):
    """Quick generation with query parameters"""
    request = GenerationRequest(
        input_text=input_text,
        max_nodes=max_nodes,
        max_depth=max_depth,
        generation_mode=generation_mode,
        creativity_level=creativity_level,
        clustering_method=clustering_method,
        domain_focus=domain_focus
    )
    return await generate_thoughts(request)

@app.get("/presets")
async def get_presets():
    """Get predefined parameter presets"""
    return {
        "creative_brainstorm": {
            "name": "Creative Brainstorming",
            "description": "Maximum creativity for innovative idea generation",
            "parameters": {
                "max_nodes": 25,
                "max_depth": 5,
                "generation_mode": "creative",
                "creativity_level": 0.9,
                "divergence_factor": 0.8,
                "depth_vs_breadth": 0.3,
                "clustering_method": "thematic",
                "enable_emotional_layer": True,
                "enable_meta_thinking": True
            }
        },
        "analytical_deep_dive": {
            "name": "Analytical Deep Dive",
            "description": "Structured analysis with logical depth",
            "parameters": {
                "max_nodes": 20,
                "max_depth": 7,
                "generation_mode": "analytical",
                "creativity_level": 0.4,
                "divergence_factor": 0.3,
                "depth_vs_breadth": 0.8,
                "clustering_method": "hierarchical",
                "enable_risk_analysis": True,
                "enable_practical_layer": True
            }
        },
        "practical_planning": {
            "name": "Practical Planning",
            "description": "Action-oriented planning and implementation focus",
            "parameters": {
                "max_nodes": 15,
                "max_depth": 4,
                "generation_mode": "practical",
                "creativity_level": 0.6,
                "divergence_factor": 0.4,
                "depth_vs_breadth": 0.6,
                "clustering_method": "functional",
                "enable_practical_layer": True,
                "min_confidence": 0.6
            }
        }
    }

@app.post("/validate")
async def validate_parameters(parameters: Dict[str, Any]):
    """Validate parameter configuration"""
    try:
        # Convert string enums to proper enum values
        if "generation_mode" in parameters:
            parameters["generation_mode"] = GenerationMode(parameters["generation_mode"])
        if "clustering_method" in parameters:
            parameters["clustering_method"] = ClusteringMethod(parameters["clustering_method"])
        
        # Validate parameters
        graph_params = GraphParameters(**parameters)
        
        return {
            "valid": True,
            "parameters": graph_params.dict(),
            "warnings": [],
            "suggestions": []
        }
        
    except ValidationError as e:
        return {
            "valid": False,
            "errors": [{"field": err["loc"][0], "message": err["msg"]} for err in e.errors()],
            "warnings": [],
            "suggestions": ["Check parameter ranges and types"]
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Enhanced Thought Graph API",
        "version": "2.0.0",
        "features_available": True,
        "reactor_initialized": reactor is not None
    }

if __name__ == "__main__":
    print("🌌 Enhanced Thought Graph API Server")
    print("=" * 50)
    print("🚀 Starting server...")
    print("🔗 API will be available at: http://localhost:52336")
    print("📚 API documentation at: http://localhost:52336/docs")
    print("🎯 Features: Advanced parameters, 3D visualization data, real-time processing")
    print("=" * 50)
    
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=52336,
        reload=False,
        log_level="info"
    )