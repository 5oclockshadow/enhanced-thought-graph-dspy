"""
Enhanced web UI for advanced thought graph generation with comprehensive user controls.
Features professional-grade parameter configuration for complex idea generation.
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
import uvicorn
import json
from typing import Optional, List, Dict, Any
import logging

from models import (
    ExpandedInput, GraphParameters, GenerationMode, ClusteringMethod,
    ThoughtType, ConnectionType, UnifiedOutput
)
from dspy_program import AdvancedThoughtReactor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Thought Graph Generator",
    description="Professional-grade idea generation and mind mapping tool",
    version="2.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the thought reactor
reactor = AdvancedThoughtReactor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Enhanced home page with comprehensive parameter controls"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "generation_modes": [mode.value for mode in GenerationMode],
        "clustering_methods": [method.value for method in ClusteringMethod],
        "thought_types": [t.value for t in ThoughtType],
        "connection_types": [c.value for c in ConnectionType],
        "example_inputs": [
            "sustainable urban transportation",
            "AI-powered healthcare solutions", 
            "remote work productivity strategies",
            "circular economy business models",
            "climate change adaptation technologies"
        ]
    })

@app.post("/generate", response_class=HTMLResponse)
async def generate_thoughts(
    request: Request,
    input_text: str = Form(...),
    max_nodes: int = Form(20),
    max_depth: int = Form(4),
    generation_mode: str = Form("balanced"),
    creativity_level: float = Form(0.7),
    divergence_factor: float = Form(0.6),
    depth_vs_breadth: float = Form(0.5),
    clustering_method: str = Form("semantic"),
    max_clusters: int = Form(5),
    min_confidence: float = Form(0.3),
    domain_focus: Optional[str] = Form(None),
    perspective: Optional[str] = Form(None),
    enable_meta_thinking: bool = Form(False),
    enable_emotional_layer: bool = Form(True),
    enable_practical_layer: bool = Form(True),
    enable_risk_analysis: bool = Form(False),
    focus_areas: str = Form(""),
    constraints: str = Form(""),
    requirements: str = Form("")
):
    """Enhanced thought generation with comprehensive parameter support"""
    try:
        # Parse list inputs
        focus_areas_list = [area.strip() for area in focus_areas.split(",") if area.strip()]
        constraints_list = [constraint.strip() for constraint in constraints.split(",") if constraint.strip()]
        requirements_list = [req.strip() for req in requirements.split(",") if req.strip()]
        
        # Create enhanced parameters
        parameters = GraphParameters(
            max_nodes=max_nodes,
            max_depth=max_depth,
            generation_mode=GenerationMode(generation_mode),
            creativity_level=creativity_level,
            divergence_factor=divergence_factor,
            depth_vs_breadth=depth_vs_breadth,
            clustering_method=ClusteringMethod(clustering_method),
            max_clusters=max_clusters,
            min_confidence=min_confidence,
            domain_focus=domain_focus if domain_focus else None,
            perspective=perspective if perspective else None,
            enable_meta_thinking=enable_meta_thinking,
            enable_emotional_layer=enable_emotional_layer,
            enable_practical_layer=enable_practical_layer,
            enable_risk_analysis=enable_risk_analysis
        )
        
        # Create enhanced input
        expanded_input = ExpandedInput(
            original_text=input_text,
            parameters=parameters,
            focus_areas=focus_areas_list,
            constraints=constraints_list,
            requirements=requirements_list,
            context={
                "web_interface": True,
                "timestamp": str(datetime.now())
            }
        )
        
        # Generate thought graph
        logger.info(f"Generating enhanced thought graph for: {input_text}")
        result = reactor(expanded_input, use_fake_data=True)
        
        # Prepare visualization data
        viz_data = prepare_enhanced_visualization_data(result)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "input_text": input_text,
            "result": result,
            "viz_data": json.dumps(viz_data),
            "parameters": parameters,
            "success": result.success
        })
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {e}")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/api/generate/{input_text}")
async def api_generate(
    input_text: str,
    max_nodes: int = 20,
    max_depth: int = 4,
    generation_mode: str = "balanced",
    creativity_level: float = 0.7,
    clustering_method: str = "semantic",
    domain_focus: Optional[str] = None
):
    """Enhanced API endpoint for programmatic access"""
    try:
        parameters = GraphParameters(
            max_nodes=max_nodes,
            max_depth=max_depth,
            generation_mode=GenerationMode(generation_mode),
            creativity_level=creativity_level,
            clustering_method=ClusteringMethod(clustering_method),
            domain_focus=domain_focus
        )
        
        expanded_input = ExpandedInput(
            original_text=input_text,
            parameters=parameters
        )
        
        result = reactor(expanded_input, use_fake_data=True)
        
        return {
            "success": result.success,
            "input_text": input_text,
            "parameters": parameters.dict(),
            "graph": {
                "thoughts": [t.dict() for t in result.graph.thoughts] if result.graph else [],
                "connections": [c.dict() for c in result.graph.connections] if result.graph else [],
                "clusters": [cl.dict() for cl in result.graph.clusters] if result.graph else [],
                "metrics": {
                    "total_nodes": result.graph.total_nodes if result.graph else 0,
                    "total_edges": result.graph.total_edges if result.graph else 0,
                    "max_depth": result.graph.max_depth_achieved if result.graph else 0,
                    "quality_score": result.graph.overall_quality if result.graph else 0,
                    "creativity_score": result.graph.creativity_score if result.graph else 0,
                    "coherence_score": result.graph.coherence_score if result.graph else 0
                }
            },
            "insights": result.insights,
            "recommendations": result.recommendations,
            "summary": result.summary,
            "processing_info": {
                "iterations": result.iterations_used,
                "processing_time": result.processing_time,
                "final_state": result.context.current_state.value if result.context else "unknown"
            }
        }
        
    except Exception as e:
        logger.error(f"API generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/presets")
async def get_presets():
    """Get predefined parameter presets for different use cases"""
    presets = {
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
        },
        "exploratory_research": {
            "name": "Exploratory Research",
            "description": "Wide exploration of possibilities and connections",
            "parameters": {
                "max_nodes": 30,
                "max_depth": 6,
                "generation_mode": "exploratory",
                "creativity_level": 0.7,
                "divergence_factor": 0.9,
                "depth_vs_breadth": 0.4,
                "clustering_method": "semantic",
                "enable_meta_thinking": True,
                "max_clusters": 6
            }
        },
        "critical_analysis": {
            "name": "Critical Analysis",
            "description": "Critical thinking and evaluation focus",
            "parameters": {
                "max_nodes": 18,
                "max_depth": 5,
                "generation_mode": "critical",
                "creativity_level": 0.5,
                "divergence_factor": 0.5,
                "depth_vs_breadth": 0.7,
                "clustering_method": "hierarchical",
                "enable_risk_analysis": True,
                "min_confidence": 0.7
            }
        }
    }
    
    return presets

@app.post("/api/validate")
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

def prepare_enhanced_visualization_data(result: UnifiedOutput) -> Dict[str, Any]:
    """Prepare comprehensive data for enhanced visualization"""
    if not result.graph or not result.graph.thoughts:
        return {"nodes": [], "edges": [], "clusters": []}
    
    # Enhanced node data with comprehensive attributes
    nodes = []
    for thought in result.graph.thoughts:
        node = {
            "id": thought.id,
            "label": thought.content[:50] + "..." if len(thought.content) > 50 else thought.content,
            "title": f"<b>{thought.thought_type.value.title()}</b><br/>"
                    f"Content: {thought.content}<br/>"
                    f"Confidence: {thought.confidence:.1%}<br/>"
                    f"Depth: {thought.depth}<br/>"
                    f"Relevance: {thought.relevance_score:.1%}<br/>"
                    f"Novelty: {thought.novelty_score:.1%}<br/>"
                    f"Practicality: {thought.practicality_score:.1%}<br/>"
                    f"Impact: {thought.potential_impact}/5<br/>"
                    f"Complexity: {thought.complexity_level}/5",
            "group": thought.thought_type.value,
            "level": thought.depth,
            "size": 10 + (thought.confidence * 20),
            "color": get_enhanced_node_color(thought),
            "font": {"size": 12 + (thought.confidence * 8)},
            "borderWidth": 2 + (thought.relevance_score * 3),
            "chosen": True,
            "physics": True,
            "metadata": {
                "thought_type": thought.thought_type.value,
                "confidence": thought.confidence,
                "depth": thought.depth,
                "relevance": thought.relevance_score,
                "novelty": thought.novelty_score,
                "practicality": thought.practicality_score,
                "impact": thought.potential_impact,
                "complexity": thought.complexity_level,
                "tags": thought.tags,
                "keywords": thought.keywords,
                "domain": thought.domain,
                "cluster_id": thought.cluster_id
            }
        }
        nodes.append(node)
    
    # Enhanced edge data with relationship details
    edges = []
    for connection in result.graph.connections:
        edge = {
            "id": connection.id,
            "from": connection.source_id,
            "to": connection.target_id,
            "label": connection.connection_type.value.replace("_", " ").title(),
            "title": f"<b>{connection.connection_type.value.replace('_', ' ').title()}</b><br/>"
                    f"Strength: {connection.strength:.1%}<br/>"
                    f"Confidence: {connection.confidence:.1%}<br/>"
                    f"Weight: {connection.weight:.1f}<br/>"
                    f"Bidirectional: {'Yes' if connection.is_bidirectional else 'No'}",
            "width": 1 + (connection.strength * 4),
            "color": {
                "color": get_enhanced_edge_color(connection),
                "opacity": 0.6 + (connection.confidence * 0.4)
            },
            "arrows": {
                "to": {"enabled": True, "scaleFactor": 0.8}
            },
            "smooth": {"type": "continuous"},
            "chosen": True,
            "metadata": {
                "connection_type": connection.connection_type.value,
                "strength": connection.strength,
                "confidence": connection.confidence,
                "weight": connection.weight,
                "bidirectional": connection.is_bidirectional,
                "description": connection.description
            }
        }
        edges.append(edge)
    
    # Enhanced cluster data
    clusters = []
    for cluster in result.graph.clusters:
        cluster_data = {
            "id": cluster.id,
            "name": cluster.name,
            "description": cluster.description,
            "thought_ids": cluster.thought_ids,
            "coherence_score": cluster.coherence_score,
            "importance_score": cluster.importance_score,
            "cluster_type": cluster.cluster_type,
            "color": get_cluster_color(cluster.cluster_type),
            "size": len(cluster.thought_ids)
        }
        clusters.append(cluster_data)
    
    # Graph statistics and metrics
    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_clusters": len(clusters),
        "max_depth": max([n["level"] for n in nodes]) if nodes else 0,
        "avg_confidence": sum([n["metadata"]["confidence"] for n in nodes]) / len(nodes) if nodes else 0,
        "connectivity": len(edges) / len(nodes) if nodes else 0,
        "thought_type_distribution": {},
        "connection_type_distribution": {},
        "depth_distribution": {}
    }
    
    # Calculate distributions
    for node in nodes:
        thought_type = node["metadata"]["thought_type"]
        depth = node["metadata"]["depth"]
        stats["thought_type_distribution"][thought_type] = stats["thought_type_distribution"].get(thought_type, 0) + 1
        stats["depth_distribution"][depth] = stats["depth_distribution"].get(depth, 0) + 1
    
    for edge in edges:
        conn_type = edge["metadata"]["connection_type"]
        stats["connection_type_distribution"][conn_type] = stats["connection_type_distribution"].get(conn_type, 0) + 1
    
    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
        "stats": stats,
        "layout_options": {
            "hierarchical": {
                "enabled": True,
                "levelSeparation": 150,
                "nodeSpacing": 100,
                "treeSpacing": 200,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
                "direction": "UD",
                "sortMethod": "directed"
            },
            "physics": {
                "enabled": True,
                "stabilization": {"iterations": 100},
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                }
            }
        }
    }

def get_enhanced_node_color(thought: Thought) -> Dict[str, str]:
    """Get enhanced color scheme for nodes based on thought type and attributes"""
    base_colors = {
        ThoughtType.CORE_CONCEPT: "#2E8B57",      # Sea Green
        ThoughtType.SUB_CONCEPT: "#3CB371",       # Medium Sea Green
        ThoughtType.QUESTION: "#FF6347",          # Tomato
        ThoughtType.SOLUTION: "#32CD32",          # Lime Green
        ThoughtType.CHALLENGE: "#FF4500",         # Orange Red
        ThoughtType.OPPORTUNITY: "#FFD700",       # Gold
        ThoughtType.INSIGHT: "#9370DB",           # Medium Purple
        ThoughtType.HYPOTHESIS: "#20B2AA",        # Light Sea Green
        ThoughtType.ASSOCIATION: "#4169E1",       # Royal Blue
        ThoughtType.ANALOGY: "#DA70D6",           # Orchid
        ThoughtType.CONTRADICTION: "#DC143C",     # Crimson
        ThoughtType.EMOTION: "#FF69B4",           # Hot Pink
        ThoughtType.MEMORY: "#DDA0DD",            # Plum
        ThoughtType.PATTERN: "#8A2BE2",           # Blue Violet
        ThoughtType.TREND: "#00CED1",             # Dark Turquoise
        ThoughtType.CAUSE: "#B22222",             # Fire Brick
        ThoughtType.EFFECT: "#228B22",            # Forest Green
        ThoughtType.ALTERNATIVE: "#FF8C00",       # Dark Orange
        ThoughtType.SYNTHESIS: "#9932CC",         # Dark Orchid
        ThoughtType.CRITIQUE: "#8B0000"           # Dark Red
    }
    
    base_color = base_colors.get(thought.thought_type, "#808080")
    
    # Adjust opacity based on confidence
    opacity = 0.6 + (thought.confidence * 0.4)
    
    return {
        "background": base_color,
        "border": darken_color(base_color, 0.2),
        "highlight": {
            "background": lighten_color(base_color, 0.2),
            "border": base_color
        }
    }

def get_enhanced_edge_color(connection: Connection) -> str:
    """Get enhanced color for edges based on connection type"""
    colors = {
        ConnectionType.LEADS_TO: "#4169E1",       # Royal Blue
        ConnectionType.CAUSED_BY: "#DC143C",      # Crimson
        ConnectionType.SIMILAR_TO: "#32CD32",     # Lime Green
        ConnectionType.OPPOSITE_TO: "#FF4500",    # Orange Red
        ConnectionType.PART_OF: "#9370DB",        # Medium Purple
        ConnectionType.CONTAINS: "#2E8B57",       # Sea Green
        ConnectionType.ENABLES: "#FFD700",        # Gold
        ConnectionType.PREVENTS: "#FF6347",       # Tomato
        ConnectionType.SUPPORTS: "#20B2AA",       # Light Sea Green
        ConnectionType.CONFLICTS_WITH: "#B22222", # Fire Brick
        ConnectionType.BUILDS_ON: "#228B22",      # Forest Green
        ConnectionType.QUESTIONS: "#FF69B4",      # Hot Pink
        ConnectionType.ANSWERS: "#00CED1",        # Dark Turquoise
        ConnectionType.TRIGGERS: "#FF8C00",       # Dark Orange
        ConnectionType.TRANSFORMS: "#9932CC"      # Dark Orchid
    }
    
    return colors.get(connection.connection_type, "#808080")

def get_cluster_color(cluster_type: str) -> str:
    """Get color for cluster visualization"""
    colors = {
        "semantic": "#E6F3FF",      # Light Blue
        "hierarchical": "#F0FFF0",  # Honeydew
        "thematic": "#FFF8DC",      # Cornsilk
        "functional": "#F5F5DC",    # Beige
        "temporal": "#E0E6FF",      # Lavender
        "causal": "#FFE4E1"         # Misty Rose
    }
    return colors.get(cluster_type, "#F5F5F5")

def darken_color(color: str, factor: float) -> str:
    """Darken a hex color by a factor"""
    # Simple darkening - in production, use proper color manipulation library
    return color  # Placeholder

def lighten_color(color: str, factor: float) -> str:
    """Lighten a hex color by a factor"""
    # Simple lightening - in production, use proper color manipulation library
    return color  # Placeholder

# Add datetime import
from datetime import datetime

if __name__ == "__main__":
    uvicorn.run(
        "web_ui:app",
        host="0.0.0.0",
        port=52336,
        reload=True,
        log_level="info"
    )