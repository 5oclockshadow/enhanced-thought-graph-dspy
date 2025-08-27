"""
Enhanced Pydantic models for advanced thought graph generation with user-configurable parameters.
Designed for complex idea generation, brainstorming, and mind mapping.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime

class SystemState(str, Enum):
    """Enhanced system states for complex processing"""
    INITIALIZING = "initializing"
    ANALYZING_INPUT = "analyzing_input"
    GENERATING_CORE = "generating_core"
    EXPANDING_BREADTH = "expanding_breadth"
    DEEPENING_DEPTH = "deepening_depth"
    CONNECTING_IDEAS = "connecting_ideas"
    CLUSTERING = "clustering"
    VALIDATING = "validating"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    ERROR = "error"

class ThoughtType(str, Enum):
    """Extended thought types for comprehensive idea generation"""
    CORE_CONCEPT = "core_concept"
    SUB_CONCEPT = "sub_concept"
    ASSOCIATION = "association"
    ANALOGY = "analogy"
    CONTRADICTION = "contradiction"
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    SOLUTION = "solution"
    CHALLENGE = "challenge"
    OPPORTUNITY = "opportunity"
    EMOTION = "emotion"
    MEMORY = "memory"
    INSIGHT = "insight"
    PATTERN = "pattern"
    TREND = "trend"
    CAUSE = "cause"
    EFFECT = "effect"
    ALTERNATIVE = "alternative"
    SYNTHESIS = "synthesis"
    CRITIQUE = "critique"

class ConnectionType(str, Enum):
    """Enhanced connection types for rich relationship mapping"""
    LEADS_TO = "leads_to"
    CAUSED_BY = "caused_by"
    SIMILAR_TO = "similar_to"
    OPPOSITE_TO = "opposite_to"
    PART_OF = "part_of"
    CONTAINS = "contains"
    ENABLES = "enables"
    PREVENTS = "prevents"
    DEPENDS_ON = "depends_on"
    SUPPORTS = "supports"
    CONFLICTS_WITH = "conflicts_with"
    BUILDS_ON = "builds_on"
    QUESTIONS = "questions"
    ANSWERS = "answers"
    EXEMPLIFIES = "exemplifies"
    GENERALIZES = "generalizes"
    TRIGGERS = "triggers"
    REINFORCES = "reinforces"
    TRANSFORMS = "transforms"
    SYNTHESIZES = "synthesizes"

class GenerationMode(str, Enum):
    """Different modes for idea generation"""
    CREATIVE = "creative"           # Maximum creativity and divergent thinking
    ANALYTICAL = "analytical"       # Logical, structured analysis
    BALANCED = "balanced"          # Mix of creative and analytical
    EXPLORATORY = "exploratory"    # Wide exploration of possibilities
    FOCUSED = "focused"            # Deep dive into specific areas
    CRITICAL = "critical"          # Critical thinking and evaluation
    INNOVATIVE = "innovative"      # Focus on novel solutions
    PRACTICAL = "practical"       # Emphasis on actionable ideas

class ClusteringMethod(str, Enum):
    """Methods for organizing thoughts into clusters"""
    SEMANTIC = "semantic"          # Group by meaning similarity
    TEMPORAL = "temporal"          # Group by time relationships
    CAUSAL = "causal"             # Group by cause-effect chains
    HIERARCHICAL = "hierarchical"  # Group by abstraction levels
    THEMATIC = "thematic"         # Group by themes/topics
    FUNCTIONAL = "functional"      # Group by function/purpose
    NONE = "none"                 # No clustering

class GraphParameters(BaseModel):
    """Comprehensive parameters for controlling graph generation"""
    
    # Core generation parameters
    max_nodes: int = Field(default=20, ge=5, le=200, description="Maximum number of thought nodes")
    max_depth: int = Field(default=4, ge=1, le=10, description="Maximum depth of thought hierarchy")
    min_connections_per_node: int = Field(default=1, ge=0, le=10, description="Minimum connections per node")
    max_connections_per_node: int = Field(default=5, ge=1, le=20, description="Maximum connections per node")
    
    # Generation behavior
    generation_mode: GenerationMode = Field(default=GenerationMode.BALANCED, description="Mode of idea generation")
    creativity_level: float = Field(default=0.7, ge=0.0, le=1.0, description="Creativity vs structure balance")
    divergence_factor: float = Field(default=0.6, ge=0.0, le=1.0, description="How much to explore diverse ideas")
    depth_vs_breadth: float = Field(default=0.5, ge=0.0, le=1.0, description="0=breadth focused, 1=depth focused")
    
    # Content filtering
    include_thought_types: List[ThoughtType] = Field(default_factory=lambda: list(ThoughtType), description="Types of thoughts to include")
    exclude_thought_types: List[ThoughtType] = Field(default_factory=list, description="Types of thoughts to exclude")
    include_connection_types: List[ConnectionType] = Field(default_factory=lambda: list(ConnectionType), description="Types of connections to include")
    exclude_connection_types: List[ConnectionType] = Field(default_factory=list, description="Types of connections to exclude")
    
    # Quality controls
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    require_validation: bool = Field(default=True, description="Whether to validate generated thoughts")
    remove_duplicates: bool = Field(default=True, description="Remove duplicate or very similar thoughts")
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Threshold for duplicate detection")
    
    # Organization
    clustering_method: ClusteringMethod = Field(default=ClusteringMethod.SEMANTIC, description="How to cluster related thoughts")
    max_clusters: int = Field(default=5, ge=1, le=20, description="Maximum number of clusters")
    prioritize_by_relevance: bool = Field(default=True, description="Prioritize thoughts by relevance to input")
    
    # Domain-specific
    domain_focus: Optional[str] = Field(default=None, description="Specific domain to focus on (e.g., 'technology', 'business')")
    perspective: Optional[str] = Field(default=None, description="Specific perspective to take (e.g., 'user', 'developer', 'manager')")
    time_horizon: Optional[str] = Field(default=None, description="Time horizon for ideas (e.g., 'short-term', 'long-term')")
    
    # Advanced features
    enable_meta_thinking: bool = Field(default=False, description="Include thoughts about the thinking process itself")
    enable_emotional_layer: bool = Field(default=True, description="Include emotional aspects of ideas")
    enable_practical_layer: bool = Field(default=True, description="Include practical implementation aspects")
    enable_risk_analysis: bool = Field(default=False, description="Include risk and challenge identification")
    
    # Iteration control
    max_iterations: int = Field(default=10, ge=1, le=50, description="Maximum processing iterations")
    convergence_threshold: float = Field(default=0.95, ge=0.5, le=1.0, description="When to consider generation complete")

class Thought(BaseModel):
    """Enhanced thought model with rich metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1, max_length=1000)
    thought_type: ThoughtType
    confidence: float = Field(..., ge=0.0, le=1.0)
    depth: int = Field(..., ge=0, le=20)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    practicality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Hierarchical relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    
    # Clustering and organization
    cluster_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    generation_iteration: int = Field(default=1)
    source_prompt: Optional[str] = None
    
    # Domain-specific attributes
    domain: Optional[str] = None
    complexity_level: int = Field(default=1, ge=1, le=5)
    implementation_difficulty: int = Field(default=1, ge=1, le=5)
    potential_impact: int = Field(default=1, ge=1, le=5)
    
    class Config:
        extra = "allow"

class Connection(BaseModel):
    """Enhanced connection model with rich relationship data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    connection_type: ConnectionType
    strength: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Relationship metadata
    description: Optional[str] = None
    evidence: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    generation_iteration: int = Field(default=1)
    
    # Directional properties
    is_bidirectional: bool = Field(default=False)
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    
    class Config:
        extra = "allow"

class ThoughtCluster(BaseModel):
    """Cluster of related thoughts"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    thought_ids: List[str] = Field(default_factory=list)
    cluster_type: str = Field(default="semantic")
    coherence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Cluster relationships
    parent_cluster_id: Optional[str] = None
    child_cluster_ids: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"

class ProcessingContext(BaseModel):
    """Enhanced context for autonomous processing with detailed state management"""
    current_state: SystemState = SystemState.INITIALIZING
    iteration_count: int = 0
    should_continue: bool = True
    reasoning: str = ""
    next_action: str = ""
    
    # Progress tracking
    nodes_generated: int = 0
    connections_generated: int = 0
    clusters_generated: int = 0
    current_depth: int = 0
    
    # Quality metrics
    average_confidence: float = 0.0
    coverage_score: float = 0.0  # How well we've covered the topic
    diversity_score: float = 0.0  # How diverse the ideas are
    coherence_score: float = 0.0  # How well ideas connect
    
    # Processing history
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    decision_log: List[str] = Field(default_factory=list)
    
    # Parameters being used
    active_parameters: Optional[GraphParameters] = None
    
    class Config:
        extra = "allow"

class ValidationResult(BaseModel):
    """Enhanced validation with detailed feedback"""
    is_valid: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    
    # Detailed metrics
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    practicality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Recommendations
    recommended_additions: List[str] = Field(default_factory=list)
    recommended_removals: List[str] = Field(default_factory=list)
    recommended_modifications: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"

class ThoughtGraph(BaseModel):
    """Enhanced thought graph with comprehensive organization"""
    thoughts: List[Thought] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    clusters: List[ThoughtCluster] = Field(default_factory=list)
    
    # Graph metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    generation_parameters: Optional[GraphParameters] = None
    
    # Graph metrics
    total_nodes: int = 0
    total_edges: int = 0
    max_depth_achieved: int = 0
    average_connectivity: float = 0.0
    graph_density: float = 0.0
    
    # Quality scores
    overall_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    creativity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    class Config:
        extra = "allow"

class ExpandedInput(BaseModel):
    """Enhanced input model with comprehensive configuration options"""
    original_text: str = Field(..., min_length=1, max_length=5000)
    
    # Generation parameters
    parameters: GraphParameters = Field(default_factory=GraphParameters)
    
    # Context and constraints
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    
    # Domain-specific inputs
    domain_knowledge: Optional[str] = None
    existing_ideas: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)
    
    # User preferences
    preferred_styles: List[str] = Field(default_factory=list)
    avoid_topics: List[str] = Field(default_factory=list)
    focus_areas: List[str] = Field(default_factory=list)
    
    # Session information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    previous_graphs: List[str] = Field(default_factory=list)  # IDs of previous graphs
    
    class Config:
        extra = "allow"  # Allow additional fields

class UnifiedOutput(BaseModel):
    """Enhanced unified output with comprehensive results"""
    # Core outputs
    graph: Optional[ThoughtGraph] = None
    validation: Optional[ValidationResult] = None
    context: Optional[ProcessingContext] = None
    
    # Individual components (for partial results)
    thoughts: Optional[List[Thought]] = Field(default_factory=list)
    connections: Optional[List[Connection]] = Field(default_factory=list)
    clusters: Optional[List[ThoughtCluster]] = Field(default_factory=list)
    
    # Processing information
    success: bool = True
    processing_time: float = 0.0
    iterations_used: int = 0
    
    # Analytics and insights
    insights: List[str] = Field(default_factory=list)
    patterns_discovered: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Export formats
    summary: Optional[str] = None
    key_takeaways: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"  # Allow additional fields

# Utility functions for fake data generation with enhanced realism
def generate_fake_thought(thought_type: ThoughtType = None, depth: int = 1, domain: str = None) -> Thought:
    """Generate realistic fake thought data"""
    import random
    
    if thought_type is None:
        thought_type = random.choice(list(ThoughtType))
    
    # Domain-specific content templates
    content_templates = {
        "technology": [
            "AI-powered {concept} could revolutionize {field}",
            "Blockchain integration with {system} for enhanced {benefit}",
            "Machine learning algorithms to optimize {process}",
            "IoT sensors for real-time {monitoring} in {environment}",
            "Cloud-native architecture for scalable {solution}"
        ],
        "business": [
            "Market opportunity in {segment} through {approach}",
            "Cost reduction strategy via {method} optimization",
            "Customer experience enhancement using {technology}",
            "Revenue diversification through {channel} expansion",
            "Operational efficiency gains from {process} automation"
        ],
        "creative": [
            "Artistic expression combining {medium1} with {medium2}",
            "Narrative structure exploring {theme} through {perspective}",
            "Visual metaphor representing {concept} as {symbol}",
            "Interactive experience engaging {sense} and {emotion}",
            "Cultural fusion of {tradition1} and {tradition2}"
        ]
    }
    
    # Generate content based on domain and type
    if domain and domain in content_templates:
        template = random.choice(content_templates[domain])
        # Simple template filling (in real implementation, use more sophisticated NLP)
        content = template.format(
            concept=random.choice(["innovation", "solution", "system", "platform"]),
            field=random.choice(["healthcare", "education", "finance", "manufacturing"]),
            system=random.choice(["supply chain", "payment", "identity", "data"]),
            benefit=random.choice(["security", "transparency", "efficiency", "trust"]),
            process=random.choice(["workflow", "decision-making", "resource allocation"]),
            monitoring=random.choice(["performance", "quality", "compliance"]),
            environment=random.choice(["factory", "office", "home", "city"])
        )
    else:
        # Generic content based on thought type
        type_content = {
            ThoughtType.CORE_CONCEPT: f"Central idea about {random.choice(['innovation', 'transformation', 'optimization', 'integration'])}",
            ThoughtType.QUESTION: f"What if we could {random.choice(['eliminate', 'enhance', 'automate', 'personalize'])} {random.choice(['the process', 'user experience', 'efficiency', 'outcomes'])}?",
            ThoughtType.SOLUTION: f"Implement {random.choice(['advanced', 'integrated', 'scalable', 'adaptive'])} {random.choice(['system', 'approach', 'framework', 'methodology'])}",
            ThoughtType.CHALLENGE: f"Potential obstacle: {random.choice(['resource constraints', 'technical complexity', 'user adoption', 'regulatory compliance'])}",
            ThoughtType.OPPORTUNITY: f"Market opportunity in {random.choice(['emerging markets', 'underserved segments', 'new technologies', 'changing behaviors'])}"
        }
        content = type_content.get(thought_type, f"Thought about {thought_type.value}")
    
    return Thought(
        content=content,
        thought_type=thought_type,
        confidence=random.uniform(0.4, 0.95),
        depth=depth,
        relevance_score=random.uniform(0.3, 0.9),
        novelty_score=random.uniform(0.2, 0.8),
        practicality_score=random.uniform(0.3, 0.9),
        domain=domain,
        complexity_level=random.randint(1, 5),
        implementation_difficulty=random.randint(1, 5),
        potential_impact=random.randint(1, 5),
        tags=random.sample(["innovative", "practical", "scalable", "disruptive", "sustainable", "efficient"], k=random.randint(1, 3)),
        keywords=random.sample(["technology", "process", "system", "solution", "strategy", "optimization"], k=random.randint(2, 4))
    )

def generate_fake_connection(source_id: str, target_id: str, connection_type: ConnectionType = None) -> Connection:
    """Generate realistic fake connection data"""
    import random
    
    if connection_type is None:
        connection_type = random.choice(list(ConnectionType))
    
    return Connection(
        source_id=source_id,
        target_id=target_id,
        connection_type=connection_type,
        strength=random.uniform(0.3, 0.9),
        confidence=random.uniform(0.4, 0.95),
        description=f"Connection via {connection_type.value.replace('_', ' ')}",
        is_bidirectional=random.choice([True, False]),
        weight=random.uniform(0.5, 2.0)
    )

def generate_fake_graph(parameters: GraphParameters, input_text: str = "sample input") -> ThoughtGraph:
    """Generate a realistic fake thought graph based on parameters"""
    import random
    
    # Generate thoughts
    thoughts = []
    num_thoughts = min(parameters.max_nodes, random.randint(5, parameters.max_nodes))
    
    for i in range(num_thoughts):
        depth = min(parameters.max_depth, random.randint(1, parameters.max_depth))
        thought_type = random.choice(parameters.include_thought_types)
        thought = generate_fake_thought(thought_type, depth, parameters.domain_focus)
        thoughts.append(thought)
    
    # Generate connections
    connections = []
    for thought in thoughts:
        num_connections = random.randint(
            parameters.min_connections_per_node,
            min(parameters.max_connections_per_node, len(thoughts) - 1)
        )
        
        # Select random targets for connections
        possible_targets = [t for t in thoughts if t.id != thought.id]
        targets = random.sample(possible_targets, min(num_connections, len(possible_targets)))
        
        for target in targets:
            connection_type = random.choice(parameters.include_connection_types)
            connection = generate_fake_connection(thought.id, target.id, connection_type)
            connections.append(connection)
    
    # Generate clusters if clustering is enabled
    clusters = []
    if parameters.clustering_method != ClusteringMethod.NONE:
        num_clusters = min(parameters.max_clusters, max(1, len(thoughts) // 4))
        for i in range(num_clusters):
            cluster_thoughts = random.sample(thoughts, random.randint(2, min(6, len(thoughts))))
            cluster = ThoughtCluster(
                name=f"Cluster {i+1}: {random.choice(['Core Ideas', 'Supporting Concepts', 'Implementation', 'Challenges', 'Opportunities'])}",
                description=f"Related thoughts grouped by {parameters.clustering_method.value}",
                thought_ids=[t.id for t in cluster_thoughts],
                cluster_type=parameters.clustering_method.value,
                coherence_score=random.uniform(0.6, 0.9),
                importance_score=random.uniform(0.4, 0.8)
            )
            clusters.append(cluster)
    
    # Calculate graph metrics
    total_nodes = len(thoughts)
    total_edges = len(connections)
    max_depth_achieved = max([t.depth for t in thoughts]) if thoughts else 0
    average_connectivity = total_edges / total_nodes if total_nodes > 0 else 0
    graph_density = total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
    
    return ThoughtGraph(
        thoughts=thoughts,
        connections=connections,
        clusters=clusters,
        generation_parameters=parameters,
        total_nodes=total_nodes,
        total_edges=total_edges,
        max_depth_achieved=max_depth_achieved,
        average_connectivity=average_connectivity,
        graph_density=graph_density,
        overall_quality=random.uniform(0.7, 0.95),
        creativity_score=parameters.creativity_level,
        coherence_score=random.uniform(0.6, 0.9),
        completeness_score=random.uniform(0.7, 0.9)
    )