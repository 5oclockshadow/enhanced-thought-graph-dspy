# 🧠 Enhanced Thought Graph Generator

A professional-grade DSPy-powered system for complex idea generation, brainstorming, and mind mapping with advanced user-configurable parameters.

## ✨ Key Features

### 🎯 Advanced Parameter Control
- **Generation Modes**: Creative, Analytical, Practical, Exploratory, Critical, Innovative
- **Clustering Methods**: Semantic, Hierarchical, Thematic, Functional, Temporal, Causal
- **Quality Controls**: Confidence thresholds, duplicate removal, validation
- **Depth vs Breadth**: Configurable balance between exploration styles
- **Domain Focus**: Specialized generation for specific fields

### 🧠 Sophisticated Thought Types
- Core Concepts & Sub-concepts
- Questions & Hypotheses
- Solutions & Alternatives
- Challenges & Opportunities
- Insights & Patterns
- Analogies & Contradictions
- Causes & Effects
- Synthesis & Critique

### 🔗 Rich Relationship Mapping
- 20+ connection types (leads_to, caused_by, similar_to, etc.)
- Bidirectional relationships
- Weighted connections
- Confidence scoring
- Intelligent connection suggestions

### 📊 Professional Organization
- Intelligent clustering algorithms
- Hierarchical thought structures
- Thematic grouping
- Quality-based filtering
- Comprehensive analytics

### 🌐 Interactive Web Interface
- Professional parameter configuration
- Real-time visualization with vis.js
- Preset configurations for common use cases
- Export capabilities
- Mobile-responsive design

### 🔧 Integration Ready
- MCP server for tool integration
- REST API endpoints
- JSON import/export
- Extensible architecture

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd enhanced-thought-graph
pip install -r requirements.txt
```

### Usage Modes

#### 1. Interactive Web Interface (Recommended)
```bash
python main.py web
# Visit http://localhost:52336
```

#### 2. Demonstration Mode
```bash
python main.py demo
```

#### 3. MCP Server Mode
```bash
python main.py mcp
```

## 🎯 Use Cases

### Creative Brainstorming
Perfect for innovation sessions, product development, and creative problem-solving.

**Example Configuration:**
- Mode: Creative
- Creativity Level: 0.9
- Divergence Factor: 0.8
- Clustering: Thematic
- Enable: Emotional layer, Meta-thinking

### Business Analysis
Structured analysis for strategic planning, market research, and business development.

**Example Configuration:**
- Mode: Analytical
- Creativity Level: 0.4
- Depth vs Breadth: 0.8 (more depth)
- Clustering: Hierarchical
- Enable: Risk analysis, Practical layer

### Technical Planning
Implementation-focused planning for technical projects and system design.

**Example Configuration:**
- Mode: Practical
- Min Confidence: 0.6
- Clustering: Functional
- Enable: Practical layer
- Focus: Implementation, feasibility

### Research Exploration
Wide exploration of topics for academic research and knowledge discovery.

**Example Configuration:**
- Mode: Exploratory
- Max Nodes: 30
- Divergence Factor: 0.9
- Clustering: Semantic
- Enable: Meta-thinking

## 📊 Parameter Guide

### Core Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Max Nodes | 5-200 | Maximum number of thoughts to generate |
| Max Depth | 1-10 | Maximum depth of thought hierarchy |
| Creativity Level | 0.0-1.0 | Balance between creative and structured thinking |
| Divergence Factor | 0.0-1.0 | How diverse and varied the ideas should be |
| Depth vs Breadth | 0.0-1.0 | 0=more breadth, 1=more depth |
| Min Confidence | 0.0-1.0 | Filter out low-quality ideas |

### Generation Modes

- **Creative**: Maximum creativity and divergent thinking
- **Analytical**: Logical, structured analysis
- **Balanced**: Mix of creative and analytical
- **Exploratory**: Wide exploration of possibilities
- **Focused**: Deep dive into specific areas
- **Critical**: Critical thinking and evaluation
- **Innovative**: Focus on novel solutions
- **Practical**: Emphasis on actionable ideas

### Clustering Methods

- **Semantic**: Group by meaning similarity
- **Hierarchical**: Group by abstraction levels
- **Thematic**: Group by themes/topics
- **Functional**: Group by function/purpose
- **Temporal**: Group by time relationships
- **Causal**: Group by cause-effect chains

## 🔧 API Reference

### REST Endpoints

#### Generate Thought Graph
```http
GET /api/generate/{input_text}?max_nodes=20&generation_mode=creative
```

#### Get Presets
```http
GET /api/presets
```

#### Validate Parameters
```http
POST /api/validate
Content-Type: application/json

{
  "max_nodes": 25,
  "generation_mode": "creative",
  "creativity_level": 0.8
}
```

### Python API

```python
from models import ExpandedInput, GraphParameters, GenerationMode
from dspy_program import AdvancedThoughtReactor

# Configure parameters
params = GraphParameters(
    max_nodes=25,
    max_depth=5,
    generation_mode=GenerationMode.CREATIVE,
    creativity_level=0.8,
    clustering_method=ClusteringMethod.SEMANTIC
)

# Create input
expanded_input = ExpandedInput(
    original_text="sustainable energy solutions",
    parameters=params,
    focus_areas=["innovation", "implementation"],
    constraints=["budget limitations", "regulatory compliance"]
)

# Generate thoughts
reactor = AdvancedThoughtReactor()
result = reactor(expanded_input)

# Access results
print(f"Generated {len(result.graph.thoughts)} thoughts")
print(f"Quality score: {result.graph.overall_quality:.1%}")
```

## 🎨 Visualization Features

### Interactive Graph
- **Node Styling**: Size and color based on confidence and type
- **Edge Styling**: Width and color based on relationship strength
- **Clustering**: Visual grouping of related thoughts
- **Hierarchical Layout**: Automatic depth-based positioning
- **Physics Simulation**: Natural node positioning

### Analytics Dashboard
- Thought type distribution
- Connection type analysis
- Depth distribution
- Quality metrics
- Coverage analysis

## 🔬 Advanced Features

### Meta-Thinking
Enable thoughts about the thinking process itself for deeper self-reflection.

### Emotional Layer
Include emotional aspects and psychological considerations in idea generation.

### Practical Layer
Focus on implementation feasibility and practical constraints.

### Risk Analysis
Identify potential challenges, obstacles, and risk factors.

### Domain Specialization
Tailor generation for specific domains like technology, business, healthcare, etc.

## 📈 Quality Metrics

### Thought Quality
- **Confidence**: How certain the system is about the thought
- **Relevance**: How relevant to the original input
- **Novelty**: How original and creative the thought is
- **Practicality**: How actionable and implementable

### Graph Quality
- **Completeness**: Coverage of the topic
- **Coherence**: How well thoughts connect
- **Diversity**: Variety of thought types and perspectives
- **Depth**: Levels of analysis achieved

## 🛠️ Customization

### Custom Thought Types
Extend the `ThoughtType` enum to add domain-specific thought categories.

### Custom Connection Types
Add new relationship types to the `ConnectionType` enum.

### Custom Clustering
Implement new clustering algorithms by extending the clustering methods.

### Custom Validation
Add domain-specific validation rules and quality checks.

## 🔄 Integration

### MCP Server
The system includes a full MCP server implementation with tools for:
- Thought graph generation
- Parameter validation
- Graph analysis
- Export functionality

### External APIs
Easy integration with external systems through REST API and Python SDK.

### Data Export
- JSON format for programmatic access
- Visualization data for custom UIs
- Summary reports for documentation

## 🎯 Best Practices

### For Creative Sessions
- Use high creativity level (0.8-0.9)
- Enable emotional and meta-thinking layers
- Use thematic clustering
- Set high divergence factor

### For Business Analysis
- Use analytical mode
- Enable risk analysis
- Use hierarchical clustering
- Set higher confidence threshold

### For Technical Planning
- Use practical mode
- Focus on implementation aspects
- Use functional clustering
- Include constraints and requirements

### For Research
- Use exploratory mode
- Set high node count
- Enable meta-thinking
- Use semantic clustering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with DSPy for advanced language model programming
- Powered by Pydantic for robust data validation
- Visualized with vis.js for interactive graphs
- Served with FastAPI for modern web APIs

---

**Transform your thinking with professional-grade idea generation tools.**