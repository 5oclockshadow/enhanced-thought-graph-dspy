"""
Enhanced DSPy program for advanced thought graph generation with user-configurable parameters.
Optimized for complex idea generation, brainstorming, and mind mapping.
"""

import dspy
import json
import logging
from typing import Dict, Any, List
from models import (
    ExpandedInput, UnifiedOutput, ProcessingContext, ThoughtGraph, 
    SystemState, GraphParameters, generate_fake_graph, ValidationResult,
    Thought, Connection, ThoughtCluster, ThoughtType, ConnectionType,
    GenerationMode, ClusteringMethod
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedReactSignature(dspy.Signature):
    """
    Enhanced React signature for autonomous thought graph generation with comprehensive parameter support.
    Handles complex idea generation, brainstorming, and mind mapping scenarios.
    """
    input_data = dspy.InputField(desc="Enhanced input data as JSON string containing ExpandedInput model with comprehensive parameters")
    context = dspy.InputField(desc="Current processing context with detailed state and progress tracking")
    reasoning = dspy.OutputField(desc="Detailed step-by-step reasoning about current state, progress, and next actions")
    action = dspy.OutputField(desc="Specific next action to take based on current state and parameters")
    output_data = dspy.OutputField(desc="Enhanced output data as JSON string containing UnifiedOutput model with comprehensive results")
    updated_context = dspy.OutputField(desc="Updated processing context with progress, metrics, and state transitions")

class AutonomousThoughtReactor(dspy.Module):
    """
    Enhanced autonomous thought graph generator with advanced parameter support.
    Designed for complex idea generation, brainstorming, and mind mapping workflows.
    """
    
    def __init__(self):
        super().__init__()
        self.react = dspy.ChainOfThought(EnhancedReactSignature)
        
    def forward(self, expanded_input: ExpandedInput, use_fake_data: bool = True) -> UnifiedOutput:
        """
        Enhanced autonomous processing with comprehensive parameter support and advanced features.
        """
        # Initialize enhanced processing context
        context = ProcessingContext(
            current_state=SystemState.INITIALIZING,
            iteration_count=0,
            should_continue=True,
            reasoning="Starting enhanced autonomous thought graph generation",
            next_action="analyze_input_and_parameters",
            active_parameters=expanded_input.parameters
        )
        
        # Initialize comprehensive output
        unified_output = UnifiedOutput(
            success=True,
            processing_time=0.0,
            iterations_used=0,
            thoughts=[],
            connections=[],
            clusters=[],
            insights=[],
            patterns_discovered=[],
            recommendations=[]
        )
        
        # Main autonomous processing loop with enhanced error handling
        while context.should_continue and context.iteration_count < expanded_input.parameters.max_iterations:
            try:
                context.iteration_count += 1
                logger.info(f"Enhanced iteration {context.iteration_count}, State: {context.current_state}")
                
                # Process with enhanced LLM reasoning
                result = self._process_with_enhanced_llm(expanded_input, context, unified_output, use_fake_data)
                
                # Update context and output based on LLM decisions
                context = result.get('updated_context', context)
                unified_output = result.get('updated_output', unified_output)
                
                # Enhanced convergence checking
                if self._check_enhanced_convergence(context, expanded_input.parameters):
                    context.should_continue = False
                    context.current_state = SystemState.COMPLETED
                    context.reasoning = "Enhanced processing completed successfully with all parameters satisfied"
                
                logger.info(f"Completed enhanced iteration {context.iteration_count}, State: {context.current_state}, Should continue: {context.should_continue}")
                
            except Exception as e:
                logger.error(f"Enhanced processing error in iteration {context.iteration_count}: {str(e)}")
                context.current_state = SystemState.ERROR
                context.reasoning = f"Error encountered: {str(e)}"
                unified_output.errors.append(f"Iteration {context.iteration_count}: {str(e)}")
                
                # Enhanced error recovery
                if context.iteration_count < expanded_input.parameters.max_iterations - 1:
                    context.current_state = SystemState.ANALYZING_INPUT
                    context.reasoning = "Attempting recovery from error state"
                    logger.info("Attempting enhanced error recovery")
                else:
                    context.should_continue = False
                    unified_output.success = False
                    
            finally:
                # Always update output with current state and enhanced metrics
                unified_output.context = context
                unified_output.iterations_used = context.iteration_count
                
                # Update progress metrics
                context.nodes_generated = len(unified_output.thoughts) if unified_output.thoughts else 0
                context.connections_generated = len(unified_output.connections) if unified_output.connections else 0
                context.clusters_generated = len(unified_output.clusters) if unified_output.clusters else 0
                
                # Calculate enhanced quality metrics
                if unified_output.thoughts:
                    context.average_confidence = sum(t.confidence for t in unified_output.thoughts) / len(unified_output.thoughts)
                    context.diversity_score = self._calculate_diversity_score(unified_output.thoughts)
                    context.coherence_score = self._calculate_coherence_score(unified_output.thoughts, unified_output.connections)
                
                # Log enhanced state history
                context.state_history.append({
                    "iteration": context.iteration_count,
                    "state": context.current_state.value,
                    "nodes": context.nodes_generated,
                    "connections": context.connections_generated,
                    "quality": context.average_confidence
                })
        
        # Final enhanced processing and validation
        unified_output = self._finalize_enhanced_output(unified_output, expanded_input, context)
        
        return unified_output
    
    def _process_with_enhanced_llm(self, expanded_input: ExpandedInput, context: ProcessingContext, 
                                 current_output: UnifiedOutput, use_fake_data: bool) -> Dict[str, Any]:
        """Enhanced LLM processing with comprehensive parameter support"""
        
        if use_fake_data:
            return self._generate_enhanced_fake_response(expanded_input, context, current_output)
        
        try:
            # Prepare enhanced input for LLM
            input_json = json.dumps({
                "original_text": expanded_input.original_text,
                "parameters": expanded_input.parameters.dict(),
                "context": expanded_input.context,
                "constraints": expanded_input.constraints,
                "requirements": expanded_input.requirements,
                "domain_knowledge": expanded_input.domain_knowledge,
                "existing_ideas": expanded_input.existing_ideas,
                "current_progress": {
                    "nodes_generated": context.nodes_generated,
                    "connections_generated": context.connections_generated,
                    "current_depth": context.current_depth,
                    "quality_metrics": {
                        "average_confidence": context.average_confidence,
                        "diversity_score": context.diversity_score,
                        "coherence_score": context.coherence_score
                    }
                }
            })
            
            context_json = json.dumps({
                "current_state": context.current_state.value,
                "iteration_count": context.iteration_count,
                "reasoning": context.reasoning,
                "progress": {
                    "nodes_generated": context.nodes_generated,
                    "connections_generated": context.connections_generated,
                    "clusters_generated": context.clusters_generated,
                    "current_depth": context.current_depth
                },
                "quality_metrics": {
                    "average_confidence": context.average_confidence,
                    "coverage_score": context.coverage_score,
                    "diversity_score": context.diversity_score,
                    "coherence_score": context.coherence_score
                },
                "parameters": expanded_input.parameters.dict()
            })
            
            # Call enhanced LLM
            response = self.react(input_data=input_json, context=context_json)
            
            # Parse enhanced response
            try:
                output_data = json.loads(response.output_data)
                updated_context_data = json.loads(response.updated_context)
                
                # Update context with enhanced information
                context.reasoning = response.reasoning
                context.next_action = response.action
                context.current_state = SystemState(updated_context_data.get('current_state', context.current_state.value))
                context.should_continue = updated_context_data.get('should_continue', True)
                
                # Update quality metrics
                if 'quality_metrics' in updated_context_data:
                    metrics = updated_context_data['quality_metrics']
                    context.average_confidence = metrics.get('average_confidence', context.average_confidence)
                    context.coverage_score = metrics.get('coverage_score', context.coverage_score)
                    context.diversity_score = metrics.get('diversity_score', context.diversity_score)
                    context.coherence_score = metrics.get('coherence_score', context.coherence_score)
                
                # Update output with enhanced data
                if 'thoughts' in output_data:
                    current_output.thoughts = [Thought(**t) for t in output_data['thoughts']]
                if 'connections' in output_data:
                    current_output.connections = [Connection(**c) for c in output_data['connections']]
                if 'clusters' in output_data:
                    current_output.clusters = [ThoughtCluster(**cl) for cl in output_data['clusters']]
                if 'insights' in output_data:
                    current_output.insights = output_data['insights']
                if 'recommendations' in output_data:
                    current_output.recommendations = output_data['recommendations']
                
                return {
                    'updated_context': context,
                    'updated_output': current_output
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Enhanced JSON parsing error: {e}")
                return self._generate_enhanced_fake_response(expanded_input, context, current_output)
                
        except Exception as e:
            logger.error(f"Enhanced LLM processing error: {e}")
            return self._generate_enhanced_fake_response(expanded_input, context, current_output)
    
    def _generate_enhanced_fake_response(self, expanded_input: ExpandedInput, context: ProcessingContext, 
                                       current_output: UnifiedOutput) -> Dict[str, Any]:
        """Generate enhanced fake response with comprehensive parameter support"""
        import random
        
        # Enhanced state transitions based on parameters and progress
        state_transitions = {
            SystemState.INITIALIZING: SystemState.ANALYZING_INPUT,
            SystemState.ANALYZING_INPUT: SystemState.GENERATING_CORE,
            SystemState.GENERATING_CORE: SystemState.EXPANDING_BREADTH if expanded_input.parameters.depth_vs_breadth < 0.5 else SystemState.DEEPENING_DEPTH,
            SystemState.EXPANDING_BREADTH: SystemState.CONNECTING_IDEAS,
            SystemState.DEEPENING_DEPTH: SystemState.CONNECTING_IDEAS,
            SystemState.CONNECTING_IDEAS: SystemState.CLUSTERING if expanded_input.parameters.clustering_method.value != "none" else SystemState.VALIDATING,
            SystemState.CLUSTERING: SystemState.VALIDATING,
            SystemState.VALIDATING: SystemState.OPTIMIZING if context.average_confidence < 0.8 else SystemState.COMPLETED,
            SystemState.OPTIMIZING: SystemState.COMPLETED,
            SystemState.COMPLETED: SystemState.COMPLETED
        }
        
        # Update state
        next_state = state_transitions.get(context.current_state, SystemState.COMPLETED)
        
        # Enhanced reasoning based on current state and parameters
        reasoning_templates = {
            SystemState.ANALYZING_INPUT: f"Analyzing input '{expanded_input.original_text}' with {expanded_input.parameters.generation_mode.value} mode, targeting {expanded_input.parameters.max_nodes} nodes at {expanded_input.parameters.max_depth} depth levels",
            SystemState.GENERATING_CORE: f"Generating core concepts with creativity level {expanded_input.parameters.creativity_level}, focusing on {expanded_input.parameters.domain_focus or 'general domain'}",
            SystemState.EXPANDING_BREADTH: f"Expanding breadth with divergence factor {expanded_input.parameters.divergence_factor}, including {len(expanded_input.parameters.include_thought_types)} thought types",
            SystemState.DEEPENING_DEPTH: f"Deepening analysis to level {context.current_depth + 1}, maintaining minimum confidence {expanded_input.parameters.min_confidence}",
            SystemState.CONNECTING_IDEAS: f"Creating connections with {len(expanded_input.parameters.include_connection_types)} connection types, targeting {expanded_input.parameters.min_connections_per_node}-{expanded_input.parameters.max_connections_per_node} per node",
            SystemState.CLUSTERING: f"Organizing thoughts using {expanded_input.parameters.clustering_method.value} clustering into max {expanded_input.parameters.max_clusters} clusters",
            SystemState.VALIDATING: f"Validating graph quality with current metrics: confidence={context.average_confidence:.2f}, diversity={context.diversity_score:.2f}",
            SystemState.OPTIMIZING: "Optimizing graph structure and improving low-quality connections",
            SystemState.COMPLETED: f"Enhanced processing completed with {context.nodes_generated} nodes, {context.connections_generated} connections, quality score {context.average_confidence:.2f}"
        }
        
        reasoning = reasoning_templates.get(context.current_state, f"Processing in {context.current_state.value} state")
        
        # Generate enhanced fake data based on current state and parameters
        if context.current_state in [SystemState.GENERATING_CORE, SystemState.EXPANDING_BREADTH, SystemState.DEEPENING_DEPTH]:
            # Generate or update thought graph
            if not current_output.graph:
                current_output.graph = generate_fake_graph(expanded_input.parameters, expanded_input.original_text)
                current_output.thoughts = current_output.graph.thoughts
                current_output.connections = current_output.graph.connections
                current_output.clusters = current_output.graph.clusters
            
            # Add insights based on generation mode
            if expanded_input.parameters.generation_mode.value == "creative":
                current_output.insights.extend([
                    "High creativity mode enabled - exploring unconventional connections",
                    "Divergent thinking patterns identified in idea generation",
                    "Novel concept combinations discovered"
                ])
            elif expanded_input.parameters.generation_mode.value == "analytical":
                current_output.insights.extend([
                    "Analytical mode - focusing on logical relationships",
                    "Structured hierarchy of concepts established",
                    "Evidence-based connections prioritized"
                ])
        
        # Update context with enhanced metrics
        context.current_state = next_state
        context.reasoning = reasoning
        context.should_continue = next_state != SystemState.COMPLETED
        
        # Calculate enhanced progress metrics
        if current_output.thoughts:
            context.nodes_generated = len(current_output.thoughts)
            context.average_confidence = sum(t.confidence for t in current_output.thoughts) / len(current_output.thoughts)
            context.current_depth = max(t.depth for t in current_output.thoughts)
            context.diversity_score = self._calculate_diversity_score(current_output.thoughts)
            
        if current_output.connections:
            context.connections_generated = len(current_output.connections)
            context.coherence_score = self._calculate_coherence_score(current_output.thoughts, current_output.connections)
            
        if current_output.clusters:
            context.clusters_generated = len(current_output.clusters)
        
        # Calculate coverage score based on parameters
        target_coverage = min(1.0, context.nodes_generated / expanded_input.parameters.max_nodes)
        context.coverage_score = target_coverage
        
        # Add decision to log
        context.decision_log.append(f"Iteration {context.iteration_count}: {context.current_state.value} - {reasoning}")
        
        return {
            'updated_context': context,
            'updated_output': current_output
        }
    
    def _check_enhanced_convergence(self, context: ProcessingContext, parameters: GraphParameters) -> bool:
        """Enhanced convergence checking with comprehensive criteria"""
        
        # Check if we've reached completion state
        if context.current_state == SystemState.COMPLETED:
            return True
        
        # Check parameter-based completion criteria
        criteria_met = 0
        total_criteria = 0
        
        # Node count criterion
        total_criteria += 1
        if context.nodes_generated >= parameters.max_nodes * 0.8:  # 80% of target
            criteria_met += 1
        
        # Depth criterion
        total_criteria += 1
        if context.current_depth >= parameters.max_depth * 0.8:
            criteria_met += 1
        
        # Quality criterion
        total_criteria += 1
        if context.average_confidence >= parameters.min_confidence:
            criteria_met += 1
        
        # Coverage criterion
        total_criteria += 1
        if context.coverage_score >= 0.8:
            criteria_met += 1
        
        # Convergence threshold check
        convergence_ratio = criteria_met / total_criteria if total_criteria > 0 else 0
        return convergence_ratio >= parameters.convergence_threshold
    
    def _calculate_diversity_score(self, thoughts: List[Thought]) -> float:
        """Calculate diversity score based on thought types and content"""
        if not thoughts:
            return 0.0
        
        # Count unique thought types
        unique_types = len(set(t.thought_type for t in thoughts))
        max_types = len(ThoughtType)
        type_diversity = unique_types / max_types
        
        # Count unique domains
        unique_domains = len(set(t.domain for t in thoughts if t.domain))
        domain_diversity = min(1.0, unique_domains / 3)  # Normalize to max 3 domains
        
        # Average novelty score
        novelty_diversity = sum(t.novelty_score for t in thoughts) / len(thoughts)
        
        return (type_diversity + domain_diversity + novelty_diversity) / 3
    
    def _calculate_coherence_score(self, thoughts: List[Thought], connections: List[Connection]) -> float:
        """Calculate coherence score based on connections and relationships"""
        if not thoughts or not connections:
            return 0.5  # Neutral score if no data
        
        # Connection density
        max_connections = len(thoughts) * (len(thoughts) - 1)
        connection_density = len(connections) / max_connections if max_connections > 0 else 0
        
        # Average connection strength
        avg_strength = sum(c.strength for c in connections) / len(connections)
        
        # Average connection confidence
        avg_confidence = sum(c.confidence for c in connections) / len(connections)
        
        return (connection_density + avg_strength + avg_confidence) / 3
    
    def _finalize_enhanced_output(self, unified_output: UnifiedOutput, expanded_input: ExpandedInput, 
                                context: ProcessingContext) -> UnifiedOutput:
        """Finalize output with enhanced analysis and recommendations"""
        
        # Create comprehensive thought graph
        if unified_output.thoughts:
            unified_output.graph = ThoughtGraph(
                thoughts=unified_output.thoughts,
                connections=unified_output.connections or [],
                clusters=unified_output.clusters or [],
                generation_parameters=expanded_input.parameters,
                total_nodes=len(unified_output.thoughts),
                total_edges=len(unified_output.connections or []),
                max_depth_achieved=max(t.depth for t in unified_output.thoughts),
                average_connectivity=context.connections_generated / context.nodes_generated if context.nodes_generated > 0 else 0,
                overall_quality=context.average_confidence,
                creativity_score=expanded_input.parameters.creativity_level,
                coherence_score=context.coherence_score,
                completeness_score=context.coverage_score
            )
        
        # Generate enhanced validation
        unified_output.validation = ValidationResult(
            is_valid=unified_output.success and context.average_confidence >= expanded_input.parameters.min_confidence,
            confidence=context.average_confidence,
            completeness_score=context.coverage_score,
            coherence_score=context.coherence_score,
            novelty_score=context.diversity_score,
            practicality_score=sum(t.practicality_score for t in unified_output.thoughts) / len(unified_output.thoughts) if unified_output.thoughts else 0.5
        )
        
        # Add quality-based recommendations
        if context.average_confidence < 0.7:
            unified_output.validation.suggestions.append("Consider increasing creativity level or reducing constraints")
        if context.diversity_score < 0.5:
            unified_output.validation.suggestions.append("Explore more diverse thought types and perspectives")
        if context.coherence_score < 0.6:
            unified_output.validation.suggestions.append("Strengthen connections between related ideas")
        
        # Generate summary and key takeaways
        unified_output.summary = f"Generated {context.nodes_generated} thoughts across {context.current_depth} depth levels with {context.connections_generated} connections. Quality score: {context.average_confidence:.2f}"
        
        unified_output.key_takeaways = [
            f"Primary focus: {expanded_input.parameters.domain_focus or 'General exploration'}",
            f"Generation approach: {expanded_input.parameters.generation_mode.value}",
            f"Depth vs breadth balance: {expanded_input.parameters.depth_vs_breadth:.1f}",
            f"Final quality metrics: Confidence {context.average_confidence:.2f}, Diversity {context.diversity_score:.2f}, Coherence {context.coherence_score:.2f}"
        ]
        
        # Generate actionable recommendations
        unified_output.recommendations = [
            "Review high-confidence thoughts for immediate implementation",
            "Explore connections between different thought clusters",
            "Consider practical constraints for low-practicality ideas",
            "Validate assumptions behind high-novelty concepts"
        ]
        
        # Add pattern analysis
        if unified_output.thoughts:
            thought_types = [t.thought_type.value for t in unified_output.thoughts]
            most_common_type = max(set(thought_types), key=thought_types.count)
            unified_output.patterns_discovered.append(f"Dominant thought pattern: {most_common_type}")
            
            high_impact_thoughts = [t for t in unified_output.thoughts if t.potential_impact >= 4]
            if high_impact_thoughts:
                unified_output.patterns_discovered.append(f"Identified {len(high_impact_thoughts)} high-impact opportunities")
        
        return unified_output

# Mock LM for testing
class MockLM:
    def __call__(self, prompt):
        return "Mock response for enhanced thought generation"

# Configure DSPy with mock LM for testing
mock_lm = MockLM()
dspy.settings.configure(lm=mock_lm)

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced parameters for complex brainstorming
    params = GraphParameters(
        max_nodes=25,
        max_depth=5,
        generation_mode="creative",
        creativity_level=0.8,
        divergence_factor=0.7,
        depth_vs_breadth=0.6,
        clustering_method="semantic",
        max_clusters=4,
        domain_focus="technology",
        enable_meta_thinking=True,
        enable_emotional_layer=True,
        enable_practical_layer=True
    )
    
    # Create enhanced input
    enhanced_input = ExpandedInput(
        original_text="sustainable urban transportation solutions",
        parameters=params,
        context={"urgency": "high", "budget": "medium", "timeline": "2-year"},
        constraints=["environmental regulations", "existing infrastructure"],
        requirements=["scalable", "cost-effective", "user-friendly"],
        domain_knowledge="Urban planning and green technology trends",
        focus_areas=["electric vehicles", "public transit", "smart infrastructure"]
    )
    
    # Run enhanced autonomous processing
    reactor = AutonomousThoughtReactor()
    result = reactor(enhanced_input, use_fake_data=True)
    
    print("🧠 Enhanced Autonomous Thought Graph Generation Complete!")
    print(f"✅ Success: {result.success}")
    print(f"🔄 Iterations: {result.iterations_used}")
    print(f"💭 Thoughts Generated: {len(result.thoughts) if result.thoughts else 0}")
    print(f"🔗 Connections: {len(result.connections) if result.connections else 0}")
    print(f"📊 Clusters: {len(result.clusters) if result.clusters else 0}")
    print(f"📈 Quality Score: {result.context.average_confidence:.2f}")
    print(f"🎯 Coverage: {result.context.coverage_score:.2f}")
    print(f"🌈 Diversity: {result.context.diversity_score:.2f}")
    print(f"🔄 Coherence: {result.context.coherence_score:.2f}")
    
    if result.insights:
        print(f"💡 Key Insights: {', '.join(result.insights[:3])}")
    
    if result.recommendations:
        print(f"🎯 Recommendations: {', '.join(result.recommendations[:2])}")