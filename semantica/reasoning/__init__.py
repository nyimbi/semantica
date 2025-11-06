"""
Reasoning Module

This module provides reasoning and inference capabilities for knowledge graph
analysis and query answering, supporting multiple reasoning strategies including
rule-based inference, SPARQL reasoning, abductive and deductive reasoning.

Key Features:
    - Rule-based inference with forward/backward chaining
    - SPARQL-based reasoning and query expansion
    - Rete algorithm implementation for efficient rule matching
    - Abductive reasoning for hypothesis generation
    - Deductive reasoning for logical inference
    - Rule management and execution tracking
    - Explanation generation for reasoning results
    - Proof generation and validation

Main Classes:
    - InferenceEngine: Rule-based inference with forward/backward chaining
    - SPARQLReasoner: SPARQL-based reasoning and query expansion
    - ReteEngine: Rete algorithm implementation for efficient rule matching
    - AbductiveReasoner: Abductive reasoning for hypothesis generation
    - DeductiveReasoner: Deductive reasoning for logical inference
    - RuleManager: Rule management and execution tracking
    - ExplanationGenerator: Explanation generation for reasoning results

Example Usage:
    >>> from semantica.reasoning import InferenceEngine, SPARQLReasoner
    >>> engine = InferenceEngine()
    >>> result = engine.infer(facts, rules)
    >>> reasoner = SPARQLReasoner()
    >>> query_result = reasoner.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")

Author: Semantica Contributors
License: MIT
"""

from .inference_engine import (
    InferenceEngine,
    InferenceResult,
    InferenceStrategy
)
from .sparql_reasoner import (
    SPARQLReasoner,
    SPARQLQueryResult
)
from .rete_engine import (
    ReteEngine,
    Fact,
    Match,
    ReteNode,
    AlphaNode,
    BetaNode,
    TerminalNode
)
from .abductive_reasoner import (
    AbductiveReasoner,
    Observation,
    Hypothesis,
    Explanation as AbductiveExplanation,
    HypothesisRanking
)
from .deductive_reasoner import (
    DeductiveReasoner,
    Premise,
    Conclusion,
    Proof,
    Argument
)
from .rule_manager import (
    RuleManager,
    Rule,
    RuleExecution,
    RuleType
)
from .explanation_generator import (
    ExplanationGenerator,
    Explanation,
    ReasoningStep,
    ReasoningPath,
    Justification
)

__all__ = [
    # Inference
    "InferenceEngine",
    "InferenceResult",
    "InferenceStrategy",
    
    # SPARQL reasoning
    "SPARQLReasoner",
    "SPARQLQueryResult",
    
    # Rete algorithm
    "ReteEngine",
    "Fact",
    "Match",
    "ReteNode",
    "AlphaNode",
    "BetaNode",
    "TerminalNode",
    
    # Abductive reasoning
    "AbductiveReasoner",
    "Observation",
    "Hypothesis",
    "AbductiveExplanation",
    "HypothesisRanking",
    
    # Deductive reasoning
    "DeductiveReasoner",
    "Premise",
    "Conclusion",
    "Proof",
    "Argument",
    
    # Rule management
    "RuleManager",
    "Rule",
    "RuleExecution",
    "RuleType",
    
    # Explanation
    "ExplanationGenerator",
    "Explanation",
    "ReasoningStep",
    "ReasoningPath",
    "Justification",
]
