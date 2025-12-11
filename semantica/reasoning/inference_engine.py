"""
Inference Engine Module

This module provides rule-based inference capabilities for knowledge graph
reasoning and analysis, supporting forward chaining, backward chaining, and
bidirectional inference strategies.

Key Features:
    - Rule-based inference and reasoning
    - Forward and backward chaining
    - Bidirectional inference
    - Rule management and execution
    - Performance optimization
    - Error handling and recovery
    - Custom rule support

Main Classes:
    - InferenceEngine: Rule-based inference engine
    - InferenceResult: Dataclass for inference results
    - InferenceStrategy: Enum for inference strategies (forward, backward, bidirectional)

Example Usage:
    >>> from semantica.reasoning import InferenceEngine, InferenceStrategy
    >>> engine = InferenceEngine()
    >>> result = engine.infer(facts, rules, strategy=InferenceStrategy.FORWARD)
    >>> conclusions = engine.forward_chain(facts, rules)

Author: Semantica Contributors
License: MIT
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .rule_manager import Rule, RuleManager


class InferenceStrategy(Enum):
    """Inference strategies."""

    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class InferenceResult:
    """Inference result."""

    conclusion: Any
    premises: List[Any] = field(default_factory=list)
    rule_used: Optional[Rule] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """
    Rule-based inference engine.

    • Rule-based inference and reasoning
    • Forward and backward chaining
    • Rule management and execution
    • Performance optimization
    • Error handling and recovery
    • Custom rule support
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize inference engine.

        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - strategy: Inference strategy (forward, backward, bidirectional)
                - max_iterations: Maximum inference iterations
        """
        self.logger = get_logger("inference_engine")
        self.config = config or {}
        self.config.update(kwargs)

        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker()

        self.rule_manager = RuleManager(**self.config)
        self.strategy = InferenceStrategy(self.config.get("strategy", "forward"))
        self.max_iterations = self.config.get("max_iterations", 100)

        self.facts: Set[Any] = set()
        self.unhashable_facts: List[Any] = []
        self.inferred_facts: List[InferenceResult] = []

    def add_rule(self, rule_definition: str, **options) -> Rule:
        """
        Add inference rule to engine.

        Args:
            rule_definition: Rule definition string or Rule object
            **options: Additional options

        Returns:
            Created rule
        """
        if isinstance(rule_definition, str):
            rule = self.rule_manager.define_rule(rule_definition, **options)
        else:
            rule = rule_definition

        self.rule_manager.add_rule(rule)
        self.logger.debug(f"Added rule: {rule.name}")

        return rule

    def add_fact(self, fact: Any) -> bool:
        """
        Add fact to knowledge base.

        Args:
            fact: Fact to add

        Returns:
            True if fact was newly added, False if it already existed
        """
        try:
            if fact in self.facts:
                return False
            self.facts.add(fact)
            self.logger.debug(f"Added fact: {fact}")
            return True
        except TypeError:
            if fact not in self.unhashable_facts:
                self.unhashable_facts.append(fact)
                self.logger.debug(f"Added unhashable fact: {fact}")
                return True
            return False

    def add_facts(self, facts: List[Any]) -> None:
        """
        Add multiple facts.

        Args:
            facts: List of facts
        """
        for fact in facts:
            self.add_fact(fact)

    def forward_chain(
        self, facts: Optional[List[Any]] = None, **options
    ) -> List[InferenceResult]:
        """
        Perform forward chaining inference.

        Args:
            facts: Optional initial facts
            **options: Additional options

        Returns:
            List of inference results
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="reasoning",
            submodule="InferenceEngine",
            message="Performing forward chaining inference",
        )

        try:
            if facts:
                self.progress_tracker.update_tracking(
                    tracking_id, message=f"Adding {len(facts)} initial facts..."
                )
                self.add_facts(facts)

            new_facts = True
            iterations = 0
            results = []

            self.progress_tracker.update_tracking(
                tracking_id, message="Starting forward chaining iterations..."
            )
            while new_facts and iterations < self.max_iterations:
                new_facts = False
                iterations += 1

                # Get all rules
                rules = self.rule_manager.get_all_rules()
                self.progress_tracker.update_tracking(
                    tracking_id,
                    message=f"Iteration {iterations}: Checking {len(rules)} rules...",
                )

                for rule in rules:
                    # Find all matches for the rule
                    matches = self._find_matches(rule.conditions, {})
                    
                    for bindings in matches:
                        # Apply rule with bindings
                        result = self._apply_rule(rule, bindings=bindings)
                        if result:
                            # Only consider it a new inference if the fact wasn't already known
                            if self.add_fact(result.conclusion):
                                results.append(result)
                                self.inferred_facts.append(result)
                                new_facts = True

            self.logger.info(
                f"Forward chaining completed: {len(results)} inferences in {iterations} iterations"
            )
            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Forward chaining completed: {len(results)} inferences in {iterations} iterations",
            )
            return results

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def backward_chain(self, goal: Any, **options) -> Optional[InferenceResult]:
        """
        Perform backward chaining inference.

        Args:
            goal: Goal to prove
            **options: Additional options

        Returns:
            Inference result or None
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="reasoning",
            submodule="InferenceEngine",
            message=f"Performing backward chaining for goal: {goal}",
        )

        try:
            # Check if goal is already a fact
            self.progress_tracker.update_tracking(
                tracking_id, message="Checking if goal is already a fact..."
            )
            
            # Check for direct match or unification with facts
            found_fact = None
            
            # First try direct match (fastest)
            try:
                if goal in self.facts:
                    found_fact = goal
            except TypeError:
                if goal in self.unhashable_facts:
                    found_fact = goal
            
            # If not found and goal looks like a pattern (string with ?), try unification
            if found_fact is None and isinstance(goal, str) and "?" in goal:
                for fact in self.facts:
                    if isinstance(fact, str):
                        # Try to unify to see if it matches
                        if self._unify(goal, fact, {}) is not None:
                            found_fact = fact
                            break
            
            if found_fact:
                self.progress_tracker.stop_tracking(
                    tracking_id, status="completed", message=f"Goal proven by fact: {found_fact}"
                )
                return InferenceResult(conclusion=found_fact, confidence=1.0)

            # Find rules that can prove the goal
            self.progress_tracker.update_tracking(
                tracking_id, message="Finding rules that can prove the goal..."
            )
            rules = self.rule_manager.get_all_rules()
            
            # Use unified matching for finding applicable rules
            applicable_rules_and_bindings = []
            for r in rules:
                bindings = self._unify(r.conclusion, goal, {})
                if bindings is not None:
                    applicable_rules_and_bindings.append((r, bindings))

            self.progress_tracker.update_tracking(
                tracking_id,
                message=f"Found {len(applicable_rules_and_bindings)} applicable rules, trying to prove premises...",
            )
            
            for rule, initial_bindings in applicable_rules_and_bindings:
                # Try to prove premises with bindings, propagating bindings between premises
                current_bindings = initial_bindings.copy()
                premises_results = []
                all_premises_proven = True
                
                for cond in rule.conditions:
                    # Instantiate condition with current bindings
                    instantiated_cond = self._substitute_bindings(cond, current_bindings)
                    
                    # Recursively prove this condition
                    premise_result = self.backward_chain(instantiated_cond, **options)
                    
                    if premise_result:
                        premises_results.append(premise_result.conclusion)
                        
                        # If the premise had variables, update bindings based on the proven fact
                        # We unify the instantiated condition (which might still have vars) with the proven conclusion
                        new_bindings = self._unify(instantiated_cond, premise_result.conclusion, current_bindings)
                        if new_bindings is not None:
                            current_bindings = new_bindings
                        else:
                            # This implies a conflict, which shouldn't happen if backward_chain returned success
                            # on instantiated_cond, but good to be safe
                            all_premises_proven = False
                            break
                    else:
                        all_premises_proven = False
                        break

                if all_premises_proven:
                    # All premises proven, rule can fire
                    result = self._apply_rule(rule, premises=premises_results, bindings=current_bindings)
                    if result:
                        self.progress_tracker.stop_tracking(
                            tracking_id,
                            status="completed",
                            message=f"Successfully proved goal using rule: {rule.name}",
                        )
                        return result

            self.progress_tracker.stop_tracking(
                tracking_id, status="completed", message="Could not prove goal"
            )
            return None

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def _parse_predicate(self, text: str) -> tuple[str, List[str]]:
        """Parse 'Predicate(arg1, arg2)' into ('Predicate', ['arg1', 'arg2'])."""
        if not isinstance(text, str):
            return text, []
        match = re.match(r"(\w+)\((.+)\)", text)
        if not match:
            return text, []
        predicate = match.group(1)
        args = [arg.strip() for arg in match.group(2).split(",")]
        return predicate, args

    def _unify(self, condition: str, fact: str, bindings: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Try to unify a condition (with vars) against a fact.
        Returns new bindings if successful, None otherwise.
        """
        # Handle exact string match shortcut
        if condition == fact:
            return bindings
            
        cond_pred, cond_args = self._parse_predicate(condition)
        fact_pred, fact_args = self._parse_predicate(fact)
        
        if cond_pred != fact_pred:
            return None
        if len(cond_args) != len(fact_args):
            return None
            
        new_bindings = bindings.copy()
        for c_arg, f_arg in zip(cond_args, fact_args):
            if c_arg.startswith("?"):
                if c_arg in new_bindings:
                    if new_bindings[c_arg] != f_arg:
                        return None # Conflict
                else:
                    new_bindings[c_arg] = f_arg
            else:
                if c_arg != f_arg:
                    return None # Constant mismatch
        return new_bindings

    def _find_matches(self, conditions: List[str], bindings: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Recursively find all bindings that satisfy the conditions.
        """
        if not conditions:
            return [bindings]
            
        first = conditions[0]
        # Substitute current bindings into first condition before matching
        first_substituted = self._substitute_bindings(first, bindings)
        rest = conditions[1:]
        
        valid_bindings = []
        
        # Try to match 'first' against all facts
        for fact in self.facts:
            # Skip if fact is not a string (unhashable/objects) for now, or handle str()
            if not isinstance(fact, str):
                continue
                
            unified = self._unify(first_substituted, fact, bindings)
            if unified is not None:
                # Recursive step
                results = self._find_matches(rest, unified)
                valid_bindings.extend(results)
                
        return valid_bindings

    def _substitute_bindings(self, text: str, bindings: Dict[str, str]) -> str:
        """Substitute variables in text with bindings."""
        if not isinstance(text, str):
            return text
        pred, args = self._parse_predicate(text)
        if not args:
            return text
            
        new_args = []
        for arg in args:
            if arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)
        
        return f"{pred}({', '.join(new_args)})"

    def _apply_rule(
        self, rule: Rule, premises: Optional[List[Any]] = None, bindings: Optional[Dict[str, str]] = None
    ) -> Optional[InferenceResult]:
        """Apply rule and return inference result."""
        conclusion = rule.conclusion
        if bindings:
            conclusion = self._substitute_bindings(conclusion, bindings)

        if premises is None:
            # Reconstruct premises from bindings if not provided (approximate)
            premises = [self._substitute_bindings(c, bindings or {}) for c in rule.conditions]

        result = InferenceResult(
            conclusion=conclusion,
            premises=premises,
            rule_used=rule,
            confidence=rule.confidence,
            metadata={"rule_name": rule.name, "rule_id": rule.rule_id, "bindings": bindings},
        )

        return result


    def infer(self, query: Any, **options) -> List[InferenceResult]:
        """
        Perform inference based on strategy.

        Args:
            query: Query or goal
            **options: Additional options

        Returns:
            List of inference results
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="reasoning",
            submodule="InferenceEngine",
            message=f"Performing inference using {self.strategy.value} strategy",
        )

        try:
            if self.strategy == InferenceStrategy.FORWARD:
                self.progress_tracker.update_tracking(
                    tracking_id, message="Using forward chaining strategy..."
                )
                results = self.forward_chain(**options)
            elif self.strategy == InferenceStrategy.BACKWARD:
                self.progress_tracker.update_tracking(
                    tracking_id, message="Using backward chaining strategy..."
                )
                result = self.backward_chain(query, **options)
                results = [result] if result else []
            else:  # BIDIRECTIONAL
                self.progress_tracker.update_tracking(
                    tracking_id,
                    message="Using bidirectional strategy: forward chaining...",
                )
                forward_results = self.forward_chain(**options)
                self.progress_tracker.update_tracking(
                    tracking_id,
                    message="Using bidirectional strategy: backward chaining...",
                )
                backward_result = self.backward_chain(query, **options)
                if backward_result:
                    forward_results.append(backward_result)
                results = forward_results

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Inference complete: {len(results)} results using {self.strategy.value} strategy",
            )
            return results

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def get_facts(self) -> List[Any]:
        """Get all facts."""
        return list(self.facts) + self.unhashable_facts

    def get_inferred_facts(self) -> List[InferenceResult]:
        """Get all inferred facts."""
        return list(self.inferred_facts)

    def clear_facts(self) -> None:
        """Clear all facts."""
        self.facts.clear()
        self.unhashable_facts.clear()
        self.inferred_facts.clear()

    def reset(self) -> None:
        """Reset inference engine."""
        self.clear_facts()
        self.rule_manager.clear_rules()
