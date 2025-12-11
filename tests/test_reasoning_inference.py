import pytest
from unittest.mock import MagicMock, patch
from semantica.reasoning import (
    InferenceEngine,
    RuleManager,
    ExplanationGenerator,
    InferenceResult,
    Rule,
    RuleType
)

# --- Fixtures ---

@pytest.fixture
def inference_engine():
    return InferenceEngine()

@pytest.fixture
def rule_manager():
    return RuleManager()

@pytest.fixture
def explanation_generator():
    return ExplanationGenerator()

# --- Test Rule Parsing (RuleManager) ---

def test_rule_parsing_simple(rule_manager):
    rule_def = "IF Person(?x) THEN Human(?x)"
    rule = rule_manager.define_rule(rule_def, name="HumanRule")
    
    assert rule.name == "HumanRule"
    assert rule.rule_type == RuleType.IMPLICATION
    assert len(rule.conditions) == 1
    assert rule.conditions[0] == "Person(?x)"
    assert rule.conclusion == "Human(?x)"

def test_rule_parsing_multiple_conditions(rule_manager):
    rule_def = "IF Parent(?x, ?y) AND Parent(?y, ?z) THEN Grandparent(?x, ?z)"
    rule = rule_manager.define_rule(rule_def)
    
    assert len(rule.conditions) == 2
    assert rule.conditions[0] == "Parent(?x, ?y)"
    assert rule.conditions[1] == "Parent(?y, ?z)"
    assert rule.conclusion == "Grandparent(?x, ?z)"

def test_rule_parsing_invalid(rule_manager):
    with pytest.raises(Exception):
        rule_manager.define_rule("INVALID RULE SYNTAX")

# --- Test InferenceEngine (Forward Chaining) ---

def test_forward_chaining_exact_match(inference_engine):
    # Rule: IF A THEN B
    inference_engine.add_rule("IF A THEN B")
    inference_engine.add_fact("A")
    
    results = inference_engine.forward_chain()
    
    assert len(results) >= 1
    inferred_facts = [res.conclusion for res in results]
    assert "B" in inferred_facts

def test_forward_chaining_simple(inference_engine):
    # Rule: IF Person(?x) THEN Human(?x)
    inference_engine.add_rule("IF Person(?x) THEN Human(?x)")
    
    # Fact: Person(Alice)
    inference_engine.add_fact("Person(Alice)")
    
    results = inference_engine.forward_chain()
    
    assert len(results) >= 1
    # Check if Human(Alice) is inferred
    inferred_facts = [res.conclusion for res in results]
    assert "Human(Alice)" in inferred_facts

def test_forward_chaining_transitive(inference_engine):
    # Rule: IF Parent(?a, ?b) AND Parent(?b, ?c) THEN Grandparent(?a, ?c)
    inference_engine.add_rule("IF Parent(?a, ?b) AND Parent(?b, ?c) THEN Grandparent(?a, ?c)")
    
    inference_engine.add_fact("Parent(Alice, Bob)")
    inference_engine.add_fact("Parent(Bob, Charlie)")
    
    results = inference_engine.forward_chain()
    
    inferred_facts = [res.conclusion for res in results]
    assert "Grandparent(Alice, Charlie)" in inferred_facts

def test_forward_chaining_no_match(inference_engine):
    inference_engine.add_rule("IF A(?x) THEN B(?x)")
    inference_engine.add_fact("C(Item)")
    
    results = inference_engine.forward_chain()
    assert len(results) == 0

# --- Test InferenceEngine (Backward Chaining) ---

def test_backward_chaining_success(inference_engine):
    # Rule: IF Parent(?a, ?b) AND Parent(?b, ?c) THEN Grandparent(?a, ?c)
    inference_engine.add_rule("IF Parent(?a, ?b) AND Parent(?b, ?c) THEN Grandparent(?a, ?c)")
    
    inference_engine.add_fact("Parent(Alice, Bob)")
    inference_engine.add_fact("Parent(Bob, Charlie)")
    
    # Goal: Prove Grandparent(Alice, Charlie)
    proof = inference_engine.backward_chain("Grandparent(Alice, Charlie)")
    
    assert proof is not None
    # Depending on implementation, proof might be a boolean or a Proof object
    # The notebook says: if proof: print(...)
    assert bool(proof) is True

def test_backward_chaining_failure(inference_engine):
    inference_engine.add_rule("IF A(?x) THEN B(?x)")
    inference_engine.add_fact("A(1)")
    
    proof = inference_engine.backward_chain("B(2)")
    assert not proof

# --- Test ExplanationGenerator ---

def test_explanation_generation(explanation_generator):
    # Create a dummy InferenceResult
    rule = Rule("r1", "test_rule", ["A(?x)"], "B(?x)")
    result = InferenceResult(
        conclusion="B(1)",
        premises=["A(1)"],
        rule_used=rule,
        confidence=1.0
    )
    
    explanation = explanation_generator.generate_explanation(result)
    
    assert explanation is not None
    assert explanation.conclusion == "B(1)"
    # Check if natural language explanation is generated
    assert isinstance(explanation.natural_language, str)
    assert "B(1)" in explanation.natural_language
    assert "test_rule" in explanation.natural_language or "r1" in explanation.natural_language

# --- Test Edge Cases ---

def test_duplicate_facts(inference_engine):
    assert inference_engine.add_fact("A(1)") is True
    assert inference_engine.add_fact("A(1)") is False  # Duplicate
    assert len(inference_engine.facts) == 1

def test_max_iterations(inference_engine):
    # Create a rule that generates infinite facts if not controlled
    # e.g., IF Num(?x) THEN Num(?x+1) - hard to simulate with string matching without eval
    # Instead, let's just test config setting
    engine = InferenceEngine(max_iterations=5)
    assert engine.max_iterations == 5
