import pytest
from semantica.reasoning import (
    DeductiveReasoner,
    AbductiveReasoner,
    Premise,
    Observation,
    HypothesisRanking,
    Argument
)

# --- DeductiveReasoner Tests ---

@pytest.fixture
def deductive_reasoner():
    return DeductiveReasoner()

def test_deductive_apply_logic(deductive_reasoner):
    # Rule: IF Human(?x) THEN Mortal(?x)
    rule = deductive_reasoner.rule_manager.define_rule("IF Human(?x) THEN Mortal(?x)")
    deductive_reasoner.rule_manager.add_rule(rule)
    
    premises = [Premise("p1", "Human(Socrates)")]
    
    # We need to ensure the reasoner can handle variable matching or at least exact matching
    # Based on my reading of DeductiveReasoner, it uses _can_apply_rule which checks strict containment
    # if variables aren't handled.
    # Let's check if DeductiveReasoner uses InferenceEngine's unification or its own simple logic.
    # It uses self._can_apply_rule which does: condition in premise_statements.
    # This implies EXACT string match unless updated.
    
    # So for now, let's test exact match to verify baseline behavior
    deductive_reasoner.rule_manager.clear_rules()
    rule = deductive_reasoner.rule_manager.define_rule("IF Human(Socrates) THEN Mortal(Socrates)")
    deductive_reasoner.rule_manager.add_rule(rule)
    
    conclusions = deductive_reasoner.apply_logic(premises)
    
    assert len(conclusions) >= 1
    assert conclusions[0].statement == "Mortal(Socrates)"

def test_deductive_apply_logic_with_variables(deductive_reasoner):
    # This test checks if DeductiveReasoner supports variables like InferenceEngine
    rule = deductive_reasoner.rule_manager.define_rule("IF Human(?x) THEN Mortal(?x)")
    deductive_reasoner.rule_manager.add_rule(rule)
    
    premises = [Premise("p1", "Human(Plato)")]
    
    # If DeductiveReasoner supports variables, it should deduce Mortal(Plato)
    conclusions = deductive_reasoner.apply_logic(premises)
    
    assert len(conclusions) > 0
    assert conclusions[0].statement == "Mortal(Plato)"

def test_deductive_prove_theorem_with_variables(deductive_reasoner):
    # Rule: IF Parent(?a, ?b) THEN Ancestor(?a, ?b)
    rule = deductive_reasoner.rule_manager.define_rule("IF Parent(?a, ?b) THEN Ancestor(?a, ?b)")
    deductive_reasoner.rule_manager.add_rule(rule)
    deductive_reasoner.add_fact("Parent(Zeus, Ares)")
    
    # Prove Ancestor(Zeus, Ares)
    proof = deductive_reasoner.prove_theorem("Ancestor(Zeus, Ares)")
    
    assert proof is not None
    assert proof.valid is True
    assert proof.steps[-1].statement == "Ancestor(Zeus, Ares)"

def test_deductive_prove_theorem(deductive_reasoner):
    # Rule: IF P THEN Q
    rule = deductive_reasoner.rule_manager.define_rule("IF P THEN Q")
    deductive_reasoner.rule_manager.add_rule(rule)
    deductive_reasoner.add_fact("P")
    
    proof = deductive_reasoner.prove_theorem("Q")
    
    assert proof is not None
    assert proof.valid is True
    assert len(proof.steps) > 0
    assert proof.steps[-1].statement == "Q"

def test_deductive_validate_argument(deductive_reasoner):
    rule = deductive_reasoner.rule_manager.define_rule("IF Rain THEN Wet")
    deductive_reasoner.rule_manager.add_rule(rule)
    deductive_reasoner.add_fact("Rain")
    
    premises = [Premise("p1", "Rain")]
    # Note: Conclusion object needed for argument? Argument class has 'conclusion' field which is Conclusion type.
    # But usually we validate if premises lead to a conclusion statement.
    # The validate_argument method checks if argument.conclusion.statement follows.
    
    from semantica.reasoning.deductive_reasoner import Conclusion
    conc = Conclusion("c1", "Wet")
    
    arg = Argument("arg1", premises=premises, conclusion=conc)
    
    result = deductive_reasoner.validate_argument(arg)
    
    assert result["valid"] is True

# --- AbductiveReasoner Tests ---

@pytest.fixture
def abductive_reasoner():
    return AbductiveReasoner()

def test_abductive_generate_hypotheses(abductive_reasoner):
    # Setup a rule that could explain an observation
    # Rule: IF Rain THEN WetGrass
    rule = abductive_reasoner.rule_manager.define_rule("IF Rain THEN WetGrass")
    abductive_reasoner.rule_manager.add_rule(rule)
    
    obs = Observation("o1", "WetGrass")
    
    # Current implementation of _rule_explains_observation returns True for everything
    # So it should find the rule as a hypothesis
    hypotheses = abductive_reasoner.generate_hypotheses([obs])
    
    assert len(hypotheses) > 0
    assert "Rain" in hypotheses[0].premises  # The premise of the rule is the hypothesis (Rain caused WetGrass)

def test_abductive_filtering(abductive_reasoner):
    # Rule 1: IF Rain THEN WetGrass
    r1 = abductive_reasoner.rule_manager.define_rule("IF Rain THEN WetGrass")
    abductive_reasoner.rule_manager.add_rule(r1)
    
    # Rule 2: IF Fire THEN Smoke
    r2 = abductive_reasoner.rule_manager.define_rule("IF Fire THEN Smoke")
    abductive_reasoner.rule_manager.add_rule(r2)
    
    obs = Observation("o1", "WetGrass")
    
    hypotheses = abductive_reasoner.generate_hypotheses([obs])
    
    # Should only find hypothesis related to WetGrass (Rain)
    # Should NOT find hypothesis related to Smoke (Fire)
    
    relevant_hypotheses = [h for h in hypotheses if "Rain" in h.premises or "IF Rain" in h.explanation]
    irrelevant_hypotheses = [h for h in hypotheses if "Fire" in h.premises or "IF Fire" in h.explanation]
    
    assert len(relevant_hypotheses) > 0
    assert len(irrelevant_hypotheses) == 0

def test_abductive_find_explanations(abductive_reasoner):
    rule = abductive_reasoner.rule_manager.define_rule("IF Fire THEN Smoke")
    abductive_reasoner.rule_manager.add_rule(rule)
    obs = Observation("o1", "Smoke")
    
    explanations = abductive_reasoner.find_explanations([obs])
    
    assert len(explanations) == 1
    assert explanations[0].best_hypothesis is not None
    assert "Fire" in explanations[0].best_hypothesis.premises

