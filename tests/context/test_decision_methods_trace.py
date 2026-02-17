"""Tests for decision trace capture convenience API."""

from datetime import datetime
from unittest.mock import Mock

from semantica.context.decision_methods import capture_decision_trace
from semantica.context.decision_models import Decision


def _sample_decision() -> Decision:
    return Decision(
        decision_id="decision_trace_test_001",
        category="credit_approval",
        scenario="Credit line increase for long-term customer",
        reasoning="Strong payment history and low utilization",
        outcome="approved",
        confidence=0.92,
        timestamp=datetime.now(),
        decision_maker="ai_agent",
    )


def test_capture_decision_trace_without_graph_store_is_backward_compatible():
    decision = _sample_decision()
    decision_id = capture_decision_trace(decision, cross_system_context={})
    assert decision_id == decision.decision_id


def test_capture_decision_trace_with_graph_store_records_trace_events():
    decision = _sample_decision()
    graph_store = Mock()
    graph_store.execute_query = Mock(return_value=[])

    decision_id = capture_decision_trace(
        decision=decision,
        cross_system_context={"crm": {"arr": 120000}},
        graph_store=graph_store,
        entities=["customer_123"],
        source_documents=["renewal_note_001"],
        policy_ids=["renewal_discount_policy_v3_2"],
        approvals=[
            {
                "approver": "vp_finance",
                "approval_method": "slack_dm",
                "approval_context": "Approved due to SEV-1 impact history",
            }
        ],
        precedents=[
            {
                "precedent_id": "decision_legacy_001",
                "relationship_type": "similar_scenario",
            }
        ],
        immutable_audit_log=True,
    )

    assert decision_id == decision.decision_id
    assert graph_store.execute_query.call_count > 0


def test_capture_decision_trace_accepts_legacy_payload_shapes():
    decision = _sample_decision()
    graph_store = Mock()
    graph_store.execute_query = Mock(return_value=[{"t": {"trace_id": "decision_trace_test_001:4", "event_index": 4, "event_hash": "abc"}}])

    decision_id = capture_decision_trace(
        decision=decision,
        cross_system_context={"crm": {"arr": 120000}},
        graph_store=graph_store,
        entities="customer_legacy_001",
        source_documents="doc_legacy_001",
        policy_ids="policy_legacy_v1",
        exceptions={"policy_id": "policy_legacy_v1", "reason": "legacy exception"},
        approvals={"approver": "vp_ops", "approval_method": "email"},
        precedents=["decision_legacy_001"],
        immutable_audit_log=True,
    )

    assert decision_id == decision.decision_id
    assert graph_store.execute_query.call_count > 0


def test_capture_decision_trace_accepts_versioned_policy_refs():
    decision = _sample_decision()
    graph_store = Mock()
    graph_store.execute_query = Mock(
        return_value={"records": [{"policy_id": "renewal_discount_policy", "version": "3.2"}]}
    )

    decision_id = capture_decision_trace(
        decision=decision,
        cross_system_context={"crm": {"arr": 120000}},
        graph_store=graph_store,
        policy_ids=[{"policy_id": "renewal_discount_policy", "version": "3.2"}],
        immutable_audit_log=False,
    )

    assert decision_id == decision.decision_id
    policy_calls = [
        c for c in graph_store.execute_query.call_args_list
        if "policy_version" in c[0][1]
    ]
    assert policy_calls
    assert policy_calls[0][0][1]["policy_version"] == "3.2"
