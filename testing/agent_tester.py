"""
agent_tester.py
------------------------------------------------------------------------------
Code written with assistance from Claude Sonnet 4.6
Run all tests in parallel (avoids real LLM calls / rate limits via mocking):
    python3 -m pytest agent_tester.py -n auto -v
"""

import sys
import os
from unittest.mock import MagicMock, patch
import builtins

# ---------------------------------------------------------------------------
# Path setup + module stubs — must happen before any agent import
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

sys.modules["callbacks"] = MagicMock()
sys.modules["tools"] = MagicMock()
sys.modules["rag"] = MagicMock()
sys.modules["langchain_google_genai"] = MagicMock()

builtins.input = lambda _="": "test input"

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_shipping_tool():
    tool = MagicMock()
    tool.name = "get_shipping_cost"
    tool.invoke = MagicMock(return_value=42.50)
    return tool


@pytest.fixture
def mock_inventory_tool():
    tool = MagicMock()
    tool.name = "get_inventory_lookup"
    tool.invoke = MagicMock(
        return_value="Item: surgical gloves | Requested: 100 | Available: 200 | Status: FULFILLABLE"
    )
    return tool


@pytest.fixture
def mock_rag_tool():
    tool = MagicMock()
    tool.name = "get_rag_info"
    tool.invoke = MagicMock(
        return_value="Surgical gloves are sterile, single-use protective equipment stored at room temperature."
    )
    return tool


@pytest.fixture
def all_tools(mock_shipping_tool, mock_inventory_tool, mock_rag_tool):
    return [mock_shipping_tool, mock_inventory_tool, mock_rag_tool]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    ai_msg = AIMessage(content="I can help with medical logistics.")
    ai_msg.tool_calls = []
    llm.invoke = MagicMock(return_value=ai_msg)
    return llm


def make_tool_call_message(tool_name: str, args: dict, call_id: str) -> AIMessage:
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": tool_name, "args": args, "id": call_id}]
    return msg


def make_final_message(content: str) -> AIMessage:
    msg = AIMessage(content=content)
    msg.tool_calls = []
    return msg


# ---------------------------------------------------------------------------
# find_tool_by_name
# ---------------------------------------------------------------------------

class TestFindToolByName:
    def test_finds_existing_tool(self, all_tools):
        from agent import find_tool_by_name
        tool = find_tool_by_name(all_tools, "get_shipping_cost")
        assert tool.name == "get_shipping_cost"

    def test_raises_for_missing_tool(self, all_tools):
        from agent import find_tool_by_name
        with pytest.raises(ValueError, match="not found"):
            find_tool_by_name(all_tools, "nonexistent_tool")

    def test_finds_inventory_tool(self, all_tools):
        from agent import find_tool_by_name
        tool = find_tool_by_name(all_tools, "get_inventory_lookup")
        assert tool.name == "get_inventory_lookup"

    def test_finds_rag_tool(self, all_tools):
        from agent import find_tool_by_name
        tool = find_tool_by_name(all_tools, "get_rag_info")
        assert tool.name == "get_rag_info"


# ---------------------------------------------------------------------------
# Tool unit tests (no LLM involved)
# ---------------------------------------------------------------------------

class TestShippingTool:
    @patch("agent.calculate_shipping_cost", return_value=25.00)
    def test_basic_cost(self, mock_calc):
        from agent import get_shipping_cost
        result = get_shipping_cost.invoke({"distance_km": 100.0, "weight_kg": 5.0})
        assert result == 25.00
        mock_calc.assert_called_once_with(100.0, 5.0)

    @patch("agent.calculate_shipping_cost", return_value=0.0)
    def test_zero_distance_zero_weight(self, mock_calc):
        from agent import get_shipping_cost
        result = get_shipping_cost.invoke({"distance_km": 0.0, "weight_kg": 0.0})
        assert result == 0.0

    @patch("agent.calculate_shipping_cost", return_value=999.99)
    def test_large_shipment(self, mock_calc):
        from agent import get_shipping_cost
        result = get_shipping_cost.invoke({"distance_km": 5000.0, "weight_kg": 500.0})
        assert result == 999.99


class TestInventoryTool:
    @patch("agent.inventory_lookup", return_value="Status: FULFILLABLE")
    def test_fulfillable_item(self, mock_lookup):
        from agent import get_inventory_lookup
        result = get_inventory_lookup.invoke({"item": "surgical gloves", "num_required": 50})
        assert "FULFILLABLE" in result
        mock_lookup.assert_called_once_with("surgical gloves", 50)

    @patch("agent.inventory_lookup", return_value="Status: INSUFFICIENT STOCK")
    def test_insufficient_stock(self, mock_lookup):
        from agent import get_inventory_lookup
        result = get_inventory_lookup.invoke({"item": "ventilators", "num_required": 1000})
        assert "INSUFFICIENT" in result

    @patch("agent.inventory_lookup", return_value="Status: ITEM NOT FOUND")
    def test_unknown_item(self, mock_lookup):
        from agent import get_inventory_lookup
        result = get_inventory_lookup.invoke({"item": "unknown_item_xyz", "num_required": 1})
        assert "NOT FOUND" in result


class TestRagTool:
    @patch("agent.rag_chain")
    def test_returns_string(self, mock_chain):
        mock_chain.invoke.return_value = "Surgical masks filter airborne particles."
        from agent import get_rag_info
        result = get_rag_info.invoke({"question": "What are surgical masks for?"})
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("agent.rag_chain")
    def test_passes_question_to_chain(self, mock_chain):
        mock_chain.invoke.return_value = "Answer about gloves."
        from agent import get_rag_info
        get_rag_info.invoke({"question": "Tell me about gloves"})
        mock_chain.invoke.assert_called_once_with("Tell me about gloves")


# ---------------------------------------------------------------------------
# Agent loop tests
# ---------------------------------------------------------------------------

class TestAgentLoop:
    def _run_loop(self, llm, tools):
        from agent import find_tool_by_name
        messages = [HumanMessage(content="test input")]

        while True:
            ai_message = llm.invoke(messages)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if tool_calls:
                messages.append(ai_message)
                for tc in tool_calls:
                    t = find_tool_by_name(tools, tc["name"])
                    obs = t.invoke(tc["args"])
                    messages.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))  # type: ignore
                continue
            return ai_message

    def test_no_tool_calls_returns_immediately(self, mock_llm, all_tools):
        result = self._run_loop(mock_llm, all_tools)
        assert result.content == "I can help with medical logistics."
        assert mock_llm.invoke.call_count == 1

    def test_single_tool_call_then_final_answer(self, all_tools):
        llm = MagicMock()
        llm.invoke.side_effect = [
            make_tool_call_message("get_shipping_cost", {"distance_km": 200.0, "weight_kg": 10.0}, "call_1"),
            make_final_message("Shipping will cost $42.50."),
        ]
        result = self._run_loop(llm, all_tools)
        assert "42.50" in result.content or result.content == "Shipping will cost $42.50."
        assert llm.invoke.call_count == 2

    def test_multiple_sequential_tool_calls(self, all_tools):
        llm = MagicMock()
        llm.invoke.side_effect = [
            make_tool_call_message("get_inventory_lookup", {"item": "gloves", "num_required": 100}, "call_1"),
            make_tool_call_message("get_shipping_cost", {"distance_km": 300.0, "weight_kg": 5.0}, "call_2"),
            make_final_message("Order is feasible and shipping cost calculated."),
        ]
        result = self._run_loop(llm, all_tools)
        assert result.content == "Order is feasible and shipping cost calculated."
        assert llm.invoke.call_count == 3
        
# ---------------------------------------------------------------------------
# Load / stress tests — multiple concurrent tool requests
# ---------------------------------------------------------------------------

import threading
import time

class TestToolLoadConcurrency:
    """Simulate concurrent tool invocations to surface race conditions
    or shared-state bugs (e.g. inventory file contention)."""

    @patch("agent.calculate_shipping_cost", side_effect=lambda d, w: round(d * 0.5 + w * 0.2, 2))
    def test_shipping_tool_concurrent_requests(self, mock_calc):
        """Fire 50 simultaneous shipping cost requests across threads."""
        from agent import get_shipping_cost

        NUM_THREADS = 50
        results = []
        errors = []
        lock = threading.Lock()

        def call_tool(i):
            try:
                result = get_shipping_cost.invoke({
                    "distance_km": float(i * 10),
                    "weight_kg": float(i)
                })
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=call_tool, args=(i,)) for i in range(1, NUM_THREADS + 1)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        assert len(errors) == 0, f"Errors during concurrent calls: {errors}"
        assert len(results) == NUM_THREADS
        assert elapsed < 5.0, f"Concurrent calls took too long: {elapsed:.2f}s"

    @patch("agent.inventory_lookup", side_effect=lambda item, qty: f"Status: FULFILLABLE for {item}")
    def test_inventory_tool_concurrent_requests(self, mock_lookup):
        """Fire 50 simultaneous inventory lookups for different items."""
        from agent import get_inventory_lookup

        NUM_THREADS = 50
        items = [f"item_{i}" for i in range(NUM_THREADS)]
        results = {}
        errors = []
        lock = threading.Lock()

        def call_tool(item):
            try:
                result = get_inventory_lookup.invoke({"item": item, "num_required": 10})
                with lock:
                    results[item] = result
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=call_tool, args=(item,)) for item in items]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == NUM_THREADS
        # Each result should correspond to the correct item (no cross-contamination)
        for item, result in results.items():
            assert item in result, f"Result for {item} doesn't contain item name: {result}"

    def test_agent_loop_concurrent_sessions(self, all_tools):
        """Simulate 20 independent agent sessions running simultaneously,
        each making a tool call and receiving a final answer."""
        from agent import find_tool_by_name

        NUM_SESSIONS = 20
        outcomes = []
        errors = []
        lock = threading.Lock()

        def run_session(session_id):
            try:
                llm = MagicMock()
                llm.invoke.side_effect = [
                    make_tool_call_message(
                        "get_shipping_cost",
                        {"distance_km": float(session_id * 5), "weight_kg": 2.0},
                        f"call_{session_id}"
                    ),
                    make_final_message(f"Session {session_id} complete."),
                ]

                messages = [HumanMessage(content=f"Session {session_id} query")]
                while True:
                    ai_message = llm.invoke(messages)
                    tool_calls = getattr(ai_message, "tool_calls", None) or []
                    if tool_calls:
                        messages.append(ai_message)
                        for tc in tool_calls:
                            t = find_tool_by_name(all_tools, tc["name"])
                            obs = t.invoke(tc["args"])
                            messages.append(ToolMessage(content=str(obs), tool_call_id=tc["id"])) # type: ignore
                        continue
                    with lock:
                        outcomes.append(ai_message.content)
                    return

            except Exception as e:
                with lock:
                    errors.append((session_id, e))

        threads = [threading.Thread(target=run_session, args=(i,)) for i in range(NUM_SESSIONS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Session errors: {errors}"
        assert len(outcomes) == NUM_SESSIONS
        for i, outcome in enumerate(outcomes):
            assert "complete" in outcome or "Session" in outcome