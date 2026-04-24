"""
tools_tester.py
------------------------------------------------------------------------------
Code written with assistance from Claude Sonnet 4.6
Tests the functions of tools.py and prints whether each test passes or fails.
"""
import os
import sys
sys.path.insert(0, '..')
from src.tools import calculate_shipping_cost, inventory_lookup, seed_inventory

INVENTORY_FILE = "./inventory.json"

# ── test harness ─────────────────────────────────────────────────────────────
 
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
 
results = []
 
def run(name, fn, *args, expect=None, contains=None, expect_type=None):
    try:
        result = fn(*args)
        ok = True
        reason = ""
        if expect is not None and result != expect:
            ok = False
            reason = f"expected {expect!r}, got {result!r}"
        if contains is not None and (result is None or contains not in str(result)):
            ok = False
            reason = f"expected output to contain {contains!r}, got {result!r}"
        if expect_type is not None and not isinstance(result, expect_type):
            ok = False
            reason = f"expected type {expect_type.__name__}, got {type(result).__name__}"
        label = PASS if ok else FAIL
        print(f"  [{label}] {name}")
        if not ok:
            print(f"         {reason}")
        results.append(ok)
        return result
    except Exception as e:
        print(f"  [{FAIL}] {name}")
        print(f"         raised {type(e).__name__}: {e}")
        results.append(False)
        return None
 
# ── setup ────────────────────────────────────────────────────────────────────
 
if os.path.exists(INVENTORY_FILE):
    os.remove(INVENTORY_FILE)
seed_inventory()
 
# ── calculate_shipping_cost ───────────────────────────────────────────────────
 
print("\ncalculate_shipping_cost")
run("100 km, 10 kg",        calculate_shipping_cost, 100, 10,  expect=52.0)
run("0 km, 0 kg",           calculate_shipping_cost, 0,   0,   expect=0.0)
run("1 km, 1 kg",           calculate_shipping_cost, 1,   1,   expect=0.7)
run("large values",         calculate_shipping_cost, 1000, 500, expect=600.0)
run("returns float",        calculate_shipping_cost, 10,  5,   expect_type=float)
 
# # ── parse_shipping_input ──────────────────────────────────────────────────────
 
# print("\nparse_shipping_input")
# run("'100,10' -> 52.0",     parse_shipping_input, "100,10",   expect=52.0)
# run("spaces around comma",  parse_shipping_input, "100 , 10", expect=52.0)
# run("floats in string",     parse_shipping_input, "50.5,20.0", expect_type=float)
 
# ── inventory_lookup ──────────────────────────────────────────────────────────
 
print("\ninventory_lookup")
run("fulfillable request",      inventory_lookup, "amoxicillin_500mg", 50,   contains="can be fulfilled")
run("over-request partial",     inventory_lookup, "amoxicillin_500mg", 200,  contains="only")
run("exact stock match",        inventory_lookup, "saline_bag_1l",     80,   contains="can be fulfilled")
run("unknown item = 0 stock",   inventory_lookup, "fake_drug",         1,    contains="out of stock")
run("space in item name",       inventory_lookup, "n95 mask",          5,    contains="available")
 
# # ── keep_inventory ────────────────────────────────────────────────────────────
 
# print("\nkeep_inventory")
# run("confirmed keep",           keep_inventory, "amoxicillin_500mg,50,yes",  contains="Confirmed")
# run("not confirmed",            keep_inventory, "amoxicillin_500mg,50,no",   contains="No inventory action")
# run("amount > stock",           keep_inventory, "amoxicillin_500mg,9999,yes",contains="kept unchanged")
# run("confirm case insensitive", keep_inventory, "ibuprofen_200mg,10,YES",    contains="Confirmed")
 
# # ── parse_model_output ────────────────────────────────────────────────────────
 
# print("\nparse_model_output")
# run("None input",               parse_model_output, None,                          expect=None)
# run("empty string",             parse_model_output, "",                            expect=None)
# run("plain JSON",               parse_model_output, '{"key": "val"}',              expect={"key": "val"})
# run("fenced ```json",           parse_model_output, '```json\n{"a":1}\n```',       expect={"a": 1})
# run("fenced ``` no lang",       parse_model_output, '```\n{"b":2}\n```',           expect={"b": 2})
# run("JSON buried in text",      parse_model_output, 'Here is the result: {"x":9}', expect={"x": 9})
# run("non-JSON string",          parse_model_output, "just some text",              expect=None)
# run("whitespace only",          parse_model_output, "   \n  ",                     expect=None)
# run("nested JSON",              parse_model_output, '{"a": {"b": [1,2,3]}}',       expect={"a": {"b": [1,2,3]}})
 
# ── summary ───────────────────────────────────────────────────────────────────
 
total = len(results)
passed = sum(results)
failed = total - passed
print(f"\n{'─'*40}")
print(f"Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} failed)")
else:
    print("  — all clear")
print()
 
# cleanup
if os.path.exists(INVENTORY_FILE):
    os.remove(INVENTORY_FILE)
