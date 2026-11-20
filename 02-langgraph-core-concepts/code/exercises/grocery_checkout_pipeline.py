"""Grocery checkout pipeline (exercise): four nodes, one job each, partial updates only.

Fill in every TODO. The graph wiring and runner are already done, so the file
runs as-is: until you implement the nodes they return {} and nothing changes.
"""
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# Step 1: Reference data for the store (plain Python, not part of the graph).
PRICES = {"milk": 3.50, "bread": 2.75, "eggs": 4.20, "apples": 1.10}
COUPONS = {"SAVE10": 0.10, "HALFOFF": 0.50}   # code -> discount fraction
CARD_LIMIT = 20.00                             # pretend the card has this limit


# Step 2: The state, the "basket" every node can read.
class Basket(TypedDict):
    items: list[str]
    coupon: str
    subtotal: float
    discount: float
    total: float
    status: str


# Step 3: Node 1, scan_items. Adds up the basket.
def scan_items(state: Basket) -> dict:
    """Sync node: pure calculation, sums item prices."""
    # TODO: Sum PRICES[item] for every item in state["items"].
    # TODO: Return ONLY {"subtotal": <rounded>, "status": "scanned"}.
    return {}


# Step 4: Node 2, validate_coupon. Async, because a real coupon check would
# call a promotions service. asyncio.sleep stands in for that call.
async def validate_coupon(state: Basket) -> dict:
    """Async node: simulates an I/O-bound coupon lookup."""
    await asyncio.sleep(0.1)
    # TODO: If state["coupon"] is in COUPONS, discount = subtotal * that fraction.
    # TODO: Otherwise discount = 0.0.
    # TODO: Return ONLY {"discount": <rounded>}.
    return {}


# Step 5: Node 3, charge_card. Reads subtotal and discount from earlier nodes.
def charge_card(state: Basket) -> dict:
    """Sync node: computes the total and tries to charge it."""
    # TODO: total = subtotal - discount.
    # TODO: If total > CARD_LIMIT, return {"total": total, "status": "declined"}.
    # TODO: Otherwise return {"total": total, "status": "paid"}.
    return {}


# Step 6: Node 4, print_receipt. Overwrites status (last writer wins).
def print_receipt(state: Basket) -> dict:
    """Sync node: final message, only touches status."""
    # TODO: If paid, return a status like "receipt: $<total> (saved $<discount>)".
    # TODO: If declined, return {"status": "please use another card"}.
    return {}


# Step 7-10: Build, wire and compile the graph (done for you).
builder = StateGraph(Basket)
builder.add_node("scan_items", scan_items)
builder.add_node("validate_coupon", validate_coupon)
builder.add_node("charge_card", charge_card)
builder.add_node("print_receipt", print_receipt)

builder.add_edge(START, "scan_items")
builder.add_edge("scan_items", "validate_coupon")
builder.add_edge("validate_coupon", "charge_card")
builder.add_edge("charge_card", "print_receipt")
builder.add_edge("print_receipt", END)

graph = builder.compile()


# Step 11: Run one checkout and show each node's partial update.
async def run_checkout(items: list[str], coupon: str) -> None:
    initial = {
        "items": items,
        "coupon": coupon,
        "subtotal": 0.0,
        "discount": 0.0,
        "total": 0.0,
        "status": "new",
    }
    print(f"\n--- Checkout: {items} coupon={coupon or 'none'} ---")
    async for step in graph.astream(initial, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"{node_name:>16} -> {update}")
    final = await graph.ainvoke(initial)
    print(f"{'final state':>16} -> {final}")


# Step 12: A small basket with a coupon, and a big one that gets declined.
async def main() -> None:
    await run_checkout(["milk", "bread", "apples"], "SAVE10")
    await run_checkout(["eggs"] * 6, "")
    # TODO: Re-run the big basket with "HALFOFF". Does it get paid now?


asyncio.run(main())
