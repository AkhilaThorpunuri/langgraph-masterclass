"""Pizza delivery pipeline (solution): four nodes, one job each, partial updates only."""
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# Step 1: Reference data for the pizza shop (plain Python, not part of the graph).
MENU = {"margherita": 10.00, "pepperoni": 12.00, "veggie": 11.00}
SIZE_MULTIPLIER = {"small": 0.8, "medium": 1.0, "large": 1.3}
DELIVERY_ZONES = {"downtown": 2.00, "suburbs": 4.00}  # zone -> delivery fee


# Step 2: The state, the "delivery ticket" every node can read.
class PizzaOrder(TypedDict):
    pizza: str
    size: str
    zone: str
    price: float
    deliverable: bool
    box: str
    status: str


# Step 3: Node 1, take_order. Prices the pizza.
def take_order(state: PizzaOrder) -> dict:
    """Sync node: pure calculation, prices the pizza."""
    price = MENU[state["pizza"]] * SIZE_MULTIPLIER[state["size"]]
    return {"price": round(price, 2), "status": "ordered"}


# Step 4: Node 2, check_address. Async, because a real address check would
# call a maps/geo API. asyncio.sleep stands in for that call.
async def check_address(state: PizzaOrder) -> dict:
    """Async node: simulates an I/O-bound address lookup."""
    await asyncio.sleep(0.1)
    if state["zone"] not in DELIVERY_ZONES:
        return {"deliverable": False}
    # One node may update several keys at once: here deliverable AND price.
    fee = DELIVERY_ZONES[state["zone"]]
    return {"deliverable": True, "price": round(state["price"] + fee, 2)}


# Step 5: Node 3, bake. Reads deliverable, written by the previous node.
def bake(state: PizzaOrder) -> dict:
    """Sync node: only bakes if the address is deliverable."""
    if not state["deliverable"]:
        return {"status": "cannot deliver"}
    return {"box": f"{state['size']} {state['pizza']}", "status": "baked"}


# Step 6: Node 4, deliver. Overwrites status (last writer wins).
def deliver(state: PizzaOrder) -> dict:
    """Sync node: final hand-off, only touches status."""
    if state["status"] != "baked":
        return {"status": "cancelled, no charge"}
    return {"status": f"delivered, charged ${state['price']:.2f}"}


# Step 7-10: Build, wire and compile the graph.
builder = StateGraph(PizzaOrder)
builder.add_node("take_order", take_order)
builder.add_node("check_address", check_address)
builder.add_node("bake", bake)
builder.add_node("deliver", deliver)

builder.add_edge(START, "take_order")
builder.add_edge("take_order", "check_address")
builder.add_edge("check_address", "bake")
builder.add_edge("bake", "deliver")
builder.add_edge("deliver", END)

graph = builder.compile()


# Step 11: Run one order and show each node's partial update.
async def run_order(pizza: str, size: str, zone: str) -> None:
    initial = {
        "pizza": pizza,
        "size": size,
        "zone": zone,
        "price": 0.0,
        "deliverable": False,
        "box": "",
        "status": "new",
    }
    print(f"\n--- Order: {size} {pizza} to {zone} ---")
    async for step in graph.astream(initial, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"{node_name:>16} -> {update}")
    final = await graph.ainvoke(initial)
    print(f"{'final state':>16} -> {final}")


# Step 12: One deliverable order, one outside the delivery zones, plus a third.
async def main() -> None:
    await run_order("pepperoni", "large", "downtown")
    await run_order("veggie", "small", "countryside")
    await run_order("margherita", "medium", "suburbs")


asyncio.run(main())
