"""Library checkout pipeline (solution): four nodes, one job each, partial updates only."""
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# Step 1: Reference data for the library (plain Python, not part of the graph).
MEMBERS = {"M001": "alice", "M002": "bob"}          # member_id -> name
LOAN_DAYS = {"book": 21, "dvd": 7, "magazine": 3}    # item type -> loan length
COPIES_ON_SHELF = {"Dune": 2, "Inception": 0, "Wired": 4}
ITEM_TYPE = {"Dune": "book", "Inception": "dvd", "Wired": "magazine"}


# Step 2: The state, the "checkout slip" every node can read.
class Checkout(TypedDict):
    member_id: str
    title: str
    member_name: str
    available: bool
    due_in_days: int
    status: str


# Step 3: Node 1, lookup_member. Finds who is borrowing.
def lookup_member(state: Checkout) -> dict:
    """Sync node: pure lookup, identifies the member."""
    name = MEMBERS.get(state["member_id"])
    if name is None:
        return {"status": "unknown member"}
    return {"member_name": name, "status": "member found"}


# Step 4: Node 2, check_shelf. Async, because a real catalogue lookup would
# hit a database. asyncio.sleep stands in for that call.
async def check_shelf(state: Checkout) -> dict:
    """Async node: simulates an I/O-bound catalogue lookup."""
    await asyncio.sleep(0.1)
    return {"available": COPIES_ON_SHELF.get(state["title"], 0) > 0}


# Step 5: Node 3, check_out. Reads available and member_name from earlier nodes.
def check_out(state: Checkout) -> dict:
    """Sync node: lends the item if possible."""
    if not state["member_name"]:
        return {"status": "rejected: not a member"}
    if not state["available"]:
        return {"status": "on hold list"}
    days = LOAN_DAYS[ITEM_TYPE[state["title"]]]
    return {"due_in_days": days, "status": "checked out"}


# Step 6: Node 4, print_slip. Overwrites status (last writer wins).
def print_slip(state: Checkout) -> dict:
    """Sync node: final message for the member, only touches status."""
    if state["status"] != "checked out":
        return {}  # An empty update is valid: nothing changes.
    return {
        "status": (
            f"{state['member_name']}: return {state['title']} "
            f"in {state['due_in_days']} days"
        )
    }


# Step 7-10: Build, wire and compile the graph.
builder = StateGraph(Checkout)
builder.add_node("lookup_member", lookup_member)
builder.add_node("check_shelf", check_shelf)
builder.add_node("check_out", check_out)
builder.add_node("print_slip", print_slip)

builder.add_edge(START, "lookup_member")
builder.add_edge("lookup_member", "check_shelf")
builder.add_edge("check_shelf", "check_out")
builder.add_edge("check_out", "print_slip")
builder.add_edge("print_slip", END)

graph = builder.compile()


# Step 11: Run one checkout and show each node's partial update.
async def run_checkout(member_id: str, title: str) -> None:
    initial = {
        "member_id": member_id,
        "title": title,
        "member_name": "",
        "available": False,
        "due_in_days": 0,
        "status": "new",
    }
    print(f"\n--- Checkout: {member_id} wants {title} ---")
    async for step in graph.astream(initial, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"{node_name:>16} -> {update}")
    final = await graph.ainvoke(initial)
    print(f"{'final state':>16} -> {final}")


# Step 12: A successful loan, an out-of-stock DVD, and an unknown member.
async def main() -> None:
    await run_checkout("M001", "Dune")
    await run_checkout("M002", "Inception")
    await run_checkout("M999", "Wired")


asyncio.run(main())
