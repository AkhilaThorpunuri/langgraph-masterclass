"""Hotel booking pipeline (solution): four nodes, one job each, partial updates only."""
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# Step 1: Reference data for the hotel (plain Python, not part of the graph).
NIGHTLY_RATE = {"single": 80.00, "double": 120.00, "suite": 250.00}
ROOMS_FREE = {"single": 3, "double": 0, "suite": 1}
TAX_RATE = 0.12


# Step 2: The state, the "reservation" every node can read.
class Reservation(TypedDict):
    guest: str
    room_type: str
    nights: int
    price: float
    vacancy: bool
    room_number: str
    status: str


# Step 3: Node 1, quote_stay. Prices the stay including tax.
def quote_stay(state: Reservation) -> dict:
    """Sync node: pure calculation, prices the stay."""
    price = NIGHTLY_RATE[state["room_type"]] * state["nights"] * (1 + TAX_RATE)
    return {"price": round(price, 2), "status": "quoted"}


# Step 4: Node 2, check_vacancy. Async, because a real availability check
# would call the property-management system. asyncio.sleep stands in for it.
async def check_vacancy(state: Reservation) -> dict:
    """Async node: simulates an I/O-bound availability lookup."""
    await asyncio.sleep(0.1)
    return {"vacancy": ROOMS_FREE[state["room_type"]] > 0}


# Step 5: Node 3, assign_room. Reads vacancy, written by the previous node.
def assign_room(state: Reservation) -> dict:
    """Sync node: assigns a room if one is free."""
    if not state["vacancy"]:
        return {"status": "fully booked"}
    room_number = f"{state['room_type'][0].upper()}-101"
    return {"room_number": room_number, "status": "assigned"}


# Step 6: Node 4, send_confirmation. Overwrites status (last writer wins) and
# reads guest and price, which were set much earlier.
def send_confirmation(state: Reservation) -> dict:
    """Sync node: final hand-off, only touches status."""
    if state["status"] != "assigned":
        return {"status": f"waitlisted {state['guest']}"}
    return {
        "status": (
            f"confirmed for {state['guest']}: room {state['room_number']}, "
            f"${state['price']:.2f}"
        )
    }


# Step 7-10: Build, wire and compile the graph.
builder = StateGraph(Reservation)
builder.add_node("quote_stay", quote_stay)
builder.add_node("check_vacancy", check_vacancy)
builder.add_node("assign_room", assign_room)
builder.add_node("send_confirmation", send_confirmation)

builder.add_edge(START, "quote_stay")
builder.add_edge("quote_stay", "check_vacancy")
builder.add_edge("check_vacancy", "assign_room")
builder.add_edge("assign_room", "send_confirmation")
builder.add_edge("send_confirmation", END)

graph = builder.compile()


# Step 11: Run one reservation and show each node's partial update.
async def run_reservation(guest: str, room_type: str, nights: int) -> None:
    initial = {
        "guest": guest,
        "room_type": room_type,
        "nights": nights,
        "price": 0.0,
        "vacancy": False,
        "room_number": "",
        "status": "new",
    }
    print(f"\n--- Reservation: {guest}, {nights} nights in a {room_type} ---")
    async for step in graph.astream(initial, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"{node_name:>17} -> {update}")
    final = await graph.ainvoke(initial)
    print(f"{'final state':>17} -> {final}")


# Step 12: A suite that is free, a double that is fully booked, and a single.
async def main() -> None:
    await run_reservation("Priya", "suite", 2)
    await run_reservation("Sam", "double", 3)
    await run_reservation("Lee", "single", 1)


asyncio.run(main())
