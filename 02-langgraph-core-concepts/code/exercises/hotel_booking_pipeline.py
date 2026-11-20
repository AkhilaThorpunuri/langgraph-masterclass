"""Hotel booking pipeline (exercise): four nodes, one job each, partial updates only.

Fill in every TODO. The graph wiring and runner are already done, so the file
runs as-is: until you implement the nodes they return {} and nothing changes.
"""
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
    # TODO: price = NIGHTLY_RATE[room_type] * nights * (1 + TAX_RATE)
    # TODO: Return ONLY {"price": <rounded>, "status": "quoted"}.
    return {}


# Step 4: Node 2, check_vacancy. Async, because a real availability check
# would call the property-management system. asyncio.sleep stands in for it.
async def check_vacancy(state: Reservation) -> dict:
    """Async node: simulates an I/O-bound availability lookup."""
    await asyncio.sleep(0.1)
    # TODO: vacancy is True if ROOMS_FREE[room_type] > 0.
    # TODO: Return ONLY {"vacancy": <bool>}.
    return {}


# Step 5: Node 3, assign_room. Reads vacancy, written by the previous node.
def assign_room(state: Reservation) -> dict:
    """Sync node: assigns a room if one is free."""
    # TODO: If no vacancy, return {"status": "fully booked"} (room_number stays empty).
    # TODO: Otherwise make a room number like "S-101" (first letter of room_type,
    #       uppercased) and return {"room_number": ..., "status": "assigned"}.
    return {}


# Step 6: Node 4, send_confirmation. Overwrites status (last writer wins) and
# reads guest and price, which were set much earlier.
def send_confirmation(state: Reservation) -> dict:
    """Sync node: final hand-off, only touches status."""
    # TODO: If status is "assigned", return a status like
    #       "confirmed for <guest>: room <room_number>, $<price>".
    # TODO: Otherwise return {"status": "waitlisted <guest>"}.
    return {}


# Step 7-10: Build, wire and compile the graph (done for you).
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


# Step 12: A suite that is free, and a double that is fully booked.
async def main() -> None:
    await run_reservation("Priya", "suite", 2)
    await run_reservation("Sam", "double", 3)
    # TODO: Add a "single" booking of your own and predict the final state.


asyncio.run(main())
