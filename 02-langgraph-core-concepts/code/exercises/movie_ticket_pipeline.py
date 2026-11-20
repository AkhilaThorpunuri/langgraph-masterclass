"""Movie ticket pipeline (exercise): four nodes, one job each, partial updates only.

Fill in every TODO. The graph wiring and runner are already done, so the file
runs as-is: until you implement the nodes they return {} and nothing changes.
"""
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# Step 1: Reference data for the cinema (plain Python, not part of the graph).
TICKET_PRICE = {"matinee": 8.00, "evening": 12.00}
SEAT_UPGRADE = {"standard": 0.00, "recliner": 5.00}
SEATS_LEFT = {"matinee": 40, "evening": 0}


# Step 2: The state, the "booking" every node can read.
class TicketBooking(TypedDict):
    showtime: str
    seat_type: str
    quantity: int
    price: float
    seats_available: bool
    ticket_code: str
    status: str


# Step 3: Node 1, quote_price. Prices the tickets.
def quote_price(state: TicketBooking) -> dict:
    """Sync node: pure calculation, prices the booking."""
    # TODO: price = (TICKET_PRICE[showtime] + SEAT_UPGRADE[seat_type]) * quantity
    # TODO: Return ONLY {"price": <rounded>, "status": "quoted"}.
    return {}


# Step 4: Node 2, check_seats. Async, because a real seat map lives in a
# booking service. asyncio.sleep stands in for that call.
async def check_seats(state: TicketBooking) -> dict:
    """Async node: simulates an I/O-bound seat lookup."""
    await asyncio.sleep(0.1)
    # TODO: seats_available is True if SEATS_LEFT[showtime] >= state["quantity"].
    # TODO: Return ONLY {"seats_available": <bool>}.
    return {}


# Step 5: Node 3, reserve. Reads seats_available, written by the previous node.
def reserve(state: TicketBooking) -> dict:
    """Sync node: holds the seats if any are left."""
    # TODO: If no seats, return {"status": "sold out"} (ticket_code stays empty).
    # TODO: Otherwise build a code like "EVE-REC-2" from showtime/seat_type/quantity
    #       and return {"ticket_code": <code>, "status": "reserved"}.
    return {}


# Step 6: Node 4, issue_ticket. Overwrites status (last writer wins) and reads
# price, which quote_price wrote three steps earlier.
def issue_ticket(state: TicketBooking) -> dict:
    """Sync node: final hand-off, only touches status."""
    # TODO: If status is not "reserved", return {"status": "no charge"}.
    # TODO: Otherwise return {"status": "issued <ticket_code>, paid $<price>"}.
    return {}


# Step 7-10: Build, wire and compile the graph (done for you).
builder = StateGraph(TicketBooking)
builder.add_node("quote_price", quote_price)
builder.add_node("check_seats", check_seats)
builder.add_node("reserve", reserve)
builder.add_node("issue_ticket", issue_ticket)

builder.add_edge(START, "quote_price")
builder.add_edge("quote_price", "check_seats")
builder.add_edge("check_seats", "reserve")
builder.add_edge("reserve", "issue_ticket")
builder.add_edge("issue_ticket", END)

graph = builder.compile()


# Step 11: Run one booking and show each node's partial update.
async def run_booking(showtime: str, seat_type: str, quantity: int) -> None:
    initial = {
        "showtime": showtime,
        "seat_type": seat_type,
        "quantity": quantity,
        "price": 0.0,
        "seats_available": False,
        "ticket_code": "",
        "status": "new",
    }
    print(f"\n--- Booking: {quantity} x {seat_type} for {showtime} ---")
    async for step in graph.astream(initial, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"{node_name:>16} -> {update}")
    final = await graph.ainvoke(initial)
    print(f"{'final state':>16} -> {final}")


# Step 12: A matinee with seats, and a sold-out evening show.
async def main() -> None:
    await run_booking("matinee", "recliner", 2)
    await run_booking("evening", "standard", 1)
    # TODO: Try booking 50 matinee seats. Which node stops it?


asyncio.run(main())
