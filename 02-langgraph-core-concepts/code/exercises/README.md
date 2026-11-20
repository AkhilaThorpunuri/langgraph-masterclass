# Exercises: Building Node Pipelines
> Venkata Bhattaram (c) 2026

## Contents
* How to work on these exercises
* Exercise 1: Pizza Delivery
* Exercise 2: Library Checkout
* Exercise 3: Movie Tickets
* Exercise 4: Grocery Checkout
* Exercise 5: Hotel Booking
* Solutions

## How to work on these exercises
Each exercise follows the same pattern as
[coffee_order_pipeline.py](../coffee_order_pipeline.py): four nodes in a
straight line, where node 2 is **async** (it pretends to call an API).

The state, the reference data, the graph wiring and the runner are already
written for you. **Your job is to fill in the node functions** wherever you
see a `# TODO`.

The files run before you change anything. Each unfinished node returns `{}`
(an empty update), so the state never changes. Fill in one node at a time,
re-run, and watch that node's line in the streamed output change.

```bash
cd 02-langgraph-core-concepts/code/exercises
python pizza_delivery_pipeline.py
```

### Rules that apply to every exercise
1. **Return only the keys you change.** A node returns a partial `dict`, not
   the whole state. LangGraph merges it into the state for you.
2. **Only write keys declared in the state class.** Any other key is an error.
3. **Read what earlier nodes wrote.** Later nodes make decisions using values
   earlier nodes put into `state`.
4. **Last writer wins.** When two nodes write `status`, the later value
   replaces the earlier one.
5. **Do not modify `state` in place** (for example `state["price"] = 5`).
   Return a new dict instead.
6. Round money to 2 decimal places with `round(x, 2)`.

### Check your work
Compare your **final state** line for each run with the expected output
listed under each exercise. The `status` value has to match exactly, so watch
the spacing and the `$` signs.

---

## Exercise 1: Pizza Delivery
**File:** [pizza_delivery_pipeline.py](pizza_delivery_pipeline.py)
**New idea:** one node can update **more than one key** at a time.

`take_order -> check_address (async) -> bake -> deliver`

| Node | Reads | Must return |
|---|---|---|
| `take_order` | `pizza`, `size` | `price` = `MENU[pizza] * SIZE_MULTIPLIER[size]`, `status = "ordered"` |
| `check_address` | `zone`, `price` | Zone in `DELIVERY_ZONES`: `deliverable = True` **and** `price` with the zone's fee added. Otherwise only `deliverable = False`. |
| `bake` | `deliverable` | Not deliverable: `status = "cannot deliver"` (leave `box` alone). Otherwise `box = "<size> <pizza>"`, `status = "baked"`. |
| `deliver` | `status`, `price` | If `status` is not `"baked"`: `status = "cancelled, no charge"`. Otherwise `status = "delivered, charged $<price>"`. |

**Your extra task:** add a third order in `main()` and predict its final state
before you run it.

**Expected `status` at the end:**

| Order | Final `price` | Final `status` |
|---|---|---|
| large pepperoni, downtown | `17.6` | `delivered, charged $17.60` |
| small veggie, countryside | `8.8` | `cancelled, no charge` |

---

## Exercise 2: Library Checkout
**File:** [library_checkout_pipeline.py](library_checkout_pipeline.py)
**New idea:** a node can return `{}` on purpose to leave the state as it is.

`lookup_member -> check_shelf (async) -> check_out -> print_slip`

| Node | Reads | Must return |
|---|---|---|
| `lookup_member` | `member_id` | Found in `MEMBERS`: `member_name`, `status = "member found"`. Otherwise only `status = "unknown member"`. |
| `check_shelf` | `title` | `available` = more than 0 copies in `COPIES_ON_SHELF`. |
| `check_out` | `member_name`, `available`, `title` | Empty `member_name`: `status = "rejected: not a member"`. Not available: `status = "on hold list"`. Otherwise `due_in_days` from `LOAN_DAYS[ITEM_TYPE[title]]` and `status = "checked out"`. |
| `print_slip` | `status`, `member_name`, `title`, `due_in_days` | If checked out: `status = "<member_name>: return <title> in <due_in_days> days"`. Otherwise return `{}`. |

**Expected output:**

| Checkout | Final `status` |
|---|---|
| M001 wants Dune | `alice: return Dune in 21 days` |
| M002 wants Inception | `on hold list` |
| M999 wants Wired | `rejected: not a member` |

**Think about it:** in the M999 run, `check_shelf` still runs even though the
member is unknown. Why? (Hint: these are normal edges, so every node always
runs. Skipping nodes is what **conditional edges** are for, which is the next
lesson.)

---

## Exercise 3: Movie Tickets
**File:** [movie_ticket_pipeline.py](movie_ticket_pipeline.py)
**New idea:** building a value (a ticket code) out of several state keys.

`quote_price -> check_seats (async) -> reserve -> issue_ticket`

| Node | Reads | Must return |
|---|---|---|
| `quote_price` | `showtime`, `seat_type`, `quantity` | `price` = `(TICKET_PRICE[showtime] + SEAT_UPGRADE[seat_type]) * quantity`, `status = "quoted"` |
| `check_seats` | `showtime`, `quantity` | `seats_available` = `SEATS_LEFT[showtime] >= quantity` |
| `reserve` | `seats_available`, `showtime`, `seat_type`, `quantity` | No seats: `status = "sold out"`. Otherwise `ticket_code` = first 3 letters of showtime and seat type in upper case, plus the quantity (e.g. `MAT-REC-2`), and `status = "reserved"`. |
| `issue_ticket` | `status`, `ticket_code`, `price` | If not reserved: `status = "no charge"`. Otherwise `status = "issued <ticket_code>, paid $<price>"`. |

**Your extra task:** book 50 matinee seats. Which node stops the booking?

**Expected output:**

| Booking | Final `status` |
|---|---|
| 2 x recliner, matinee | `issued MAT-REC-2, paid $26.00` |
| 1 x standard, evening | `no charge` |
| 50 x standard, matinee | `no charge` (stopped by `check_seats`) |

---

## Exercise 4: Grocery Checkout
**File:** [grocery_checkout_pipeline.py](grocery_checkout_pipeline.py)
**New idea:** working with a **list** in state, and a value (`total`) that is
calculated from two earlier nodes' output.

`scan_items -> validate_coupon (async) -> charge_card -> print_receipt`

| Node | Reads | Must return |
|---|---|---|
| `scan_items` | `items` | `subtotal` = sum of `PRICES[item]` for every item, `status = "scanned"` |
| `validate_coupon` | `coupon`, `subtotal` | `discount` = `subtotal * COUPONS[coupon]`, or `0.0` if the coupon is unknown or empty |
| `charge_card` | `subtotal`, `discount` | `total = subtotal - discount`. Over `CARD_LIMIT`: `status = "declined"`, otherwise `status = "paid"`. Always return `total`. |
| `print_receipt` | `status`, `total`, `discount` | Paid: `status = "receipt: $<total> (saved $<discount>)"`. Otherwise `status = "please use another card"`. |

**Your extra task:** run the 6-eggs basket again with `"HALFOFF"`. Is it paid now?

**Expected output:**

| Checkout | Final `total` | Final `status` |
|---|---|---|
| milk, bread, apples + `SAVE10` | `6.62` | `receipt: $6.62 (saved $0.73)` |
| 6 x eggs, no coupon | `25.2` | `please use another card` |
| 6 x eggs + `HALFOFF` | `12.6` | `receipt: $12.60 (saved $12.60)` |

---

## Exercise 5: Hotel Booking
**File:** [hotel_booking_pipeline.py](hotel_booking_pipeline.py)
**New idea:** the last node reads values written by the **first** node and by
the initial input.

`quote_stay -> check_vacancy (async) -> assign_room -> send_confirmation`

| Node | Reads | Must return |
|---|---|---|
| `quote_stay` | `room_type`, `nights` | `price` = `NIGHTLY_RATE[room_type] * nights * (1 + TAX_RATE)`, `status = "quoted"` |
| `check_vacancy` | `room_type` | `vacancy` = `ROOMS_FREE[room_type] > 0` |
| `assign_room` | `vacancy`, `room_type` | No vacancy: `status = "fully booked"`. Otherwise `room_number` = first letter of room type in upper case + `"-101"` (e.g. `S-101`), `status = "assigned"`. |
| `send_confirmation` | `status`, `guest`, `room_number`, `price` | Assigned: `status = "confirmed for <guest>: room <room_number>, $<price>"`. Otherwise `status = "waitlisted <guest>"`. |

**Your extra task:** add a booking for a `"single"` room and predict the final
state.

**Expected output:**

| Reservation | Final `price` | Final `status` |
|---|---|---|
| Priya, suite, 2 nights | `560.0` | `confirmed for Priya: room S-101, $560.00` |
| Sam, double, 3 nights | `403.2` | `waitlisted Sam` |

---

## Solutions
Complete, runnable solutions are in the [solutions/](solutions/) folder, with
one file per exercise and the same file names. Try each exercise yourself
first, then compare:

| Exercise | Solution |
|---|---|
| 1. Pizza Delivery | [solutions/pizza_delivery_pipeline.py](solutions/pizza_delivery_pipeline.py) |
| 2. Library Checkout | [solutions/library_checkout_pipeline.py](solutions/library_checkout_pipeline.py) |
| 3. Movie Tickets | [solutions/movie_ticket_pipeline.py](solutions/movie_ticket_pipeline.py) |
| 4. Grocery Checkout | [solutions/grocery_checkout_pipeline.py](solutions/grocery_checkout_pipeline.py) |
| 5. Hotel Booking | [solutions/hotel_booking_pipeline.py](solutions/hotel_booking_pipeline.py) |

The solutions also include the "extra task" runs, so their output has one more
case than the starter files.
