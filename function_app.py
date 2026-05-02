from __future__ import annotations

import os
from typing import Annotated, Any

import azure.durable_functions as df
import azure.functions as func
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

from codeact_provider import CodeActProvider, register_durable_codeact


# ---------------------------------------------------------------------------
# Sample tools (same shape as the MAF codeact benchmark)
# ---------------------------------------------------------------------------

_USERS: list[dict[str, Any]] = [
    {"id": 1, "name": "Alice", "region": "EU", "tier": "gold"},
    {"id": 2, "name": "Bob", "region": "US", "tier": "silver"},
    {"id": 3, "name": "Charlie", "region": "US", "tier": "gold"},
    {"id": 4, "name": "Diana", "region": "APAC", "tier": "bronze"},
    {"id": 5, "name": "Evan", "region": "EU", "tier": "silver"},
    {"id": 6, "name": "Fiona", "region": "US", "tier": "gold"},
    {"id": 7, "name": "George", "region": "APAC", "tier": "gold"},
    {"id": 8, "name": "Hana", "region": "EU", "tier": "bronze"},
]

_ORDERS: dict[int, list[dict[str, Any]]] = {
    1: [{"product": "Widget", "qty": 3, "unit_price": 9.99}, {"product": "Gadget", "qty": 1, "unit_price": 19.99}],
    2: [{"product": "Widget", "qty": 1, "unit_price": 9.99}],
    3: [{"product": "Gadget", "qty": 2, "unit_price": 19.99}, {"product": "Thingamajig", "qty": 4, "unit_price": 4.50}],
    4: [{"product": "Widget", "qty": 10, "unit_price": 9.99}],
    5: [{"product": "Gadget", "qty": 1, "unit_price": 19.99}],
    6: [{"product": "Widget", "qty": 2, "unit_price": 9.99}, {"product": "Thingamajig", "qty": 5, "unit_price": 4.50}],
    7: [{"product": "Gadget", "qty": 3, "unit_price": 19.99}],
    8: [{"product": "Thingamajig", "qty": 2, "unit_price": 4.50}],
}

_DISCOUNTS: dict[str, float] = {"gold": 0.20, "silver": 0.10, "bronze": 0.05}
_TAX_RATES: dict[str, float] = {"EU": 0.21, "US": 0.08, "APAC": 0.10}


def list_users() -> list[dict[str, Any]]:
    """Return all users as a list of dictionaries.

    Each entry has keys: id (int), name (str), region (str), tier (str).
    """
    return _USERS


def get_orders_for_user(
    user_id: Annotated[int, "The user id whose orders to retrieve."],
) -> list[dict[str, Any]]:
    """Return the user's orders as a list of dictionaries.

    Each entry has keys: product (str), qty (int), unit_price (float).
    """
    return _ORDERS.get(user_id, [])


def get_discount_rate(
    tier: Annotated[str, "The customer tier (gold, silver, or bronze)."],
) -> float:
    """Return the discount rate as a float fraction (e.g. 0.2 for 20%)."""
    return _DISCOUNTS[tier]


def get_tax_rate(
    region: Annotated[str, "The region code (EU, US, or APAC)."],
) -> float:
    """Return the tax rate as a float fraction (e.g. 0.21 for 21%)."""
    return _TAX_RATES[region]


def compute_line_total(
    qty: Annotated[int, "Line item quantity."],
    unit_price: Annotated[float, "Line item unit price."],
    discount_rate: Annotated[float, "Discount rate as a fraction (e.g. 0.2 for 20%)."],
    tax_rate: Annotated[float, "Tax rate as a fraction (e.g. 0.21 for 21%)."],
) -> float:
    """Compute a single order line total.

    Formula: qty * unit_price * (1 - discount_rate) * (1 + tax_rate), rounded to 2 decimals.
    """
    subtotal = qty * unit_price
    discounted = subtotal * (1.0 - discount_rate)
    return round(discounted * (1.0 + tax_rate), 2)


TOOLS = [list_users, get_orders_for_user, get_discount_rate, get_tax_rate, compute_line_total]

INSTRUCTIONS = "You are a careful assistant. Use the provided tools for every lookup and computation."

BENCHMARK_PROMPT = (
    "For every user in our system (there are 8 of them), compute the grand total "
    "of all their orders. "
    "Use the compute_line_total tool for each user's orders, after looking up "
    "the relevant discount and tax rates for that user. "
    "Use the provided tools for EVERY data lookup (users, orders, discount rates, "
    "tax rates) and for EVERY line-total computation via compute_line_total — "
    "do not invent values or hardcode any numbers. "
    "The total per order item should apply the discount first and then the tax "
    "(e.g. total = qty * unit_price * (1-discount) * (1+tax)). "
    "Return one entry per user, sorted by grand_total descending."
)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def get_client() -> FoundryChatClient:
    return FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["FOUNDRY_MODEL"],
        credential=AzureCliCredential(),
    )


# ---------------------------------------------------------------------------
# Function App
# ---------------------------------------------------------------------------

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Register hidden durable infrastructure (orchestrator + activity)
register_durable_codeact(app, tools=TOOLS)


@app.route(route="run", methods=["POST"])
async def run(req: func.HttpRequest):
    prompt = req.get_body().decode().strip() or BENCHMARK_PROMPT
    codeact = CodeActProvider(tools=TOOLS)
    agent = Agent(
        client=get_client(),
        name="CodeActAgent",
        instructions=INSTRUCTIONS,
        context_providers=[codeact],
    )
    result = await agent.run(prompt)
    return func.HttpResponse(result.text or "", mimetype="text/plain")


@app.route(route="run-durable", methods=["POST"])
@app.durable_client_input(client_name="client")
async def run_durable(req: func.HttpRequest, client):
    prompt = req.get_body().decode().strip() or BENCHMARK_PROMPT
    codeact = CodeActProvider(tools=TOOLS, durable=True, durable_client=client)
    agent = Agent(
        client=get_client(),
        name="CodeActAgent",
        instructions=INSTRUCTIONS,
        context_providers=[codeact],
    )
    result = await agent.run(prompt)
    return func.HttpResponse(result.text or "", mimetype="text/plain")
