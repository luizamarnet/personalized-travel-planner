import json
import logging

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

from travel_agent.schemas import (
    Activity,
    ActivityData,
    ActivityPricing,
    ActivityPricingItem,
    ActivityPricingResult,
)



logger = logging.getLogger(__name__)

def create_pricing_agent(llm: ChatOpenAI):
    """Create an agent responsible for extracting activity prices."""

    return create_agent(
        model=llm,
        tools=[],
        system_prompt="""
        You are a travel pricing specialist.

        Your task is to extract structured pricing information
        from the supplied activity descriptions.

        IMPORTANT: All prices must be normalized to a
        per-person basis whenever the source supports it.

        Rules:

        1. Never invent prices or currencies.
        2. Use ISO 4217 currency codes.
        3. Preserve minimum and maximum price ranges.
        4. Distinguish verified prices from estimates.
        5. Never add alternative currency representations.
        6. Different currencies may represent the same price.
        7. Keep alternative currencies in the same PriceOption.
        8. Distinguish alternative currencies from separate costs.
        9. Identify whether prices are per person, per group,
           per ticket, per vehicle or unknown.
        10. Never divide group prices by an assumed group size.
        11. Do not interpret an unknown price as free.
        12. Use status "free" only when the source explicitly
            indicates that the activity is free.
        13. Use null or empty representations when no reliable
            price is available.
        14. Preserve the original activity IDs.
        """,
        response_format=ToolStrategy(
            schema=ActivityPricingResult,
            handle_errors=True,
        ),
    )


def extract_activity_pricing(
    llm: ChatOpenAI,
    activities: list[Activity],
) -> list[Activity]:
    """Extract structured pricing information for activities."""

    if not activities:
        return []

    agent = create_pricing_agent(llm)

    batch_size = 3
    pricing_by_id = {}

    for start in range(0, len(activities), batch_size):
        batch = activities[start:start + batch_size]

        activity_information = [
            {
                "activity_id": activity.id,
                "name": activity.name,
                "estimated_price": activity.estimated_price,
                "price_currency": activity.price_currency,
                "description": activity.description,
            }
            for activity in batch
        ]

        logger.info(
            "Extracting prices for batch %d/%d.",
            start // batch_size + 1,
            (len(activities) + batch_size - 1) // batch_size,
        )

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        activity_information,
                        ensure_ascii=False,
                    ),
                }
            ]
        })

        extracted = result.get("structured_response")

        if extracted is None:
            raise ValueError(
                "Pricing agent returned no structured response."
            )

        batch_ids = {activity.id for activity in batch}

        for item in extracted.activities:
            if item.activity_id not in batch_ids:
                logger.warning(
                    "Ignoring unexpected activity ID: %s",
                    item.activity_id,
                )
                continue

            pricing_by_id[item.activity_id] = item.pricing

    enriched_activities = []

    for activity in activities:
        pricing = pricing_by_id.get(activity.id)

        if pricing is None:
            logger.warning(
                "No pricing extracted for activity %s.",
                activity.id,
            )
            enriched_activities.append(activity)
            continue

        enriched_activities.append(
            activity.model_copy(
                update={"pricing": pricing}
            )
        )

    return enriched_activities