
"""Currency conversion using the Frankfurter exchange rate API."""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

import requests

from travel_agent.schemas import Activity

logger = logging.getLogger(__name__)

FRANKFURTER_API_URL = "https://api.frankfurter.dev/v2"


def _get_exchange_rate(
    source_currency: str,
    target_currency: str,
) -> tuple[Decimal, date]:
    """Retrieve the latest available exchange rate."""

    response = requests.get(
        f"{FRANKFURTER_API_URL}/rate/"
        f"{source_currency.lower()}/{target_currency.lower()}",
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()

    return Decimal(str(data["rate"])), date.fromisoformat(data["date"])


def convert_activity_pricing(
    activities: list[Activity],
    target_currency: str,
) -> list[Activity]:
    """Convert activity prices while preserving their original values."""

    target_currency = target_currency.upper()
    rate_cache: dict[str, tuple[Decimal, date]] = {}

    converted_activities = []

    for activity in activities:
        if activity.pricing is None:
            converted_activities.append(activity)
            continue

        converted_options = []

        for option in activity.pricing.options:
            converted_representations = []

            for representation in option.representations:
                source_currency = representation.currency.upper()

                if (
                    representation.minimum is None
                    and representation.maximum is None
                ):
                    converted_representations.append(
                        representation.model_copy(
                            update={
                                "converted_currency": target_currency,
                                "conversion_status": "unavailable",
                            }
                        )
                    )
                    continue

                if source_currency == target_currency:
                    rate = Decimal("1")
                    rate_date = None
                    status = "same_currency"

                else:
                    try:
                        if source_currency not in rate_cache:
                            rate_cache[source_currency] = _get_exchange_rate(
                                source_currency,
                                target_currency,
                            )

                        rate, rate_date = rate_cache[source_currency]
                        status = "converted"

                    except (
                        requests.RequestException,
                        KeyError,
                        ValueError,
                    ):
                        logger.warning(
                            "Could not convert %s to %s for activity %s.",
                            source_currency,
                            target_currency,
                            activity.id,
                            exc_info=True,
                        )

                        converted_representations.append(
                            representation.model_copy(
                                update={
                                    "converted_currency": target_currency,
                                    "conversion_status": "unavailable",
                                }
                            )
                        )
                        continue

                minimum = (
                    float(Decimal(str(representation.minimum)) * rate)
                    if representation.minimum is not None
                    else None
                )

                maximum = (
                    float(Decimal(str(representation.maximum)) * rate)
                    if representation.maximum is not None
                    else None
                )

                converted_representations.append(
                    representation.model_copy(
                        update={
                            "converted_currency": target_currency,
                            "converted_minimum": minimum,
                            "converted_maximum": maximum,
                            "exchange_rate": float(rate),
                            "exchange_rate_date": (
                                rate_date.isoformat()
                                if rate_date
                                else None
                            ),
                            "conversion_status": status,
                        }
                    )
                )

            converted_options.append(
                option.model_copy(
                    update={
                        "representations": converted_representations,
                    }
                )
            )

        converted_activities.append(
            activity.model_copy(
                update={
                    "pricing": activity.pricing.model_copy(
                        update={"options": converted_options}
                    )
                }
            )
        )

    return converted_activities
