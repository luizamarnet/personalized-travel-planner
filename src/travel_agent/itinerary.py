"""Generate personalized daily itineraries using Azure AI Foundry."""

from __future__ import annotations

from datetime import time
import json
import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from travel_agent.schemas import (
    Itinerary,
    ItineraryItemType,
    ItineraryPlanningInput,
)

logger = logging.getLogger(__name__)


class ItineraryGenerationError(RuntimeError):
    """Raised when itinerary generation fails."""


class _ItineraryReview(BaseModel):
    """Compact structured response returned by the itinerary reviewer."""

    is_valid: bool = Field(
        description="Whether the itinerary satisfies the planning requirements."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems that must be fixed before accepting the itinerary.",
    )


MAX_GENERATION_ATTEMPTS = 3


def _build_chat_model() -> ChatOpenAI:
    """Create the DeepSeek chat model."""

   
    return ChatOpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model=os.getenv(
            "DEEPSEEK_MODEL",
            "deepseek-flash",
        ),
        temperature=0,
        max_retries=3,
        timeout=120,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },

    )


def format_planning_input(
    planning_input: ItineraryPlanningInput,
) -> str:
    """Serialize relevant planning data into JSON."""

    data = planning_input.model_dump(
        mode="json",
        exclude={
            "activities": {
                "__all__": {
                    "image_url",
                    "image_path",
                }
            }
        },
    )

    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )


def build_itinerary_prompt() -> str:
    """Build instructions for detailed itinerary generation."""

    return """
You are an experienced travel planner.

Your task is to create a detailed, realistic and personalized
travel itinerary using the planning data provided by the user.

This stage is responsible for ITINERARY PLANNING, not web research.

You do not have access to web search or a routing service in this
stage. Use only the information supplied in the planning data.

Do not pretend that information has been researched or verified.

==================================================
1. GENERAL PLANNING RULES
==================================================

1. Create a complete itinerary covering every date from the
   arrival date through the departure date, including both dates.

2. Use the destination's local timezone.

3. Respect arrival and departure dates, times and locations.

4. Respect the user's:
   - Activity preferences.
   - Interests.
   - Travel pace.
   - If they are traveling with children or not.
   - Accessibility requirements.
   - Maximum daily walking time.
   - Preferred transportation.
   - Meal preferences and restrictions.
   - Accommodation arrangements.
   - Budget.

5. Do not force every activity into the itinerary.

6. Prioritize activities according to the user's preferences:
   - Absolutely!: highest priority.
   - Sounds good: medium priority.
   - Maybe: lowest priority.

7. Use unplanned_activity_ids for activities that cannot
   reasonably fit into the itinerary.

8. Do not invent new attractions in this planning stage.
   Additional activities can be researched separately.

==================================================
2. DETAILED DAILY SCHEDULE
==================================================

The itinerary must be a time-based schedule.

Do not return a simple list of attractions or a short
description such as "Day 1: Visit Sugarloaf Mountain".

Each day must contain individual, chronological schedule items.

For a normal full sightseeing day, aim for approximately
5 to 9 schedule items, depending on the user's travel pace
and the available time.

A schedule item represents exactly one activity or block.

Do not combine multiple attractions into a single item.

Include separate items for:

- Breakfast, when appropriate.
- Each individual attraction.
- Transportation between locations, when supported.
- Lunch.
- Coffee breaks.
- Rest periods.
- Dinner, when appropriate.
- Free time.
- Return to accommodation, when relevant.

Every schedule item must contain:

- A specific title.
- A useful and informative description.
- start_time.
- end_time.
- duration_minutes, consistent with the time interval.
- Location, whenever known.

Avoid generic descriptions such as:
"Visit the attraction and enjoy the experience."

Explain what the traveler should actually do.

ITEM TYPE CLASSIFICATION:

Use ACTIVITY only for real, researched activities included in
the provided activities list.

Use FREE_TIME for:
- Free mornings or afternoons.
- Unscheduled time.
- Relaxing at the accommodation.
- Revisiting previously visited places.
- Exploring the destination without a specific researched attraction.
- Flexible time for the user to decide what to do.

IMPORTANT:
- Every ACTIVITY item MUST have an activity_id matching one of
  the provided activities.
- Never create an ACTIVITY without a valid activity_id.
- FREE_TIME items MUST NOT have an activity_id.
- Do not invent activities to fill empty periods.
- A suggestion such as "Free morning for revisiting favorites"
  must have item_type FREE_TIME, not ACTIVITY.

==================================================
3. ACTIVITY DETAILS
==================================================

For each attraction:

1. Create a separate ACTIVITY item.
2. Reference the original activity using its exact activity_id.
3. Use the activity's existing description and special notes.
4. Include useful highlights describing what to see or do.
5. Respect estimated_duration_hours when provided.
6. Use the original location, address and coordinates when available.
7. Preserve the original source_url.
8. Preserve booking_url when available.

Do not invent:
- Attraction addresses.
- Opening hours.
- Admission prices.
- Booking requirements.
- Specific facilities or experiences not supported
  by the supplied information.

When the recommended visit duration is unavailable, use a
conservative planning estimate only when necessary to create
a schedule. Clearly identify it as an unverified estimate
in the item's notes and the itinerary warnings.

IMPORTANT: ACTIVITY UNIQUENESS

1. Every selected activity must be scheduled at most once
   during the entire trip.

2. Never schedule the same activity on different days.

3. Never create multiple schedule items referencing the
   same activity_id.

4. Each activity_id must appear in exactly one of these places:
   - A scheduled ACTIVITY item.
   - unplanned_activity_ids.

5. Before returning the itinerary, verify that no activity_id
   appears more than once.
   
==================================================
4. TIME MANAGEMENT
==================================================

1. Schedule activities in realistic time slots.

2. Use reasonable start and end times.

3. Avoid overlapping items.

4. Include sufficient time for meals and breaks.

5. Do not overload the user's days.

6. Respect the estimated duration of each attraction.

7. Consider the user's travel pace:
   - relaxed: fewer activities and longer breaks.
   - balanced: a reasonable mix of activities and breaks.
   - intensive: more activities, while remaining realistic.

8. Do not schedule sightseeing before the arrival time
   on the arrival date.

9. Do not schedule activities that conflict with the
   departure time on the departure date.

10. Do not assume that arrival or departure transportation
    has a known duration when it was not provided.

11. When arrival or departure details are incomplete,
    leave sufficient flexibility and explain the limitation
    in warnings.

12. If the available information does not support a full day
    of activities, include free time rather than inventing plans.

==================================================
5. GEOGRAPHICAL ORGANIZATION
==================================================

1. Organize activities in a geographically sensible order
   using the supplied location and address information.

2. Avoid unnecessary backtracking.

3. Group nearby attractions when their proximity is supported
   by the available information.

4. Do not assume that two attractions are close merely because
   their names mention the same neighborhood.

5. Do not invent distances, routes or travel durations.

6. Only create a TRANSPORT item with a numerical duration
   when that duration is supported by supplied information.

7. When route information is missing, add a warning explaining
   that transportation times need verification.

8. When creating a TRANSPORT item, specify:
   - transport_from.
   - transport_to.
   - transport_mode, when known.
   - travel_duration_minutes, when known.

==================================================
6. MEALS AND RESTAURANTS
==================================================

Schedule meals according to the user's meals_per_day preference.

Use sensible local meal times.

Meals must be represented as MEAL items and must contain
MealDetails.

Choose meal locations according to this priority:

1. Near the previous activity.
2. Near the next activity.
3. Between activities, when geographically convenient.
4. Near the accommodation, when appropriate.

For each meal:

- Select the appropriate location_strategy.
- Reference an existing activity through anchor_activity_id
  whenever possible.
- Use search_area only when supported by known location data.

Do not invent restaurant names, addresses, ratings, prices,
opening hours or URLs.

Always leave MealDetails.restaurant as null, unless a activity can also be a meal location.
Activities such as "street food", "food market", should be echaduled to fit an appropriate eatting time.

Actual restaurant research will be performed in a separate stage.

Respect dietary restrictions and allergies when planning
the meal requirements.

==================================================
7. ACCOMMODATION
==================================================

Consider the user's accommodation preferences and booking status.

Include accommodation-related schedule items when relevant,
such as check-in, check-out or return to accommodation.

Do not invent accommodation names, addresses or prices.

Do not assume that the accommodation has already been booked
unless the user explicitly indicated this.

==================================================
8. BUDGET
==================================================

1. Respect the total and category budgets.
2. Use the user's selected currency.
3. Do not invent prices.
4. Distinguish per-person costs from group costs.
5. Use numerical costs only when supported by supplied data.
6. Explain when the budget cannot be reliably evaluated.

==================================================
9. WALKING AND ACCESSIBILITY
==================================================

1. Respect the maximum daily walking time, when provided.
2. Consider the walking requirements of the activities.
3. Include walking between locations only when known.
4. Do not count the full duration of public transportation
   as walking time.
5. Do not invent walking durations.
6. Use null when walking information is unavailable.
7. Add warnings when accessibility or walking feasibility
   cannot be confirmed.

==================================================
10. OUTPUT REQUIREMENTS
==================================================

Return one complete Itinerary object matching the supplied schema.

The result must contain:

- The exact destination.
- The exact destination timezone.
- The correct start and end dates.
- One DayPlan for every date of the trip.
- Multiple individual schedule items on full sightseeing days,
  when the available time supports them.
- Chronologically ordered schedule items.
- Specific start and end times for every item.
- Duration in minutes for every item.
- Detailed activity descriptions.
- Highlights for attractions.
- MealDetails for meals.
- Transportation details whenever supported.
- Unplanned activity IDs.
- Warnings for missing or unverified information.

Every selected activity must either:
- Appear as a scheduled ACTIVITY item, or
- Appear in unplanned_activity_ids.

Do not repeat an activity in multiple schedule items.

The result must be a practical travel schedule that the user
could follow, not merely a collection of recommendations.
"""


def _validate_generated_itinerary(
    itinerary: Itinerary,
    planning_input: ItineraryPlanningInput,
) -> Itinerary:
    """Validate and enrich the generated itinerary."""

    preferences = planning_input.preferences

    # Reject mismatches before restoring canonical values.
    if itinerary.destination != planning_input.destination:
        raise ValueError(
            "The generated itinerary has an unexpected destination."
        )

    if itinerary.timezone != planning_input.timezone:
        raise ValueError(
            "The generated itinerary has an unexpected timezone."
        )

    if itinerary.start_date != preferences.arrival_date:
        raise ValueError(
            "The generated itinerary has an unexpected start date."
        )

    if itinerary.end_date != preferences.departure_date:
        raise ValueError(
            "The generated itinerary has an unexpected end date."
        )

    itinerary.destination = planning_input.destination
    itinerary.timezone = planning_input.timezone

    activities_by_id = {
        activity.id: activity
        for activity in planning_input.activities
    }

    scheduled_activity_ids: list[str] = []
    seen_activity_ids: set[str] = set()


    # Validate activity references and enrich items with
    # information already available in the original activity data.
    for day in itinerary.days:

        valid_items = []

        for item in day.items:

            last_item_end_time = time(0, 0)

            if item.start_time > item.end_time:
                raise ValueError(
                    f"{item.title} schaduled with with end time before start time."
                )

            if item.start_time < last_item_end_time:
                raise ValueError(
                    f"{item.title} starts before last activity ends."
                )

            if item.item_type == ItineraryItemType.ACTIVITY:

                if not item.activity_id:
                    raise ValueError(
                        f"Activity '{item.title}' has no activity_id."
                    )

                activity = activities_by_id.get(item.activity_id)

                if activity is None:
                    raise ValueError(
                        f"Unknown activity_id: {item.activity_id}"
                    )

                # Remove duplicate activities instead of failing
                # the entire itinerary generation.
                if item.activity_id in seen_activity_ids:
                    logger.warning(
                        "Removing duplicated activity: %s",
                        activity.name,
                    )
                    itinerary.warnings.append(
                        f"Activity '{activity.name}' appeared more "
                        "than once in the generated schedule. "
                        "The duplicate occurrence was removed."
                    )
                    continue

                seen_activity_ids.add(item.activity_id)
                scheduled_activity_ids.append(item.activity_id)

                # Reuse information from the original activity.
                item.source_url = (
                    item.source_url or activity.source_url
                )

                item.booking_url = (
                    item.booking_url or activity.booking_url
                )

                item.location = (
                    item.location
                    or activity.address
                    or activity.location
                )

                item.latitude = (
                    item.latitude
                    if item.latitude is not None
                    else activity.latitude
                )

                item.longitude = (
                    item.longitude
                    if item.longitude is not None
                    else activity.longitude
                )

            elif item.activity_id is not None:
                raise ValueError(
                    f"Only ACTIVITY items can reference activity_id. "
                    f"Invalid reference in '{item.title}'."
                )

            valid_items.append(item)

            last_item_end_time = item.end_time

        day.items = valid_items

    # Validate unplanned activity references.
    known_activity_ids = set(activities_by_id)
    unplanned_ids = itinerary.unplanned_activity_ids

    if len(unplanned_ids) != len(set(unplanned_ids)):
        raise ValueError(
            "unplanned_activity_ids contains duplicate IDs."
        )

    unknown_unplanned_ids = set(unplanned_ids) - known_activity_ids

    if unknown_unplanned_ids:
        raise ValueError(
            "Unknown IDs in unplanned_activity_ids: "
            + ", ".join(sorted(unknown_unplanned_ids))
        )

    scheduled_ids = set(scheduled_activity_ids)
    unplanned_set = set(unplanned_ids)

    # Ensure that every selected activity is accounted for.
    unaccounted_ids = known_activity_ids - scheduled_ids - unplanned_set

    if unaccounted_ids:
        missing_ids = [
            activity.id
            for activity in planning_input.activities
            if activity.id in unaccounted_ids
        ]

        itinerary.unplanned_activity_ids.extend(missing_ids)

        missing_names = [
            activities_by_id[activity_id].name
            for activity_id in missing_ids
        ]

        itinerary.warnings.append(
            "The following activities were not explicitly classified "
            "by the itinerary model and were added to the unplanned "
            "activities list: "
            + ", ".join(missing_names)
        )

    # Explicitly identify information that still needs research.
    has_unresearched_meals = any(
        item.item_type == ItineraryItemType.MEAL
        and item.meal_details
        and item.meal_details.restaurant is None
        for day in itinerary.days
        for item in day.items
    )

    if has_unresearched_meals:
        itinerary.warnings.append(
            "Restaurant names, addresses, prices and opening hours "
            "have not yet been researched."
        )

    logger.info(
        "Itinerary validation completed: %d scheduled activities, "
        "%d unplanned activities.",
        len(scheduled_activity_ids),
        len(itinerary.unplanned_activity_ids),
    )

    return itinerary


def _extract_json_object(content: object) -> dict:
    """Extract a JSON object from a plain-text model response."""

    if isinstance(content, list):
        content = "\n".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The itinerary model returned an empty response.")

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    object_start = text.find("{")
    if object_start < 0:
        raise ValueError("The model response does not contain a JSON object.")

    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[object_start:])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The generated JSON is incomplete or malformed: "
            f"{exc.msg} at character {exc.pos}."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("The generated JSON must be an object.")
    return parsed


def _review_itinerary(
    model: ChatOpenAI,
    itinerary: Itinerary,
    planning_input: ItineraryPlanningInput,
) -> _ItineraryReview:
    """Ask the model to check semantic consistency without rewriting the itinerary."""

    reviewer = model.with_structured_output(
        _ItineraryReview,
        method="function_calling",
    )

    review_prompt = """
You are a strict itinerary quality reviewer.
Compare the proposed itinerary with the original planning data.
Do not rewrite the itinerary. Return only the review result.

Mark is_valid=false for material issues, including:
- Activity IDs that do not exactly match supplied activity IDs.
- Duplicate scheduled activities or activities neither scheduled nor unplanned.
- Destination, timezone, or dates that conflict with the input.
- Schedule conflicts with arrival/departure times or user preferences.
- Unsupported invented facts: hotels, addresses, prices, opening hours,
  routes, travel durations, or restaurant details.
- Meals without MealDetails, or fabricated restaurant information.
- Material conflicts with accessibility, walking limits, or budget.

Do not reject correctly unknown/null optional information or warnings
that explain what still needs research.

If valid, return is_valid=true and issues=[].
If invalid, return concise, actionable corrections in issues.
"""

    messages = [
        SystemMessage(content=review_prompt),
        HumanMessage(
            content=(
                "ORIGINAL PLANNING DATA:\n"
                f"{format_planning_input(planning_input)}\n\n"
                "PROPOSED ITINERARY:\n"
                f"{itinerary.model_dump_json(indent=2)}"
            )
        ),
    ]

    review = reviewer.invoke(messages)
    if not isinstance(review, _ItineraryReview):
        review = _ItineraryReview.model_validate(review)
    return review


def _build_generation_messages(
    planning_input: ItineraryPlanningInput,
    previous_issues: list[str] | None = None,
) -> list:
    """Build generation messages, including feedback from a failed attempt."""

    schema_json = json.dumps(
        Itinerary.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    user_content = (
        "Create a detailed, time-based travel itinerary using the following "
        "planning data.\n\n"
        "Return ONLY one complete, valid JSON object matching the Itinerary "
        "schema. Do not use Markdown fences or add text before or after the JSON.\n"
        "Keep the output concise enough to fit in one response: use short "
        "descriptions (1-2 sentences), at most 3 highlights per activity, "
        "and avoid repetitive free-time or transport items. Do not repeat "
        "information already present in the input.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        f"{schema_json}\n\n"
        f"{format_planning_input(planning_input)}"
    )

    if previous_issues:
        user_content += (
            "\n\nThe previous attempt was rejected. Correct ALL of these "
            "specific issues in the new itinerary:\n- "
            + "\n- ".join(previous_issues)
        )

    return [
        SystemMessage(content=build_itinerary_prompt()),
        HumanMessage(content=user_content),
    ]


def generate_itinerary(
    planning_input: ItineraryPlanningInput,
) -> Itinerary:
    """Generate, validate and retry an itinerary until it passes all checks."""

    logger.info(
        "Generating itinerary for %s",
        planning_input.destination,
    )

    try:
        model = _build_chat_model()
        previous_issues: list[str] = []

        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            logger.info(
                "Itinerary generation attempt %d/%d for %s",
                attempt,
                MAX_GENERATION_ATTEMPTS,
                planning_input.destination,
            )

            try:
                # 1. Generate the itinerary as plain text, not function calling.
                generation_response = model.invoke(
                    _build_generation_messages(
                        planning_input=planning_input,
                        previous_issues=previous_issues or None,
                    )
                )

                raw_content = generation_response.content
                itinerary_data = _extract_json_object(raw_content)

                # 2. Validate the complete structure locally with Pydantic.
                itinerary = Itinerary.model_validate(itinerary_data)

                # Restore canonical input values and enrich from source activities.
                itinerary = _validate_generated_itinerary(
                    itinerary=itinerary,
                    planning_input=planning_input,
                )

                # 3. Run a separate, compact structured semantic review.
                review = _review_itinerary(
                    model=model,
                    itinerary=itinerary,
                    planning_input=planning_input,
                )

                if not review.is_valid:
                    previous_issues = review.issues or [
                        ("The itinerary did not pass the semantic review. "
                        "Check it against all planning data and constraints.")
                    ]
                    logger.warning(
                        "Itinerary attempt %d rejected: %s",
                        attempt,
                        "; ".join(previous_issues),
                    )
                    continue

                logger.info(
                    "Generated and validated itinerary with %d days and "
                    "%d schedule items on attempt %d.",
                    len(itinerary.days),
                    sum(len(day.items) for day in itinerary.days),
                    attempt,
                )
                return itinerary

            except Exception as attempt_error:
                previous_issues = [
                    f"Attempt {attempt} failed validation: {attempt_error}"
                ]
                logger.warning(
                    "Itinerary attempt %d/%d failed: %s",
                    attempt,
                    MAX_GENERATION_ATTEMPTS,
                    attempt_error,
                    exc_info=True,
                )

        raise ItineraryGenerationError(
            f"Could not generate a valid itinerary after "
            f"{MAX_GENERATION_ATTEMPTS} attempts. "
            f"Last issues: {'; '.join(previous_issues)}"
        )

    except ItineraryGenerationError:
        logger.exception(
            "Failed to generate itinerary for %s",
            planning_input.destination,
        )
        raise

    except Exception as exc:
        logger.exception(
            "Failed to generate itinerary for %s",
            planning_input.destination,
        )
        raise ItineraryGenerationError(
            f"Could not generate itinerary: {exc}"
        ) from exc
