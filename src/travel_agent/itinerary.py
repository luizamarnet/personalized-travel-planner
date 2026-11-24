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

from collections.abc import Iterator
from collections import Counter
from datetime import timedelta

from travel_agent.schemas import (
    DayOutline,
    DayPlan,
    Itinerary,
    ItineraryItemType,
    ItineraryPlanningInput,
    ItineraryOutline,
    Preference,
    _validate_schedule_overlaps,
)
from travel_agent.pricing import extract_activity_pricing

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


def build_planne_outline_prompt() -> str:
    """Build instructions for high-level itinerary outlining."""

    return """
You are an experienced travel planner.

Your task is to distribute the user's selected activities
across the entire trip.

This stage is responsible ONLY for activity allocation.

Do not generate a detailed daily schedule.

Use only the supplied planning data. Do not invent
attractions, routes, distances, or travel durations.

==================================================
1. TRIP ORGANIZATION
==================================================

1. Cover every date from arrival through departure,
   including both dates.

2. Respect the destination timezone.

3. Consider arrival and departure dates and times.

4. Respect the user's:
   - Activity preferences.
   - Interests.
   - Travel pace.
   - Children and accessibility requirements.
   - Maximum daily walking time.
   - Preferred transportation.
   - Budget.

5. Prioritize activities:
   - Absolutely!: highest priority.
   - Sounds good: medium priority.
   - Maybe: lowest priority.

6. Do not force every activity into the trip.

==================================================
2. ACTIVITY DISTRIBUTION
==================================================

1. Assign activities to the most appropriate travel dates.

2. Distribute activities across the entire trip rather
   than filling the first available days.

3. Consider estimated_duration_hours when available.

4. Group geographically compatible activities on the
   same day when their proximity is supported by the data.

5. Avoid unnecessary geographical backtracking.

6. Consider the available time on arrival and departure days.

7. Respect the user's travel pace and avoid overloaded days.

8. Preserve sufficient flexibility for relaxed travel.

9. Use the original activity IDs exactly as supplied.

10. Every activity can be assigned to at most one day.

==================================================
3. UNPLANNED ACTIVITIES
==================================================

Activities that cannot reasonably fit into the trip
must be included in unplanned_activity_ids.

Every supplied activity ID must appear exactly once
in either:
- One day's activity_ids.
- unplanned_activity_ids.

Never invent new activity IDs.

==================================================
4. IMPORTANT RESTRICTIONS
==================================================

Do NOT generate:
- Specific start or end times.
- Detailed daily schedules.
- Meal schedules.
- Restaurant recommendations.
- Transportation schedule items.
- Travel durations.
- Opening hours.
- Admission prices.
- Detailed attraction descriptions.

These responsibilities belong to the subsequent
daily schedule generation stage.

==================================================
5. OUTPUT REQUIREMENTS
==================================================

Return one complete ItineraryOutline object matching
the supplied schema.

Include:
- Exact destination and timezone.
- Correct start and end dates.
- One DayOutline for every travel date.
- Activity IDs in their recommended daily order.
- Unplanned activity IDs.
- Warnings when relevant.

The outline must represent a coherent trip.

Consider the entire trip when distributing activities.
Do not optimize each day independently.
"""



def build_daily_schedule_prompt() -> str:
    """Build instructions for generating one detailed daily schedule."""

    return """
You are an experienced travel planner.

Your task is to generate a detailed, realistic and personalized
schedule for ONE specific day of a previously planned trip.

This stage is responsible for DAILY SCHEDULE GENERATION,
not overall itinerary planning or web research.

The activities have already been selected and distributed
across the trip in a separate planning stage.

You must follow that allocation.

You do not have access to web search or a routing service.
Use only the information supplied in the planning data.

Do not pretend that information has been researched or verified.

==================================================
1. YOUR TASK AND CONTEXT
==================================================

You will receive:

1. PLANNING DATA:
   - Destination and timezone.
   - User preferences and constraints.
   - All available researched activities.
   - Arrival and departure information.
   - Budget, accommodation and transportation preferences.

2. COMPLETE ITINERARY OUTLINE:
   - Activity allocation across all trip dates.
   - Activities assigned to each day.
   - Activities intentionally left unplanned.

3. CURRENT DAY OUTLINE:
   - The exact date being generated.
   - The activities assigned to that date.

4. PREVIOUSLY GENERATED DAYS (when available):
   - Detailed schedules for days already completed.
   - Use them to maintain continuity and avoid repetition.

Your responsibility is to generate a detailed DayPlan
for the CURRENT DAY OUTLINE only.

Do not generate schedules for other dates.

Do not modify the overall activity allocation.

==================================================
2. GENERAL PLANNING RULES
==================================================

1. Generate a schedule for exactly the date specified
   in the current day outline.

2. Use the destination's local timezone.

3. Respect the user's:
   - Activity preferences.
   - Interests.
   - Travel pace.
   - Whether they are traveling with children.
   - Accessibility requirements.
   - Maximum daily walking time.
   - Preferred transportation.
   - Meal preferences and restrictions.
   - Accommodation arrangements.
   - Budget.

4. Respect arrival and departure constraints when generating
   the arrival or departure date.

5. Do not invent new attractions.

6. Do not schedule activities assigned to other dates.

7. Do not move activities between days.

8. Do not force every available activity into the schedule.

9. Follow the priority of the assigned activities:
   - Absolutely!: highest priority.
   - Sounds good: medium priority.
   - Maybe: lowest priority.

10. If an assigned activity cannot reasonably fit into the day,
    explain the issue rather than silently inventing an alternative.

==================================================
3. DETAILED DAILY SCHEDULE
==================================================

The result must be a time-based schedule.

Do not return a simple list of attractions or a short
description such as "Visit Sugarloaf Mountain".

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

Use ACTIVITY only for real, researched activities included
in the provided activities list and assigned to the current day.

Use FREE_TIME for:
- Free mornings or afternoons.
- Unscheduled time.
- Relaxing at the accommodation.
- Revisiting previously visited places.
- Exploring the destination without a specific researched attraction.
- Flexible time for the user to decide what to do.

IMPORTANT:

- Every ACTIVITY item MUST have an activity_id matching an
  activity assigned to the current day.
- Never create an ACTIVITY without a valid activity_id.
- FREE_TIME items MUST NOT have an activity_id.
- Do not invent activities to fill empty periods.

==================================================
4. ACTIVITY DETAILS
==================================================

For each activity assigned to the current day:

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
in the item's notes.

IMPORTANT: ACTIVITY UNIQUENESS

1. Schedule each activity assigned to the current day at most once.
2. Never schedule an activity already included in previous days.
3. Never create multiple schedule items referencing the same activity_id.
4. Do not schedule activities assigned to other dates.
5. Do not change the overall trip allocation.

==================================================
5. TIME MANAGEMENT
==================================================

1. Schedule activities in realistic time slots.
2. Use reasonable start and end times.
3. Avoid overlapping items.
4. Include sufficient time for meals and breaks.
5. Do not overload the user's day.
6. Respect the estimated duration of each attraction.
7. Consider the user's travel pace:
   - relaxed: fewer activities and longer breaks.
   - balanced: a reasonable mix of activities and breaks.
   - intensive: more activities, while remaining realistic.

8. On the arrival date, do not schedule sightseeing before
   the arrival time.

9. On the departure date, do not schedule activities that
   conflict with the departure time.

10. Do not assume arrival or departure transportation
    has a known duration when it was not provided.

11. When arrival or departure details are incomplete,
    leave sufficient flexibility.

12. If the available information does not support a full day
    of activities, include free time rather than inventing plans.

==================================================
6. GEOGRAPHICAL ORGANIZATION
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

7. When route information is missing, do not invent it.
   Add a warning explaining that transportation times
   need verification.

8. When creating a TRANSPORT item, specify:
   - transport_from.
   - transport_to.
   - transport_mode, when known.
   - travel_duration_minutes, when known.

==================================================
7. MEALS AND RESTAURANTS
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

Always leave MealDetails.restaurant as null.

Food-related activities such as street food experiences
or food markets should be scheduled at an appropriate
eating time when possible.

Actual restaurant research will be performed in a separate stage.

Respect dietary restrictions and allergies when planning
the meal requirements.

==================================================
8. ACCOMMODATION
==================================================

Consider the user's accommodation preferences and booking status.

Include accommodation-related schedule items when relevant,
such as check-in, check-out or return to accommodation.

Do not invent accommodation names, addresses or prices.

Do not assume that accommodation has already been booked
unless the user explicitly indicated this.

==================================================
9. BUDGET
==================================================

1. Respect the total and category budgets.
2. Use the user's selected currency.
3. Do not invent prices.
4. Distinguish per-person costs from group costs.
5. Use numerical costs only when supported by supplied data.
6. Explain when the budget cannot be reliably evaluated.

==================================================
10. WALKING AND ACCESSIBILITY
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
11. PREVIOUSLY GENERATED DAYS
==================================================

When previous days are provided:

1. Do not repeat activities already scheduled.
2. Consider the locations visited on previous days.
3. Avoid unnecessary repetition of experiences.
4. Maintain continuity across the trip when relevant.
5. Consider the accommodation and transportation context.
6. Do not modify previously generated schedules.

Previous days are contextual information only.
Your output must contain the CURRENT DAY ONLY.

==================================================
12. OUTPUT REQUIREMENTS
==================================================

Return exactly one DayPlan matching the supplied schema.

The result must contain:

- The exact date specified in the current day outline.
- Individual chronological schedule items.
- Specific start and end times for every item.
- Duration in minutes for every item.
- Detailed activity descriptions.
- Highlights for attractions.
- MealDetails for meals.
- Transportation details whenever supported.
- Daily walking total when it can be reliably calculated.
- Daily estimated cost when it can be reliably calculated.
- Warnings for missing or unverified information, when supported
  by the DayPlan schema.

Every activity assigned to the current day must either:
- Appear as a scheduled ACTIVITY item, or
- Be explicitly identified as not schedulable in the available
  output fields, with the reason explained in the notes or warnings.

Do not include activities assigned to other dates.

Do not return the complete Itinerary object.

The result must be a practical daily schedule that the user
could follow, not merely a collection of recommendations.
"""


def _validate_itinerary_outline(
    outline: ItineraryOutline,
    planning_input: ItineraryPlanningInput,
) -> ItineraryOutline:
    """Validate the activity allocation across the entire trip."""

    preferences = planning_input.preferences
    expected_start_date = preferences.arrival_date
    expected_end_date = preferences.departure_date

    # 1. Validate canonical trip information.
    if outline.destination != planning_input.destination:
        raise ValueError(
            f"Incorrect destination: expected "
            f"'{planning_input.destination}', got '{outline.destination}'."
        )

    if outline.timezone != planning_input.timezone:
        raise ValueError(
            f"Incorrect timezone: expected "
            f"'{planning_input.timezone}', got '{outline.timezone}'."
        )

    if outline.start_date != expected_start_date:
        raise ValueError(
            f"Incorrect start date: expected {expected_start_date}, "
            f"got {outline.start_date}."
        )

    if outline.end_date != expected_end_date:
        raise ValueError(
            f"Incorrect end date: expected {expected_end_date}, "
            f"got {outline.end_date}."
        )

    # 2. Validate that every trip date appears exactly once.
    expected_dates = {
        expected_start_date + timedelta(days=offset)
        for offset in range(
            (expected_end_date - expected_start_date).days + 1
        )
    }

    actual_dates = [day.date for day in outline.days]

    duplicate_dates = [
        day_date
        for day_date, count in Counter(actual_dates).items()
        if count > 1
    ]

    if duplicate_dates:
        raise ValueError(
            f"Duplicate dates in itinerary outline: {duplicate_dates}."
        )

    actual_dates_set = set(actual_dates)

    missing_dates = expected_dates - actual_dates_set
    unexpected_dates = actual_dates_set - expected_dates

    if missing_dates or unexpected_dates:
        raise ValueError(
            f"Invalid trip dates. Missing dates: {sorted(missing_dates)}. "
            f"Unexpected dates: {sorted(unexpected_dates)}."
        )

    # 3. Collect all available activity IDs.
    available_activity_ids = {
        activity.id for activity in planning_input.activities
    }

    # 4. Collect activity IDs assigned to each day.
    assigned_activity_ids = [
        activity_id
        for day in outline.days
        for activity_id in day.activity_ids
    ]

    # 5. Validate that assigned activities exist.
    unknown_assigned_ids = (
        set(assigned_activity_ids) - available_activity_ids
    )

    if unknown_assigned_ids:
        raise ValueError(
            "The outline contains unknown activity IDs: "
            f"{sorted(unknown_assigned_ids)}."
        )

    # 6. Validate activity uniqueness across all days.
    duplicate_assigned_ids = [
        activity_id
        for activity_id, count in Counter(
            assigned_activity_ids
        ).items()
        if count > 1
    ]

    if duplicate_assigned_ids:
        raise ValueError(
            "Activities assigned more than once: "
            f"{duplicate_assigned_ids}."
        )

    # 7. Validate unplanned activity IDs.
    unplanned_ids = outline.unplanned_activity_ids

    duplicate_unplanned_ids = [
        activity_id
        for activity_id, count in Counter(unplanned_ids).items()
        if count > 1
    ]

    if duplicate_unplanned_ids:
        raise ValueError(
            "Duplicate IDs in unplanned_activity_ids: "
            f"{duplicate_unplanned_ids}."
        )

    unknown_unplanned_ids = (
        set(unplanned_ids) - available_activity_ids
    )

    if unknown_unplanned_ids:
        raise ValueError(
            "Unknown IDs in unplanned_activity_ids: "
            f"{sorted(unknown_unplanned_ids)}."
        )

    # 8. An activity cannot be both assigned and unplanned.
    overlapping_ids = (
        set(assigned_activity_ids) & set(unplanned_ids)
    )

    if overlapping_ids:
        raise ValueError(
            "Activities cannot be both scheduled and unplanned: "
            f"{sorted(overlapping_ids)}."
        )

    # 9. Every available activity must be accounted for.
    accounted_activity_ids = (
        set(assigned_activity_ids) | set(unplanned_ids)
    )

    missing_activity_ids = (
        available_activity_ids - accounted_activity_ids
    )

    if missing_activity_ids:
        raise ValueError(
            "The following activities were not assigned to any day "
            "or included in unplanned_activity_ids: "
            f"{sorted(missing_activity_ids)}."
        )

    # 10. Warn when high-priority activities remain unplanned.
    activities_by_id = {
        activity.id: activity
        for activity in planning_input.activities
    }

    unplanned_high_priority = [
        activity_id
        for activity_id in unplanned_ids
        if activities_by_id[activity_id].preference == Preference.ABSOLUTELY
    ]

    if unplanned_high_priority:
        warning = (
            "High-priority activities were left unplanned: "
            f"{unplanned_high_priority}."
        )

        if warning not in outline.warnings:
            outline.warnings.append(warning)

        logger.warning(warning)

    logger.info(
        "Itinerary outline validated: %d days, %d assigned activities, "
        "%d unplanned activities.",
        len(outline.days),
        len(assigned_activity_ids),
        len(unplanned_ids),
    )

    return outline

def _validate_generated_day(
    day_plan: DayPlan,
    planning_input: ItineraryPlanningInput,
    day_outline: DayOutline,
    previous_days: list[DayPlan] | None = None,
) -> DayPlan:
    """Validate and enrich a generated daily schedule."""

    # 1. Validate the date.
    if day_plan.date != day_outline.date:
        raise ValueError(
            f"Generated day has date {day_plan.date}, "
            f"but the outline specifies {day_outline.date}."
        )

    activities_by_id = {
        activity.id: activity
        for activity in planning_input.activities
    }

    assigned_activity_ids = set(day_outline.activity_ids)
    previous_activity_ids = {
        item.activity_id
        for previous_day in (previous_days or [])
        for item in previous_day.items
        if item.item_type == ItineraryItemType.ACTIVITY
        and item.activity_id is not None
    }

    # 2. Validate the outline itself against the supplied activities.
    unknown_assigned_ids = (
        assigned_activity_ids - set(activities_by_id)
    )

    if unknown_assigned_ids:
        raise ValueError(
            "The daily outline contains unknown activity IDs: "
            + ", ".join(sorted(unknown_assigned_ids))
        )

    # 3. Validate activity references and enrich schedule items.
    scheduled_activity_ids: list[str] = []

    for item in day_plan.items:

        if item.item_type == ItineraryItemType.ACTIVITY:

            if not item.activity_id:
                raise ValueError(
                    f"Activity '{item.title}' has no activity_id."
                )

            activity_id = item.activity_id

            if activity_id not in activities_by_id:
                raise ValueError(
                    f"Unknown activity_id: {activity_id}"
                )

            if activity_id not in assigned_activity_ids:
                raise ValueError(
                    f"Activity '{activity_id}' was not assigned "
                    f"to {day_plan.date} in the itinerary outline."
                )

            if activity_id in previous_activity_ids:
                raise ValueError(
                    f"Activity '{activity_id}' was already scheduled "
                    "on a previous day."
                )

            if activity_id in scheduled_activity_ids:
                raise ValueError(
                    f"Activity '{activity_id}' appears more than "
                    "once on {day_plan.date}."
                )

            scheduled_activity_ids.append(activity_id)

            # Reuse information from the original activity.
            activity = activities_by_id[activity_id]

            item.source_url = item.source_url or activity.source_url
            item.booking_url = item.booking_url or activity.booking_url

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

    # 4. Check that every assigned activity was scheduled.
    missing_activity_ids = (
        assigned_activity_ids - set(scheduled_activity_ids)
    )

    if missing_activity_ids:
        missing_names = [
            activities_by_id[activity_id].name
            for activity_id in sorted(missing_activity_ids)
        ]

        raise ValueError(
            f"The following activities assigned to {day_plan.date} "
            f"were not scheduled: {', '.join(missing_names)}."
        )

    logger.info(
        "Validated daily schedule for %s: %d/%d assigned activities scheduled.",
        day_plan.date,
        len(scheduled_activity_ids),
        len(assigned_activity_ids),
    )

    return day_plan

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
        _validate_schedule_overlaps(day.items)

        for item in day.items:


            if item.start_time > item.end_time:
                raise ValueError(
                    f"{item.title} schaduled with with end time before start time."
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

def _build_planne_outliner_generation_messages(
    planning_input: ItineraryPlanningInput,
    previous_issues: list[str] | None = None,
) -> list:
    """Build generation messages, including feedback from a failed attempt."""

    schema_json = json.dumps(
        ItineraryOutline.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    user_content = (
        "Create an outline of a travel itinerary using the following "
        "planning data.\n\n"
        "Return a object matching the given schema. "
        "Keep the output concise enough to fit in one response "
        "Avoid repetitive free-time or transport items. Do not repeat "
        "information already present in the input.\n\n"
        "SCHEMA:\n"
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
        SystemMessage(content=build_planne_outline_prompt()),
        HumanMessage(content=user_content),
    ]



def _build_daily_schedule_generation_messages(
    planning_input: ItineraryPlanningInput,
    itinerary_outline: ItineraryOutline,
    day_outline: DayOutline,
    previous_days: list[DayPlan] | None = None,
    previous_issues: list[str] | None = None,
) -> list:
    """Build messages for generating one day's detailed schedule."""

    user_content = (
        "Generate a detailed, time-based schedule for ONE day of the trip.\n\n"
        "Use the complete itinerary outline to understand how activities "
        "are distributed across the trip. However, generate items ONLY "
        "for the date specified in the current day outline.\n\n"
        "Do not schedule activities assigned to other dates. "
        "Do not omit activities assigned to the current date unless "
        "there is a valid reason, which must be explained.\n\n"
        "Keep descriptions concise (1-2 sentences), use at most 3 "
        "highlights per activity, and avoid unnecessary free-time "
        "or transport items.\n\n"
        "PLANNING DATA:\n"
        f"{format_planning_input(planning_input)}\n\n"
        "COMPLETE ITINERARY OUTLINE:\n"
        f"{itinerary_outline.model_dump_json(indent=2)}\n\n"
        "CURRENT DAY OUTLINE:\n"
        f"{day_outline.model_dump_json(indent=2)}"
    )

    if previous_days:
        previous_days_json = json.dumps(
            [
                day.model_dump(mode="json")
                for day in previous_days
            ],
            ensure_ascii=False,
            indent=2,
        )

        user_content += (
            "\n\nPREVIOUSLY GENERATED DAYS:\n"
            "Use these schedules to maintain continuity, avoid repetition, "
            "and consider the locations and activities already covered.\n"
            f"{previous_days_json}"
        )

    if previous_issues:
        user_content += (
            "\n\nThe previous attempt was rejected. Correct ALL of these "
            "specific issues in the new schedule:\n- "
            + "\n- ".join(previous_issues)
        )

    return [
        SystemMessage(content=build_daily_schedule_prompt()),
        HumanMessage(content=user_content),
    ]


def _generate_itinerary_outline(
    model: ChatOpenAI,
    planning_input: ItineraryPlanningInput,
) -> ItineraryOutline:

    previous_issues: list[str] = []

    structured_model = model.with_structured_output(
        ItineraryOutline,
        method="function_calling",
    )


    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        logger.info(
            "Generating itinerary outline, attempt %d/%d.",
            attempt,
            MAX_GENERATION_ATTEMPTS,
        )

        try:
            messages = _build_planne_outliner_generation_messages(
                planning_input=planning_input,
                previous_issues=previous_issues or None,
            )

            outline = structured_model.invoke(messages)

            outline = _validate_itinerary_outline(
                outline=outline,
                planning_input=planning_input,
            )

            logger.info(
                "Successfully generated itinerary outline on attempt %d.",
                attempt,
            )

            return outline

        except Exception as attempt_error:
            previous_issues = [
                f"Attempt {attempt} failed: {attempt_error}"
            ]

            logger.warning(
                "Itinerary outline generation failed on attempt %d/%d: %s",
                attempt,
                MAX_GENERATION_ATTEMPTS,
                attempt_error,
                exc_info=True,
            )

    raise ItineraryGenerationError(
        f"Could not generate a valid itinerary outline after "
        f"{MAX_GENERATION_ATTEMPTS} attempts. "
        f"Last issues: {'; '.join(previous_issues)}"
    )

def _generate_daily_schedule(
    model: ChatOpenAI,
    planning_input: ItineraryPlanningInput,
    itinerary_outline: ItineraryOutline,
    day_outline: DayOutline,
    previous_days: list[DayPlan] | None = None,
) -> DayPlan:
    """Generate and validate one daily schedule, retrying if necessary."""

    previous_issues: list[str] = []

    structured_model = model.with_structured_output(
        DayPlan,
        method="function_calling",
    )

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        logger.info(
            "Generating daily schedule for %s, attempt %d/%d.",
            day_outline.date,
            attempt,
            MAX_GENERATION_ATTEMPTS,
        )

        try:
            messages = _build_daily_schedule_generation_messages(
                planning_input=planning_input,
                itinerary_outline=itinerary_outline,
                day_outline=day_outline,
                previous_days=previous_days,
                previous_issues=previous_issues or None,
            )

            day_plan = structured_model.invoke(messages)

            # Validate the generated day against its assigned activities.
            day_plan = _validate_generated_day(
                day_plan=day_plan,
                planning_input=planning_input,
                day_outline=day_outline,
                previous_days=previous_days,
            )

            logger.info(
                "Successfully generated daily schedule for %s "
                "with %d schedule items on attempt %d.",
                day_outline.date,
                len(day_plan.items),
                attempt,
            )

            return day_plan

        except Exception as attempt_error:
            previous_issues = [
                f"Attempt {attempt} failed: {attempt_error}"
            ]

            logger.warning(
                "Daily schedule generation for %s failed "
                "on attempt %d/%d: %s",
                day_outline.date,
                attempt,
                MAX_GENERATION_ATTEMPTS,
                attempt_error,
                exc_info=True,
            )

    raise ItineraryGenerationError(
        f"Could not generate a valid daily schedule for "
        f"{day_outline.date} after {MAX_GENERATION_ATTEMPTS} attempts. "
        f"Last issues: {'; '.join(previous_issues)}"
    )

def generate_itinerary(
    planning_input: ItineraryPlanningInput,
) -> Iterator[str | Itinerary]:
    """Generate, validate and assemble a complete itinerary, yielding progress updates."""

    logger.info(
        "Generating itinerary for %s",
        planning_input.destination,
    )

    try:
        model = _build_chat_model()

        # 1. Generate and validate the complete activity allocation.
        yield "Planning the distribution of activities across your trip"

        itinerary_outline = _generate_itinerary_outline(
            model=model,
            planning_input=planning_input,
        )

        # 2. Generate each daily schedule independently.
        generated_days: list[DayPlan] = []
        total_days = len(itinerary_outline.days)

        for index, day_outline in enumerate(
            itinerary_outline.days,
            start=1,
        ):
            yield (
                f"Generating daily schedule: "
                f"day {index}/{total_days} "
                f"({day_outline.date.strftime('%d %b %Y')})"
            )

            day_plan = _generate_daily_schedule(
                model=model,
                planning_input=planning_input,
                itinerary_outline=itinerary_outline,
                day_outline=day_outline,
                previous_days=generated_days,
            )

            generated_days.append(day_plan)

        # 3. Assemble the complete itinerary.
        yield "Assembling your complete itinerary..."

        itinerary = Itinerary(
            destination=planning_input.destination,
            timezone=planning_input.timezone,
            start_date=planning_input.preferences.arrival_date,
            end_date=planning_input.preferences.departure_date,
            days=generated_days,
            unplanned_activity_ids=itinerary_outline.unplanned_activity_ids,
            warnings=itinerary_outline.warnings,
        )

        # 4. Validate the complete itinerary locally.
        yield "Validating your itinerary..."

        itinerary = _validate_generated_itinerary(
            itinerary=itinerary,
            planning_input=planning_input,
        )

        # 5. Perform the final semantic review.
        yield "Reviewing itinerary consistency..."

        review = _review_itinerary(
            model=model,
            itinerary=itinerary,
            planning_input=planning_input,
        )

        if not review.is_valid:
            raise ItineraryGenerationError(
                "The final itinerary did not pass semantic review: "
                + "; ".join(review.issues)
            )

        logger.info(
            "Successfully generated itinerary with %d days and "
            "%d schedule items.",
            len(itinerary.days),
            sum(len(day.items) for day in itinerary.days),
        )

        yield itinerary

    except ItineraryGenerationError:
        logger.exception(
            "Failed to generate itinerary for %s",
            planning_input.destination,
        )
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected error generating itinerary for %s",
            planning_input.destination,
        )

        raise ItineraryGenerationError(
            f"Could not generate itinerary: {exc}"
        ) from exc
