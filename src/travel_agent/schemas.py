
"""Pydantic schemas shared by the research, UI, and itinerary layers."""

from __future__ import annotations

from datetime import date, time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator, field_validator


#--------------------------------------
#pricing
#-------------------------------------

class PriceRepresentation(BaseModel):
    """Price expressed in its original currency and, when possible, in the user's currency."""

    # Original price found in the source.
    currency: str = Field(description="Original ISO 4217 currency code.")
    minimum: float | None = Field(default=None, ge=0)
    maximum: float | None = Field(default=None, ge=0)

    # Converted price.
    converted_currency: str | None = Field(
        default=None,
        description="Target ISO 4217 currency code selected by the user.",
    )
    converted_minimum: float | None = Field(default=None, ge=0)
    converted_maximum: float | None = Field(default=None, ge=0)

    # Conversion metadata.
    exchange_rate: float | None = Field(default=None, gt=0)
    exchange_rate_date: str | None = None
    conversion_status: Literal[
        "not_requested",
        "same_currency",
        "converted",
        "unavailable",
    ] = "not_requested"

  
    source_currency: str | None = None
    
    

class PriceOption(BaseModel):
    """One way to purchase or experience an activity."""

    description: str | None = None

    # Different currencies representing the same price
    representations: list[PriceRepresentation] = Field(
        default_factory=list
    )

    # Whether this is a verified price or an estimate
    confidence: Literal[
        "verified",
        "estimated",
        "unknown",
    ] = "unknown"

    # Per person, per group, per ticket, etc.
    pricing_basis: Literal[
        "per_person",
        "per_group",
        "per_ticket",
        "per_vehicle",
        "total",
        "unknown",
    ] = "unknown"

    mandatory: bool = True

    converted_representations: list[PriceRepresentation] = Field(
        default_factory=list
    )


class ActivityPricing(BaseModel):
    """Pricing information for an activity."""

    status: Literal[
        "free",
        "paid",
        "partially_free",
        "unknown",
    ]

    options: list[PriceOption] = Field(
        default_factory=list
    )

class CostEstimate(BaseModel):
    currency: str

    minimum: float | None = None
    maximum: float | None = None

    has_unknown_costs: bool = False
    has_estimated_prices: bool = False

    exchange_rate_date: str | None = None

class BudgetAssessment(BaseModel):
    budget: float
    currency: str

    estimated_minimum: float | None = None
    estimated_maximum: float | None = None

    has_unknown_costs: bool = False

    within_budget: bool
    requires_replanning: bool


class ActivityPricingItem(BaseModel):
    """Pricing information associated with an activity."""

    activity_id: str
    pricing: ActivityPricing


class ActivityPricingResult(BaseModel):
    """Pricing extraction results for multiple activities."""

    activities: list[ActivityPricingItem]

# ---------------------------------------------------------------------------
# Activity discovery and user preferences
# ---------------------------------------------------------------------------


class Preference(str, Enum):
    """A user's interest level for a suggested activity."""

    ABSOLUTELY = "Absolutely!"
    SOUNDS_GOOD = "Sounds good"
    MAYBE = "Maybe"


class ActivityData(BaseModel):
    """Activity information extracted by the LLM."""

    id: str
    name: str
    description: str
    category: str
    location: str | None = None

    estimated_duration_hours: float | None = Field(
        default=None,
        gt=0,
    )

    estimated_price: str | None = None
    price_currency: str | None = None

    special_notes: list[str] = Field(default_factory=list)

    source_url: str | None = None

    # Additional information useful for itinerary planning.
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    opening_hours: str | None = None
    booking_required: bool | None = None
    booking_url: str | None = None


class ActivityExtraction(BaseModel):
    """Structured LLM response containing extracted activities."""

    activities: list[ActivityData]


class Activity(ActivityData):
    """Complete activity enriched by the application."""

    image_url: str | None = None
    image_path: str | None = None
    preference: Preference | None = None
    pricing: ActivityPricing | None = None

class ActivityCollection(BaseModel):
    """Validated collection of complete activities."""

    activities: list[Activity]


class ActivityEnrichmentData(BaseModel):
    """Additional information collected for an existing activity."""

    id: str

    location: str | None = None
    estimated_duration_hours: float | None = Field(default=None, gt=0)
    estimated_price: str | None = None
    price_currency: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    opening_hours: str | None = None
    booking_required: bool | None = None
    source_url: str | None = None


class ActivityEnrichmentResult(BaseModel):
    """Enrichment results for a batch of activities."""

    activities: list[ActivityEnrichmentData]

# ---------------------------------------------------------------------------
# Trip preferences
# ---------------------------------------------------------------------------


class TripPreferences(BaseModel):
    """User preferences and constraints collected by the Gradio UI."""

    # Travelers
    adults: int = Field(default=1, ge=1)
    children: int = Field(default=0, ge=0)

    # Arrival and departure
    arrival_date: date | None = None
    arrival_time: time | None = None
    arrival_location: str | None = None
    departure_date: date | None = None
    departure_time: time | None = None
    departure_location: str | None = None
    arrival_mode: str | None = None

    # Travel costs
    travel_cost: float | None = Field(default=None, ge=0)
    budget_includes_travel: bool = False

    # Accommodation
    accommodation_needed: bool = False
    accommodation_booked: bool = False
    checkin_date: date | None = None
    checkout_date: date | None = None
    accommodation_types: list[str] = Field(default_factory=list)
    preferred_area: str | None = None
    accommodation_price: float | None = Field(default=None, ge=0)
    rooms: int | None = Field(default=None, ge=1)

    # Budget
    currency: str | None = None
    total_budget: float | None = Field(default=None, ge=0)
    accommodation_budget: float | None = Field(default=None, ge=0)
    food_budget: float | None = Field(default=None, ge=0)
    activities_budget: float | None = Field(default=None, ge=0)
    transport_budget: float | None = Field(default=None, ge=0)
    food_budget_per_meal: float | None = Field(default=None, ge=0)

    # Interests and restrictions
    interests: list[str] = Field(default_factory=list)
    things_to_avoid: list[str] = Field(default_factory=list)
    preferred_cuisines: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)

    # Daily planning
    meals_per_day: int = Field(default=3, ge=0, le=6)
    travel_pace: str | None = None
    preferred_transport: list[str] = Field(default_factory=list)

    # Maximum daily walking time, in minutes.
    walking_minutes: int | None = Field(default=None, ge=0)

    accessibility_needs: list[str] = Field(default_factory=list)
    must_visit_places: list[str] = Field(default_factory=list)
    additional_notes: str | None = None

    @model_validator(mode="after")
    def validate_trip_dates(self) -> TripPreferences:
        """Check date consistency."""

        if (
            self.arrival_date
            and self.departure_date
            and self.departure_date < self.arrival_date
        ):
            raise ValueError(
                "Departure date cannot be before arrival date."
            )

        if (
            self.checkin_date
            and self.checkout_date
            and self.checkout_date < self.checkin_date
        ):
            raise ValueError(
                "Checkout date cannot be before check-in date."
            )

        return self


    @field_validator(
        "things_to_avoid",
        "preferred_cuisines",
        "dietary_restrictions",
        "allergies",
        "accessibility_needs",
        "must_visit_places",
        mode="before",
    )
    @classmethod
    def parse_comma_separated_values(cls, value):
        """Convert comma-separated text into a list of strings."""

        if value is None or value == "":
            return []

        if isinstance(value, str):
            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        return value

    @field_validator("preferred_transport", mode="before")
    @classmethod
    def normalize_preferred_transport(cls, value):
        """Normalize preferred transportation options."""

        if value is None or value == "":
            return []

        if isinstance(value, str):
            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        return value


# ---------------------------------------------------------------------------
# Restaurant and meal planning
# ---------------------------------------------------------------------------


class MealLocationStrategy(str, Enum):
    NEAR_PREVIOUS_ACTIVITY = "near_previous_activity"
    NEAR_NEXT_ACTIVITY = "near_next_activity"
    BETWEEN_ACTIVITIES = "between_activities"
    NEAR_ACCOMMODATION = "near_accommodation"


class RestaurantOption(BaseModel):
    """A real restaurant retrieved from a verifiable source."""

    name: str
    address: str
    source_url: str

    cuisine: str | None = None
    estimated_price_per_person: float | None = Field(
        default=None,
        ge=0,
    )
    currency: str | None = None

    rating: float | None = Field(default=None, ge=0, le=5)
    opening_hours: str | None = None

    latitude: float | None = None
    longitude: float | None = None

    travel_time_from_anchor_minutes: int | None = Field(
        default=None,
        ge=0,
    )


class MealDetails(BaseModel):
    """Meal planning information and restaurant selection."""

    meal_type: Literal[
        "breakfast",
        "lunch",
        "dinner",
        "snack",
    ]

    location_strategy: MealLocationStrategy

    # Activity or accommodation used as the geographic reference.
    anchor_activity_id: str | None = None

    # Geographic area where restaurants should be searched.
    search_area: str | None = None

    # Filled only after retrieving real restaurant results.
    restaurant: RestaurantOption | None = None


# ---------------------------------------------------------------------------
# Daily itinerary
# ---------------------------------------------------------------------------


class ItineraryItemType(str, Enum):
    ACTIVITY = "activity"
    TRANSPORT = "transport"
    MEAL = "meal"
    FREE_TIME = "free_time"
    ACCOMMODATION = "accommodation"


class DayOutline(BaseModel):
    """High-level activity allocation for one day."""

    date: date

    # Activities assigned to this day, in the recommended order.
    activity_ids: list[str] = Field(default_factory=list)


class ItineraryOutline(BaseModel):
    """High-level allocation of activities across the entire trip."""

    destination: str
    timezone: str

    start_date: date
    end_date: date

    days: list[DayOutline] = Field(min_length=1)

    # Activities that cannot reasonably fit into the trip.
    unplanned_activity_ids: list[str] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(default_factory=list)


    

    

class ItineraryItem(BaseModel):
    """One scheduled block in a daily itinerary."""

    item_type: ItineraryItemType
    title: str

    # Reference to an activity from the discovery stage.
    activity_id: str | None = None

    description: str

    start_time: time
    end_time: time

    # Duration is calculated from start_time and end_time
    # if it is not provided by the model.
    duration_minutes: int | None = Field(
        default=None,
        gt=0,
    )

    location: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    

    # Additional details about what to visit or do.
    highlights: list[str] = Field(default_factory=list)

    # Source information.
    source_url: str | None = None
    booking_url: str | None = None

    # Used only for meal items.
    meal_details: MealDetails | None = None

    # Estimated costs.
    estimated_cost_per_person: float | None = Field(
        default=None,
        ge=0,
    )
    estimated_cost_total: float | None = Field(
        default=None,
        ge=0,
    )
    currency: str | None = None

    # Transportation details.
    transport_mode: str | None = None
    transport_from: str | None = None
    transport_to: str | None = None

    travel_duration_minutes: int | None = Field(
        default=None,
        ge=0,
    )

    walking_minutes: int | None = Field(
        default=None,
        ge=0,
    )

    notes: list[str] = Field(default_factory=list)

    # Estimated costs per person.
    estimated_cost_per_person: float | None = Field(
        default=None,
        ge=0,
    )
    estimated_cost_per_person_minimum: float | None = Field(
        default=None,
        ge=0,
    )
    estimated_cost_per_person_maximum: float | None = Field(
        default=None,
        ge=0,
    )

    # Estimated costs for the entire group.
    estimated_cost_total: float | None = Field(
        default=None,
        ge=0,
    )
    estimated_cost_total_minimum: float | None = Field(
        default=None,
        ge=0,
    )
    estimated_cost_total_maximum: float | None = Field(
        default=None,
        ge=0,
    )

    currency: str | None = None

    @model_validator(mode="after")
    def validate_item(self) -> ItineraryItem:
        """Validate item-specific constraints."""

        if self.end_time <= self.start_time:
            raise ValueError(
                "Item end time must be after its start time."
            )

        # Derive duration from the scheduled time interval.
        calculated_duration = (
            self.end_time.hour * 60
            + self.end_time.minute
            - self.start_time.hour * 60
            - self.start_time.minute
        )

        if self.duration_minutes is None:
            self.duration_minutes = calculated_duration

        elif self.duration_minutes != calculated_duration:
            raise ValueError(
                "duration_minutes must match the scheduled "
                "start and end times."
            )

        if (
            self.item_type == ItineraryItemType.MEAL
            and self.meal_details is None
        ):
            raise ValueError(
                "Meal items must contain meal_details."
            )

        if (
            self.item_type != ItineraryItemType.MEAL
            and self.meal_details is not None
        ):
            raise ValueError(
                "Only meal items can contain meal_details."
            )

        return self


def _validate_schedule_overlaps(
    items: list[ItineraryItem],
) -> None:
    """Reject overlapping items, except meals within activities."""

    for index, current in enumerate(items):
        for previous in items[:index]:

            # No overlap.
            if (
                current.start_time >= previous.end_time
                or previous.start_time >= current.end_time
            ):
                continue

            # A meal can take place during a longer activity,
            # such as a full-day trip.
            meal_inside_activity = (
                current.item_type == ItineraryItemType.MEAL
                and previous.item_type == ItineraryItemType.ACTIVITY
                and previous.start_time <= current.start_time
                and current.end_time <= previous.end_time
            ) or (
                previous.item_type == ItineraryItemType.MEAL
                and current.item_type == ItineraryItemType.ACTIVITY
                and current.start_time <= previous.start_time
                and previous.end_time <= current.end_time
            )

            if meal_inside_activity:
                continue

            raise ValueError(
                f"Schedule conflict: '{previous.title}' "
                f"overlaps with '{current.title}'."
            )


class DayPlan(BaseModel):
    """Complete itinerary for one day."""

    date: date

    items: list[ItineraryItem] = Field(min_length=1)

    estimated_cost_total: float | None = Field(
        default=None,
        ge=0,
    )

    walking_minutes_total: int | None = Field(
        default=None,
        ge=0,
    )

    summary: str | None = None

    unfulfilled_activity_ids: list[str] = Field(default_factory=list)

    
    
    @model_validator(mode="after")
    def validate_schedule(self) -> DayPlan:
        """Ensure itinerary items are chronological and do not overlap."""

        self.items = sorted(
            self.items,
            key=lambda item: item.start_time,
        )

        _validate_schedule_overlaps(self.items)

        # Calculate walking time only when every item has
        # a known walking duration.
        if all(
            item.walking_minutes is not None
            for item in self.items
        ):
            self.walking_minutes_total = sum(
                item.walking_minutes
                for item in self.items
            )
        else:
            self.walking_minutes_total = None

        return self



class Itinerary(BaseModel):
    """Complete trip itinerary."""

    destination: str
    timezone: str

    start_date: date
    end_date: date

    days: list[DayPlan] = Field(min_length=1)

    unplanned_activity_ids: list[str] = Field(
        default_factory=list
    )

    # Estimated activity costs.
    estimated_cost_per_person: float | None = Field(default=None, ge=0)
    estimated_cost_per_person_minimum: float | None = Field(default=None, ge=0)
    estimated_cost_per_person_maximum: float | None = Field(default=None, ge=0)

    estimated_cost_total: float | None = Field(default=None, ge=0)
    estimated_cost_total_minimum: float | None = Field(default=None, ge=0)
    estimated_cost_total_maximum: float | None = Field(default=None, ge=0)

    currency: str | None = None

    activity_costs_incomplete: bool = False

    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_itinerary(self) -> Itinerary:
        """Validate itinerary dates and daily plans."""

        if self.end_date < self.start_date:
            raise ValueError(
                "Itinerary end date cannot be before start date."
            )

        # Sort days chronologically.
        self.days = sorted(
            self.days,
            key=lambda day: day.date,
        )

        day_dates = [day.date for day in self.days]

        if len(day_dates) != len(set(day_dates)):
            raise ValueError(
                "Itinerary cannot contain duplicate dates."
            )

        # Require one DayPlan for every date of the trip.
        expected_dates = {
            self.start_date.fromordinal(
                self.start_date.toordinal() + offset
            )
            for offset in range(
                (self.end_date - self.start_date).days + 1
            )
        }

        actual_dates = set(day_dates)

        missing_dates = expected_dates - actual_dates

        if missing_dates:
            raise ValueError(
                "Missing daily plans for: "
                + ", ".join(
                    str(day) for day in sorted(missing_dates)
                )
            )

        return self


# ---------------------------------------------------------------------------
# Itinerary planning input
# ---------------------------------------------------------------------------


class ItineraryPlanningInput(BaseModel):
    """All information required to generate a personalized itinerary."""

    destination: str
    timezone: str

    # Contains dates, travelers, budgets, transport, meals,
    # accommodation, interests, and user restrictions.
    preferences: TripPreferences

    # Activities rated by the user.
    # Each activity must have a preference value.
    activities: list[Activity] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_planning_input(self) -> ItineraryPlanningInput:
        """Validate the information required for itinerary planning."""

        if not self.destination.strip():
            raise ValueError("Destination cannot be empty.")

        if not self.timezone.strip():
            raise ValueError("Timezone cannot be empty.")

        if not self.preferences.arrival_date:
            raise ValueError("Arrival date is required.")

        if not self.preferences.departure_date:
            raise ValueError("Departure date is required.")

        activity_ids = [activity.id for activity in self.activities]

        if len(activity_ids) != len(set(activity_ids)):
            raise ValueError("Activity IDs must be unique.")

        activities_without_preferences = [
            activity.name
            for activity in self.activities
            if activity.preference is None
        ]

        if activities_without_preferences:
            raise ValueError(
                "All activities must have a user preference. "
                f"Missing preferences: {activities_without_preferences}"
            )

        return self


