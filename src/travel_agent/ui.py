
"""Gradio interface for activity discovery and preference collection."""

from __future__ import annotations

import copy
import html
import logging
from typing import Any

import gradio as gr
import requests
import tempfile

from travel_agent.research import search_activities
from travel_agent.itinerary import generate_itinerary
from travel_agent.schemas import (
    Activity,
    ItineraryPlanningInput,
    Preference,
    TripPreferences,
)

from travel_agent.itinerary_renderer import render_itinerary_html

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

MOBILE_SCROLL_CSS = """
html,
body {
    min-height: 100%;
    height: auto !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

body {
    -webkit-overflow-scrolling: touch;
    touch-action: pan-y;
}

#root {
    min-height: 100vh;
    height: auto !important;
    overflow: visible !important;
}

.gradio-container {
    min-height: 100vh !important;
    height: auto !important;
    overflow: visible !important;
}
"""

PREFERENCE_CHOICES = [item.value for item in Preference]
RADIO_CHOICES = [*PREFERENCE_CHOICES, "Not this time"]

INITIAL_STATE: dict[str, Any] = {
    "destination": "",
    "destination_location": None,
    "location_results": {},
    "trip_preferences": {},
    "dates": None,
    "activities": [],
    "preferences": {},
    "selected_activities": [],
    "itinerary": None,
}

def _activity_to_dict(activity: Any) -> dict[str, Any]:
    """Convert an activity model into a JSON-compatible dictionary."""

    if hasattr(activity, "model_dump"):
        return activity.model_dump(mode="json")

    return dict(activity)


def loading_message(message: str = "Searching for activities") -> str:
    """Return an animated loading message."""

    return f"""
    <div class="loading-message">
        {html.escape(message)}
        <span class="loading-dots">
            <span>.</span><span>.</span><span>.</span>
        </span>
    </div>

    <style>
        .loading-message {{
            text-align: center;
            font-size: 16px;
            margin: 15px 0;
        }}

        .loading-dots span {{
            animation: travel-planner-blink 1.4s infinite both;
            font-weight: bold;
        }}

        .loading-dots span:nth-child(2) {{
            animation-delay: 0.2s;
        }}

        .loading-dots span:nth-child(3) {{
            animation-delay: 0.4s;
        }}

        @keyframes travel-planner-blink {{
            0%, 80%, 100% {{
                opacity: 0;
            }}

            40% {{
                opacity: 1;
            }}
        }}
    </style>
    """


def search_locations(
    current_value: str | None,
    app_state: dict[str, Any] | None,
    evt: gr.KeyUpData,
):
    """Search for matching locations while the user types."""

    state = copy.deepcopy(app_state or INITIAL_STATE)
    query = (evt.input_value or "").strip()

    # Typing a new query invalidates the previously selected location.
    state["destination"] = ""
    state["destination_location"] = None
    state["location_results"] = {}

    if len(query) < 2:
        return (
            gr.update(choices=[]),
            "",
            "",
            state,
        )

    try:
        response = requests.get(
            GEOCODING_URL,
            params={
                "name": query,
                "count": 10,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )
        response.raise_for_status()

        results = response.json().get("results", [])
        locations = {}

        for item in results:
            location_id = str(item["id"])

            name = item["name"]
            admin1 = item.get("admin1")
            admin2 = item.get("admin2")
            country = item.get("country")

            # Build a readable and geographically specific label.
            parts = [name]

            if admin1 and admin1 not in parts:
                parts.append(admin1)

            if admin2 and admin2 not in parts:
                parts.append(admin2)

            if country:
                parts.append(country)

            display_name = ", ".join(parts)

            locations[location_id] = {
                "id": location_id,
                "name": name,
                "region": admin1,
                "subregion": admin2,
                "country": country,
                "country_code": item.get("country_code"),
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "timezone": item.get("timezone"),
                "display_name": display_name,
            }

        state["location_results"] = locations

        choices = [
            (location["display_name"], location_id)
            for location_id, location in locations.items()
        ]

        return (
            gr.update(choices=choices),
            "",
            "",
            state,
        )

    except requests.RequestException:
        logger.exception(
            "Location search failed for query: %s",
            query,
        )

        return (
            gr.update(choices=[]),
            "",
            "",
            state,
        )


def select_location(
    location_id: str | None,
    app_state: dict[str, Any] | None,
):
    """Save the complete selected location."""

    state = copy.deepcopy(app_state or INITIAL_STATE)

    location = state.get("location_results", {}).get(
        str(location_id)
    )

    if not location:
        state["destination"] = ""
        state["destination_location"] = None

        return "", "", state

    state["destination"] = location["display_name"]
    state["destination_location"] = location

    return (
        location["display_name"],
        location.get("country") or "",
        state,
    )


def discover_activities(
    location_id: str | None,
    app_state: dict[str, Any] | None,
    *values,
):
    """Save questionnaire values and search for activities."""

    state = copy.deepcopy(app_state or INITIAL_STATE)

    # 1. Save the current questionnaire values.
    field_names = [
        "traveling_with_kids",
        "arrival_date",
        "arrival_time",
        "arrival_location",
        "departure_date",
        "departure_time",
        "departure_location",
        "arrival_mode",
        "departure_mode",
        "preferred_area",
        #"currency",
        #"activities_budget",
        "interests",
        "things_to_avoid",
        "meals_per_day",
        "travel_pace",
        "preferred_transport",
        "accessibility_needs",
        "must_visit_places",
        "additional_notes",
    ]

    state["trip_preferences"] = dict(zip(field_names, values))

    # 2. Validate the selected destination.
    location = state.get("destination_location")

    if (
        not location
        or str(location.get("id")) != str(location_id)
    ):
        yield (
            "<p>Please select a destination.</p>",
            state,
        )
        return
    if (
        not state["trip_preferences"]["arrival_date"]
    ):
        yield (
            "<p>Please select an arrival date.</p>",
            state,
        )
        return
    if (
        not state["trip_preferences"]["departure_date"]
    ):
        yield (
            "<p>Please select a departure date.</p>",
            state,
        )
        return
    if (
        state["trip_preferences"]["departure_date"] < state["trip_preferences"]["arrival_date"]
    ):
        yield (
            "<p>Departure date can't be earlier than arrival date.</p>",
            state,
        )
        return

    destination = location["display_name"]

    # 3. Display the loading animation.
    yield loading_message(), state

    # 4. Search using the saved questionnaire preferences.
    try:
        activities = search_activities(
        destination=destination,
        preferences=state["trip_preferences"],
        )

    except Exception:
        logger.exception(
            "Activity discovery failed for %s",
            destination,
        )

        yield (
            "<p>Activity search failed. Please try again.</p>",
            state,
        )
        return

    # 5. Update the results.
    state.update(
        {
            "destination": destination,
            "activities": [
                _activity_to_dict(activity)
                for activity in activities
            ],
            "preferences": {},
            "selected_activities": [],
            "itinerary": None,
        }
    )

    if not activities:
        yield (
            f"<p>No activities could be verified for "
            f"{html.escape(destination)}. Try another search.</p>",
            state,
        )
        return

    yield (
        f"<p>Found {len(activities)} activities for "
        f"{html.escape(destination)}. "
        f"Choose your preferences below.</p>",
        state,
    )


def update_preference(
    app_state: dict[str, Any] | None,
    activity_id: str,
    preference: str | None,
):
    """Update one activity preference and rebuild the selected activity list."""

    state = copy.deepcopy(app_state or INITIAL_STATE)

    if preference not in PREFERENCE_CHOICES:
        preference = None
        state.setdefault("preferences", {}).pop(activity_id, None)
    else:
        state.setdefault("preferences", {})[activity_id] = preference

    for activity in state.get("activities", []):
        if activity.get("id") == activity_id:
            activity["preference"] = preference
            break

    state["selected_activities"] = [
        {
            **activity,
            "preference": state["preferences"].get(activity["id"]),
        }
        for activity in state.get("activities", [])
        if activity["id"] in state["preferences"]
    ]

    return state


def clear_preferences(
    app_state: dict[str, Any] | None,
):
    """Clear all activity preferences."""

    state = copy.deepcopy(app_state or INITIAL_STATE)

    state["preferences"] = {}
    state["selected_activities"] = []

    for activity in state.get("activities", []):
        activity["preference"] = None

    return state

def create_itinerary(
    app_state: dict[str, Any] | None,
):
    """Generate an itinerary and create a standalone HTML file."""

    state = copy.deepcopy(app_state or INITIAL_STATE)

    try:
        # Validate the destination and timezone.
        location = state.get("destination_location")

        if not state.get("destination") or not location:
            raise ValueError("Please select a destination first.")

        timezone = location.get("timezone")

        # Keep only activities classified by the user.
        selected_activities = [
            Activity.model_validate(activity)
            for activity in state.get("activities", [])
            if activity.get("preference") is not None
        ]

        if not selected_activities:
            raise ValueError(
                "Please classify at least one activity before "
                "generating your itinerary."
            )

        # Validate the planning data.
        planning_input = ItineraryPlanningInput(
            destination=state["destination"],
            timezone=timezone,
            preferences=TripPreferences.model_validate(
                state["trip_preferences"]
            ),
            activities=selected_activities,
        )
        
        # Display the loading animation.
        yield (
            state,
            loading_message("Generating your itinerary"),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )
        # Generate the itinerary.
        itinerary = generate_itinerary(planning_input)

        # Render the itinerary HTML.
        html_output = render_itinerary_html(itinerary)

        # Save the itinerary in the session state.
        state["itinerary"] = itinerary.model_dump(mode="json")

        # Create a standalone HTML document.
        document = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Travel Itinerary - {html.escape(itinerary.destination)}</title>
        </head>
        <body>
            {html_output}
        </body>
        </html>
        """

        # Create a temporary HTML file for download.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            prefix="travel_itinerary_",
            encoding="utf-8",
            delete=False,
        ) as file:
            file.write(document)
            itinerary_file = file.name

        return (
            state,
            "Your itinerary was generated successfully!",
            gr.update(value=itinerary_file, visible=True),
            gr.update(value=html_output, visible=True),
        )

    except Exception as exc:
        logger.exception("Itinerary generation failed.")

        return (
            state,
            f"Could not generate itinerary: {html.escape(str(exc))}",
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        )

    

def _render_activity_card(
    activity: dict[str, Any],
    app_state: gr.State,
):
    """Render one activity card with its image and preference selector."""

    activity_id = activity["id"]

    safe_name = html.escape(activity.get("name", "Activity"))
    safe_description = html.escape(
        activity.get("description", "")
    )
    safe_category = html.escape(
        activity.get("category", "Other")
    )
    safe_location = html.escape(
        activity.get("location") or "Location not specified"
    )
    safe_special_notes = [
        html.escape(note, quote=True)
        for note in activity.get("special_notes", [])
    ]
    safe_source = html.escape(
        activity.get("source_url", ""),
        quote=True,
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=220):
            # Images are downloaded and cached in research.py.
            image_path = activity.get("image_path")

            if image_path:
                gr.Image(
                    value=image_path,
                    label=None,
                    show_label=False,
                    interactive=False,
                    height=180,
                    container=False,
                )
            else:
                gr.Markdown("*Image unavailable.*")

        with gr.Column(scale=2, min_width=300):
            gr.Markdown(f"### {safe_name}")

            gr.Markdown(
                f"`{safe_category}` * {safe_location}"
            )

            gr.Markdown(safe_description)

            notes_markdown = "\n".join(
                f"- {note}" for note in safe_special_notes
            )

            gr.Markdown(
                f"**Please, attempt to:**\n\n{notes_markdown}"
            )

            if safe_source.startswith(("https://", "http://")):
                gr.Markdown(f"[View source]({safe_source})")
            else:
                gr.Markdown("*Source unavailable.*")

            activity_id_state = gr.State(activity_id)

            radio = gr.Radio(
                choices=RADIO_CHOICES,
                value=activity.get("preference") or "Not this time",
                label="Your preference",
                interactive=True,
            )

            radio.change(
                fn=update_preference,
                inputs=[
                    app_state,
                    activity_id_state,
                    radio,
                ],
                outputs=[app_state],
            )

    gr.Markdown("---")


def build_app() -> gr.Blocks:
    """Build and configure the Gradio application."""

    time_choices = [
        f"{hour:02d}:{minute:02d}"
        for hour in range(24)
        for minute in (0, 30)
    ]

    currencies = [
        "EUR - Euro",
        "USD - US Dollar",
        "BRL - Brazilian Real",
        "DKK - Danish Krone",
        "GBP - British Pound",
        "SEK - Swedish Krona",
        "NOK - Norwegian Krone",
        "CHF - Swiss Franc",
        "JPY - Japanese Yen",
        "CAD - Canadian Dollar",
        "AUD - Australian Dollar",
        "CNY - Chinese Yuan",
        "INR - Indian Rupee",
        "MXN - Mexican Peso",
        "ARS - Argentine Peso",
        "CLP - Chilean Peso",
        "COP - Colombian Peso",
        "ZAR - South African Rand",
    ]

    with gr.Blocks(
        title="AI Travel Planner",
    ) as demo:

        app_state = gr.State(copy.deepcopy(INITIAL_STATE))

        gr.Markdown(
            """
            # AI Travel Planner

            Plan your trip by sharing your destination,
            dates, and travel preferences.
            """
        )
        # update text later
        """
        Plan your trip by sharing your destination,
        dates, budget and travel preferences.
        """

        # A. Destination and travelers

        gr.Markdown("## A · Destination and travelers")

        destination = gr.Dropdown(
            choices=[],
            value=None,
            label="Destination (start typing a city)",
            interactive=True,
            filterable=True,
            allow_custom_value=True,
        )

        with gr.Row():
            destination_display = gr.Textbox(
                label="Selected destination",
                interactive=False,
            )

            country = gr.Textbox(
                label="Destination country",
                interactive=False,
            )

        traveling_with_kids = gr.Checkbox(
            label="Traveling with kids",
            value=False,
        )

        # B. Dates and arrival

        gr.Markdown("## B · Dates and arrival")

        with gr.Row():
            arrival_date = gr.DateTime(
                label="Arrival date",
                include_time=False,
                type="string",
            )

            arrival_time = gr.Dropdown(
                choices=time_choices,
                label="Arrival time",
                value=None,
                allow_custom_value=False,
            )

        arrival_mode = gr.Dropdown(
                    choices=[
                        "plane",
                        "train",
                        "bus",
                        "car",
                        "ferry",
                        "other",
                    ],
                    label="Arrival mode",
                    value="train",
                )

        arrival_location = gr.Textbox(
            label="Arrival location",
            placeholder="Airport, train station, city center, historcial center, your hotel, etc.",
        )

        with gr.Row():
            departure_date = gr.DateTime(
                label="Departure date",
                include_time=False,
                type="string",
            )

            departure_time = gr.Dropdown(
                choices=time_choices,
                label="Departure time",
                value=None,
                allow_custom_value=False,
            )

        departure_location = gr.Textbox(
            label="Departure location",
            placeholder="Airport, train station, city center, historcial center, your hotel, etc.",
        )

        departure_mode = gr.Dropdown(
            choices=[
                "plane",
                "train",
                "bus",
                "car",
                "ferry",
                "other",
            ],
            label="Departure mode",
            value="train",
        )

        

        # C. Accommodation

        gr.Markdown("## C · Accommodation")


        preferred_area = gr.Textbox(
            label="Accomodation/Hotel namde or address ",
        )

        
        # D. Budget

        '''gr.Markdown("## D · Budget")

        currency = gr.Dropdown(
            choices=currencies,
            label="Currency",
            value="DKK - Danish Krone",
            filterable=True,
        )

        
        activities_budget = gr.Number(
            label="Activities whole-trip ceiling",
            minimum=0,
        )'''

    

        # E. Interests

        gr.Markdown("## E · Interests")

        interests = gr.Dropdown(
            choices=[
                "Culture and history",
                "Museums and galleries",
                "Architecture",
                "Nature and outdoors",
                "Beaches",
                "Food and gastronomy",
                "Shopping",
                "Nightlife",
                "Sports",
                "Adventure",
                "Relaxation",
                "Local experiences",
                "Family activities",
                "Photography",
            ],
            label="Interests",
            multiselect=True,
        )

        things_to_avoid = gr.Textbox(
            label="Things to avoid (comma-separated)",
        )

        # Meals

        meals_per_day = gr.Slider(
            minimum=1,
            maximum=6,
            value=3,
            step=1,
            label="Meals per day",
        )

        
        # G. Travel style

        gr.Markdown("## G · Travel style")

        travel_pace = gr.Dropdown(
            choices=[
                "relaxed",
                "balanced",
                "intensive",
            ],
            value="balanced",
            label="Travel pace",
        )

        preferred_transport = gr.Dropdown(
            choices=[
                "walking",
                "bus",
                "Tram",
                "Train",
                "Underground",
                "taxi",
                "car",
                "bicycle",
            ],
            multiselect=True,
            label="Preferred transportation",
        )

        accessibility_needs = gr.Textbox(
            label="Accessibility needs (comma-separated)",
        )

        must_visit_places = gr.Textbox(
            label="Must-visit places (comma-separated)",
        )

        # H. Additional notes

        gr.Markdown("## H · Additional notes")

        additional_notes = gr.Textbox(
            label="Additional notes",
            lines=5,
            placeholder="Tell us anything else that could help plan your trip...",
        )


               

        save_inputs = [
            app_state,
            traveling_with_kids,
            arrival_date,
            arrival_time,
            arrival_location,
            departure_date,
            departure_time,
            departure_location,
            arrival_mode,
            departure_mode,
            preferred_area,
            #currency,
            #activities_budget,
            interests,
            things_to_avoid,
            meals_per_day,
            travel_pace,
            preferred_transport,
            accessibility_needs,
            must_visit_places,
            additional_notes,
        ]
    
        # Location autocomplete.

        destination.key_up(
            fn=search_locations,
            inputs=[
                destination,
                app_state,
            ],
            outputs=[
                destination,
                destination_display,
                country,
                app_state,
            ],
            trigger_mode="always_last",
            show_progress="hidden",
        )

        destination.change(
            fn=select_location,
            inputs=[
                destination,
                app_state,
            ],
            outputs=[
                destination_display,
                country,
                app_state,
            ],
            show_progress="hidden",
        )

        # Activity discovery.

        gr.Markdown("---")

        search_button = gr.Button(
            "Find activities",
            variant="primary",
        )

        status = gr.HTML("")

        search_button.click(
            fn=discover_activities,
            inputs=[
                destination,
                app_state,
                *save_inputs[1:],
            ],
            outputs=[
                status,
                app_state,
            ],
        )

      

        # Activity rendering.

        @gr.render(inputs=[app_state])
        def render_results(state: dict[str, Any]):
            activities = state.get("activities", [])

            if not activities:
                return

            safe_destination = html.escape(
                state.get("destination", "your destination")
            )

            gr.Markdown(
                f"## Activities in {safe_destination}"
            )

            gr.Markdown(
                f"{len(state.get('preferences', {}))} of "
                f"{len(activities)} activities classified"
            )

            for activity in activities:
                _render_activity_card(
                    activity=activity,
                    app_state=app_state,
                )

            clear_button = gr.Button(
                "Clear preferences",
                size="sm",
            )

            clear_button.click(
                fn=clear_preferences,
                inputs=[app_state],
                outputs=[app_state],
            )

            '''gr.Markdown("## Your preferences")

            if state.get("selected_activities"):
                summary = [
                    {
                        "activity": item["name"],
                        "preference": item["preference"],
                        "category": item["category"],
                        "source_url": item["source_url"],
                    }
                    for item in state["selected_activities"]
                ]

                gr.JSON(
                    value=summary,
                    label="Selections saved in this session",
                )
            else:
                gr.Markdown("No preferences selected yet.")'''

        
                # Itinerary generation.

        #gr.Markdown("---")
        gr.Markdown("## Generate your itinerary")

        itinerary_button = gr.Button(
            "Generate itinerary",
            variant="primary",
        )

        itinerary_status = gr.HTML("")

        itinerary_download = gr.File(
            label="Download your itinerary",
            file_count="single",
            visible=False,
        )

        itinerary_output = gr.HTML(
            label="Itinerary preview",
            visible=False,
        )

        itinerary_button.click(
            fn=create_itinerary,
            inputs=[app_state],
            outputs=[
                app_state,
                itinerary_status,
                itinerary_download,
                itinerary_output,
            ],
        )
        
        
        gr.Markdown(
            "---\n"
            "*Activity information and images are sourced from web "
            "search results. Availability, prices, and opening hours "
            "should be checked with the provider.*"
        )

        

    return demo
