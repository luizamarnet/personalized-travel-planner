"""Focused tests for provenance checks and activity state updates."""

from travel_planner.research import validate_activity_payload
from travel_planner.schemas import Preference
from travel_planner.ui import clear_preferences, update_preference


SOURCE_URL = "https://example.com/things-to-do"
IMAGE_URL = "https://images.example.com/activity.jpg"


def _activity_payload(source_url=SOURCE_URL, image_url=IMAGE_URL):
    return {
        "activities": [
            {
                "name": "Historic City Walk",
                "description": "Explore the historic streets with a self-guided walk.",
                "category": "Sightseeing",
                "location": "Old Town",
                "estimated_duration_hours": 2,
                "estimated_price": None,
                "image_url": image_url,
                "source_url": source_url,
            }
        ]
    }


def test_only_search_result_sources_are_accepted():
    collection = validate_activity_payload(
        _activity_payload(source_url="https://unverified.example.org/page"),
        source_urls={SOURCE_URL},
        image_urls={IMAGE_URL},
    )
    assert collection.activities == []


def test_unverified_image_is_removed_but_activity_is_kept():
    collection = validate_activity_payload(
        _activity_payload(image_url="https://unverified.example.org/photo.jpg"),
        source_urls={SOURCE_URL},
        image_urls={IMAGE_URL},
    )
    assert len(collection.activities) == 1
    assert collection.activities[0].image_url is None


def test_preference_is_saved_in_session_state():
    state = {
        "destination": "Copenhagen",
        "activities": [{"id": "activity-1", "name": "Nyhavn", "preference": None}],
        "preferences": {},
        "selected_activities": [],
        "dates": None,
        "itinerary": None,
    }
    updated = update_preference(state, "activity-1", Preference.ABSOLUTELY.value)
    assert updated["preferences"]["activity-1"] == "Absolutely!"
    assert updated["selected_activities"][0]["name"] == "Nyhavn"


def test_clear_preferences_keeps_discovered_activities():
    state = {
        "destination": "Copenhagen",
        "activities": [{"id": "activity-1", "name": "Nyhavn", "preference": "Maybe"}],
        "preferences": {"activity-1": "Maybe"},
        "selected_activities": [{"id": "activity-1"}],
    }
    updated = clear_preferences(state)
    assert len(updated["activities"]) == 1
    assert updated["preferences"] == {}
    assert updated["selected_activities"] == []
