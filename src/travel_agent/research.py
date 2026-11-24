
"""Search the web and extract source-grounded tourist activities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
#from openai import AzureOpenAI
#from tavily import TavilyClient

from langchain_openai import ChatOpenAI
#from langchain.chat_models import init_chat_model
from langchain_azure_ai.tools.builtin import WebSearchTool
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_core.messages import SystemMessage, HumanMessage

from travel_agent.schemas import Activity, ActivityExtraction



logger = logging.getLogger(__name__)

load_dotenv()


credential = DefaultAzureCredential()


MAX_IMAGE_SIZE = 8 * 1024 * 1024  # 8 MB

IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class ResearchConfigurationError(RuntimeError):
    """Raised when required search or Azure OpenAI settings are missing."""


def _as_url_list(images: list[Any]) -> list[str]:
    """Extract unique HTTP image URLs from Tavily image results."""
    urls: list[str] = []

    for item in images:
        candidate = (
            item.get("url")
            if isinstance(item, dict)
            else item
        )

        if (
            isinstance(candidate, str)
            and candidate.startswith(("https://", "http://"))
        ):
            urls.append(candidate)

    return list(dict.fromkeys(urls))




def search_activity_images(
    extraction_model: TavilyClient,
    activity: str,
    destination: str,
) -> list[str]:
    """Find image candidates for one specific tourist activity."""

    response = extraction_model.invoke([
        SystemMessage(content=("""You are a travelling agent and need to find images to illustrate an itinerary.
        Find the most relevant images from which you can stract the image URL for downloading. 
        Rules:
        - Return only the direct image URL.
        - Do not return a webpage URL.
        - Do not return a Wikimedia Commons file page.
        - Prefer URLs ending in .jpg, .jpeg, .png, or .webp.
        - If no direct image URL is available, return NONE.
        - Do not invent or construct URLs.""")),
        HumanMessage(content=f"""for the touristic attraction {activity.name} in {destination}, described 
        as {activity.description}, find an imga URL"""),
    ])
    print("response.content.strip():", response.content.strip())
    url = response.content.strip()

    if (
        isinstance(url, str)
        and url.startswith(("https://", "http://"))
    ):
        url = url
    else:
        url=[]
    return url #_as_url_list(url)


def _get_image_cache_dir() -> Path:
    """Return the local directory used to cache downloaded images."""
    configured_path = os.getenv("TRAVEL_PLANNER_IMAGE_CACHE")

    if configured_path:
        cache_dir = Path(configured_path)
    else:
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / ".cache" / "travel_planner" / "images"

    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def download_activity_image(
    image_url: str | None,
    activity_id: str,
) -> str | None:
    """Download an image and return its cached local file path."""

    if not image_url:
        return None

    parsed_url = urlparse(image_url)

    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
    ):
        logger.warning("Invalid image URL: %s", image_url)
        return None

    cache_dir = _get_image_cache_dir()

    temporary_path: str | None = None

    try:
        with requests.get(
            image_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=(5, 15),
            stream=True,
        ) as response:

            response.raise_for_status()

            content_type = (
                response.headers.get("Content-Type", "")
                .split(";")[0]
                .strip()
                .lower()
            )

            extension = IMAGE_EXTENSIONS.get(content_type)

            if not extension:
                logger.warning(
                    "Unsupported image content type '%s': %s",
                    content_type,
                    image_url,
                )
                return None

            content_length = response.headers.get("Content-Length")

            if content_length:
                try:
                    if int(content_length) > MAX_IMAGE_SIZE:
                        logger.warning(
                            "Image exceeds the maximum allowed size: %s",
                            image_url,
                        )
                        return None
                except ValueError:
                    pass

            final_path = cache_dir / f"{activity_id}{extension}"

            # Reuse an existing cached image.
            if final_path.exists() and final_path.stat().st_size > 0:
                return str(final_path)

            total_size = 0

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_dir,
                suffix=".tmp",
                delete=False,
            ) as temporary_file:

                temporary_path = temporary_file.name

                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue

                    total_size += len(chunk)

                    if total_size > MAX_IMAGE_SIZE:
                        logger.warning(
                            "Image exceeds the maximum allowed size: %s",
                            image_url,
                        )
                        return None

                    temporary_file.write(chunk)

            if total_size == 0:
                logger.warning("Downloaded image is empty: %s", image_url)
                return None

            os.replace(temporary_path, final_path)
            temporary_path = None

            logger.info(
                "Downloaded image for activity %s",
                activity_id,
            )

            return str(final_path)

    except requests.RequestException as exc:
        logger.warning(
            "Could not download image (%s): %s",
            image_url,
            exc,
        )
        return None

    except OSError:
        logger.exception(
            "Could not save image for activity %s",
            activity_id,
        )
        return None

    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                logger.warning(
                    "Could not remove temporary image file: %s",
                    temporary_path,
                )


def _enrich_activity_image(
    activity: Activity,
    extraction_model: TavilyClient,
    destination: str,
) -> Activity:
    """Find and download the first accessible image for an activity."""

    try:
        image_urls = search_activity_images(
            extraction_model=extraction_model,
            activity=activity,
            destination=destination,
        )

    except Exception:
        logger.exception(
            "Image search failed for activity: %s",
            activity.name,
        )
        return activity

    #for image_url in image_urls:
    image_path = download_activity_image(
        image_url=image_urls,
        activity_id=activity.id,
    )
    print(image_urls)
    print("image_path:",image_path)
    if image_path:
        return activity.model_copy(
            update={
                "image_url": image_url,
                "image_path": image_path,
            }
        )

    logger.warning(
        "No accessible image found for activity: %s",
        activity.name,
    )

    return activity


def search_activities(
    destination: str,
    preferences: dict,
    max_results: int = 50,
) -> list[Activity]:
    """Find activities, validate their sources, and enrich them with images."""

    destination = destination.strip()

    if not destination:
        raise ValueError("Please enter a destination city.")

   
    model = ChatOpenAI(
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

  
    system_prompt = f"""
    You are a travel research assistant.

    Research real tourist and leisure activities in the specified destination.
    
    Give preference to search in webpages which give suggestions about several activities for a 
    chosen location, such as '10 thing to do in Paris'.

    If there are day trips commonly made by tourists from the chosen destination, also ass that. In this case, add notes
    about the distance from the destination and approximate travel time.
    
    Find distinct activities.

    For each activity, collect:
    - name
    - description
    - category
    - location
    - estimated_duration_hours
    - estimated_price
    - special_notes
    - source_url

    RESEARCH RULES:

    1. Use only information supported by reliable sources.
    2. Copy source_url exactly from a consulted source.
    3. Never invent URLs, prices, opening hours, addresses, or facts.
    4. Use null when information is unavailable.
    5. Prefer distinct and interesting activities.
    6. Avoid generic suggestions such as walking around the city.
    7. Suggest one specific location per activity.
    8. Write short descriptions and concise special notes.

    USER PREFERENCES:

    Consider the supplied interests, travel pace, walking limitations,
    places to visit, things to avoid, accessibility requirements,
    additional notes, travel dates, and children.

    If travel dates are provided, prioritize seasonal activities and
    events occurring during that period.

    If no dates are provided, consider seasonal activities occurring
    during the next four weeks.

    For accessibility, walking limitations, and children, mention
    relevant restrictions or suitability in special_notes.

    CATEGORIES:

    Use consistent short categories:
    Sightseeing, Museums & Culture, Nature & Outdoors,
    Food & Drinks, Entertainment, Shopping, Local Experiences,
    or Seasonal Activities.

    - Return up to {max_results} distinct activities.
    - Provide enough options for the user to choose from.
    - Prioritize variety across different categories and experiences.
    - Do not reduce the number of activities just to produce a shorter response.
    """

    
    research_model = model.bind_tools([WebSearchTool()])

    print("destination")
    print(destination)
    print("json.dumps(preferences, indent=2, ensure_ascii=False)")
    print(json.dumps(preferences, indent=2, ensure_ascii=False))

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
            f"Research activities in: {destination}\n\n"
            f"Preferences and personal travel information:\n"
            f"{json.dumps(preferences, indent=2, ensure_ascii=False)}"
        """)
    ]

    
    research_result = research_model.invoke(messages)

    extraction_model = model.with_structured_output(
        ActivityExtraction,
        method="function_calling",
    )

    activities = extraction_model.invoke([
        SystemMessage(content="""
            You are a data extraction assistant.

            Extract tourist activities from the provided research.
            Follow the ActivityExtraction schema strictly.

            Never invent missing information.
            Use only information supported by the research.
            Use None for unavailable optional fields.
        """),
        HumanMessage(content=f"""
            Extract the activities from this research:

            {research_result.content}
        """) ])

    

    # Step 4: Search and download images individually.
    enriched_activities: list[Activity] = []

    '''url_extraction_model = research_model
    
    for index, activity in enumerate(activities.activities, start=1):
        logger.info(
            "Searching image %d/%d: %s",
            index,
            len(activities.activities),
            activity.name,
        )

        enriched_activity = _enrich_activity_image(
            activity=activity,
            extraction_model=url_extraction_model,
            destination=destination,
        )

        print(enriched_activity)
        enriched_activities.append(enriched_activity)'''

    print("activities.activities")
    print(activities.activities)
    return activities.activities #enriched_activities
