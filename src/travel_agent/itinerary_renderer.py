
"""Render an itinerary as a user-friendly HTML travel plan."""

from datetime import date, datetime
from html import escape
from urllib.parse import urlparse

from travel_agent.schemas import Itinerary


ITEM_STYLES = {
    "activity": ("📍", "activity"),
    "meal": ("🍽️", "meal"),
    "transport": ("🚶", "transport"),
    "free_time": ("☕", "free-time"),
    "accommodation": ("🏨", "accommodation"),
}


def _safe_text(value) -> str:
    """Return escaped text, or an empty string for missing values."""
    if value is None:
        return ""
    return escape(str(value))


def _safe_url(value) -> str | None:
    """Allow only HTTP and HTTPS links."""
    if not value:
        return None

    parsed = urlparse(str(value))
    if parsed.scheme not in {"http", "https"}:
        return None

    return escape(str(value), quote=True)


def _format_date(value) -> str:
    if isinstance(value, str):
        value = date.fromisoformat(value[:10])
    return value.strftime("%A, %d %B %Y")


def _format_time(value) -> str:
    if not value:
        return ""

    if isinstance(value, str):
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                value = datetime.strptime(value, fmt).time()
                break
            except ValueError:
                continue

    return value.strftime("%H:%M")


def _render_item(item) -> str:
    item_type = getattr(item.item_type, "value", item.item_type)
    item_type = str(item_type).lower()

    icon, css_class = ITEM_STYLES.get(
        item_type, ("📌", "activity")
    )

    start = _format_time(item.start_time)
    end = _format_time(item.end_time)

    details = []

    if item.location:
        details.append(
            f"<span>📍 {_safe_text(item.location)}</span>"
        )

    if item.transport_from and item.transport_to:
        details.append(
            f"<span>↗ {_safe_text(item.transport_from)} "
            f"→ {_safe_text(item.transport_to)}</span>"
        )

    if item.duration_minutes:
        details.append(
            f"<span>⏱️ {item.duration_minutes} min</span>"
        )

    if item.transport_mode:
        details.append(
            f"<span>🚍 {_safe_text(item.transport_mode)}</span>"
        )

    highlights = ""
    if item.highlights:
        highlights = "<ul>" + "".join(
            f"<li>{_safe_text(point)}</li>"
            for point in item.highlights
        ) + "</ul>"

    notes = ""
    if item.notes:
        notes = (
            '<div class="notes"><strong>Notes</strong><ul>'
            + "".join(
                f"<li>{_safe_text(note)}</li>"
                for note in item.notes
            )
            + "</ul></div>"
        )

    links = []
    source_url = _safe_url(item.source_url)
    booking_url = _safe_url(item.booking_url)

    if source_url:
        links.append(
            f'<a href="{source_url}" target="_blank" '
            'rel="noopener noreferrer">Official source ↗</a>'
        )

    if booking_url:
        links.append(
            f'<a href="{booking_url}" target="_blank" '
            'rel="noopener noreferrer">Booking ↗</a>'
        )

    links_html = (
        '<div class="links">' + " · ".join(links) + "</div>"
        if links else ""
    )

    meal = item.meal_details
    meal_info = ""

    if meal:
        restaurant = meal.restaurant
        if restaurant:
            meal_info = (
                f'<p class="restaurant">Restaurant: '
                f'{_safe_text(restaurant.name)}</p>'
            )
        else:
            meal_info = (
                '<p class="pending">Restaurant suggestions '
                'have not been added yet.</p>'
            )

    cost = ""
    if item.estimated_cost_per_person is not None:
        currency = _safe_text(item.currency or "")
        cost = (
            f'<p class="cost">Estimated cost per person: '
            f'{currency} {_safe_text(item.estimated_cost_per_person)}</p>'
        )

    return f"""
    <article class="timeline-item {css_class}">
        <div class="timeline-time">
            <strong>{_safe_text(start)}</strong>
            <span>{_safe_text(end)}</span>
        </div>

        <div class="timeline-content">
            <div class="item-heading">
                <span class="item-icon">{icon}</span>
                <h3>{_safe_text(item.title)}</h3>
            </div>

            <p>{_safe_text(item.description)}</p>

            <div class="item-details">
                {"".join(details)}
            </div>

            {highlights}
            {meal_info}
            {cost}
            {notes}
            {links_html}
        </div>
    </article>
    """


def render_itinerary_html(itinerary: Itinerary) -> str:
    """Convert an Itinerary object into a styled HTML page."""

    days_html = []

    for day_number, day in enumerate(itinerary.days, start=1):
        items_html = "".join(
            _render_item(item) for item in day.items
        )

        summary = (
            f'<p class="day-summary">{_safe_text(day.summary)}</p>'
            if day.summary else ""
        )

        days_html.append(f"""
        <section class="day-card">
            <div class="day-header">
                <div>
                    <span class="day-label">DAY {day_number}</span>
                    <h2>{_safe_text(_format_date(day.date))}</h2>
                    {summary}
                </div>
            </div>

            <div class="timeline">
                {items_html}
            </div>
        </section>
        """)

    start = _format_date(itinerary.start_date)
    end = _format_date(itinerary.end_date)

    style = """
        <style>
            .travel-plan {
                font-family: Inter, system-ui, sans-serif;
                color: var(--body-text-color);
                max-width: 1000px;
                
                margin: 0 auto;
            }

            .plan-header {
                padding: 32px;
                margin-bottom: 28px;
                border-radius: 20px;
                background: var(--block-background-fill);
                border: 1px solid var(--border-color-primary);
            }

            .eyebrow, .day-label {
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.5px;
                color: var(--body-text-color-subdued);
            }

            .plan-header h1 {
                font-size: 30px;
                margin: 12px 0;
            }

            .timezone {
                font-size: 12px;
                color: var(--body-text-color-subdued);
            }

            .day-card {
                padding: 24px;
                margin-bottom: 24px;
                border-radius: 18px;
                background: var(--block-background-fill);
                border: 1px solid var(--border-color-primary);
            }

            .day-header {
                margin-bottom: 24px;
            }

            .day-header h2 {
                font-size: 21px;
                margin: 6px 0;
            }

            .day-summary {
                color: var(--body-text-color-subdued);
                margin-top: 6px;
            }

            .timeline {
                border-left: 2px solid var(--border-color-primary);
                margin-left: 55px;
            }

            .timeline-item {
                position: relative;
                display: grid;
                grid-template-columns: 1fr;
                margin: 0 0 24px 0;
                padding-left: 24px;
            }

            .timeline-item::before {
                content: "";
                position: absolute;
                left: -7px;
                top: 5px;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: var(--timeline-color, #64748b);
            }

            .activity { --timeline-color: #0d9488; }
            .meal { --timeline-color: #d97706; }
            .transport { --timeline-color: #64748b; }
            .free-time { --timeline-color: #8b5cf6; }
            .accommodation { --timeline-color: #2563eb; }

            .timeline-time {
                display: flex;
                gap: 10px;
                align-items: baseline;
                margin-bottom: 8px;
                font-size: 13px;
            }

            .timeline-time span {
                color: var(--body-text-color-subdued);
            }

            .timeline-content {
                padding: 18px;
                border-radius: 14px;
                background: var(--background-fill-secondary);
            }

            .item-heading {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .item-heading h3 {
                font-size: 17px;
                margin: 0;
            }

            .item-icon { font-size: 20px; }

            .timeline-content > p {
                line-height: 1.6;
            }

            .item-details {
                display: flex;
                flex-wrap: wrap;
                gap: 8px 16px;
                font-size: 12px;
                color: var(--body-text-color-subdued);
                margin-top: 12px;
            }

            .timeline-content ul {
                padding-left: 20px;
                line-height: 1.7;
            }

            .notes, .pending {
                font-size: 13px;
                color: var(--body-text-color-subdued);
            }

            .links {
                margin-top: 14px;
                font-size: 13px;
            }

            .links a {
                color: var(--link-text-color);
                text-decoration: none;
                font-weight: 600;
            }

            .restaurant, .cost {
                font-size: 13px;
                font-weight: 600;
            }

            @media (min-width: 650px) {
                .timeline-item {
                    grid-template-columns: 1fr;
                }

                .timeline-time {
                    position: absolute;
                    right: calc(100% + 16px);
                    top: 0;
                    width: 65px;
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                    text-align: right;
                }
            }

            @media (max-width: 500px) {
                .plan-header, .day-card { padding: 18px; }
                .timeline { margin-left: 8px; }
                .timeline-item { padding-left: 18px; }
                .timeline-time {
                    flex-direction: row;
                    margin-bottom: 8px;
                }
            }
        </style>
    """


    return f"""
    <div class="travel-plan">
        <header class="plan-header">
            <span class="eyebrow">YOUR PERSONAL TRAVEL PLAN</span>
            <h1>✈️ {_safe_text(itinerary.destination)}</h1>
            <p>{_safe_text(start)} – {_safe_text(end)}</p>
            <span class="timezone">
                Timezone: {_safe_text(itinerary.timezone)}
            </span>
        </header>

        <div class="days-container">
            {"".join(days_html)}
        </div>
    </div>

    {style}
    """
