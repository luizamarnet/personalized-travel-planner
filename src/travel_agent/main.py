"""Application entry point."""

import logging
import os

from dotenv import load_dotenv

from travel_agent.ui import build_app, MOBILE_SCROLL_CSS


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    app = build_app()
    
    app.queue().launch()
    '''app.launch(
        share=False,
        server_name="0.0.0.0",
    )'''
    '''app.launch(
        #share=True,
        server_name="0.0.0.0",
        css=MOBILE_SCROLL_CSS,
    )'''


if __name__ == "__main__":
    main()
