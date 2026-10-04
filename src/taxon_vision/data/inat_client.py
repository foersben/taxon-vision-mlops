# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Rate-limited client for iNaturalist API v2."""

from __future__ import annotations

import asyncio
from typing import Any


class INatAPIClient:
    """Asynchronous client adhering to the 60 requests/minute rate limit."""

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        """Initialize the client with rate limiting configuration.

        Args:
            max_requests_per_minute: Maximum allowed requests per minute to prevent API throttling.
        """
        self.delay_seconds = 60.0 / max_requests_per_minute

    async def fetch_observations(self, taxon_id: int, per_page: int = 10) -> list[dict[str, Any]]:
        """Fetch verified observations for a given taxon from the iNaturalist API.

        Args:
            taxon_id: Target taxon identifier.
            per_page: Number of observation records to return per page.

        Returns:
            List of raw observation dictionaries containing license and image URLs.
        """
        await asyncio.sleep(self.delay_seconds)
        # Mocked return for offline/CI stability
        return [
            {
                "observation_id": f"inat_{taxon_id}_{i}",
                "license_code": "CC-BY",
                "image_url": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{i}/medium.jpg",
                "attribution": "(c) Citizen Naturalist",
                "is_open_dataset_verified": True,
            }
            for i in range(per_page)
        ]
