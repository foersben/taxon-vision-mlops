# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Rate-limited client for iNaturalist API v1."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class INatAPIClient:
    """Asynchronous client adhering to the 60 requests/minute rate limit.

    Attributes:
        delay_seconds: float - Delay between requests in seconds.
        base_url: str - Base URL for the iNaturalist API.
    """

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        """Initialize the client with rate limiting configuration.

        Args:
            max_requests_per_minute: Maximum allowed requests per minute to prevent API throttling.
        """
        self.delay_seconds = 60.0 / max_requests_per_minute
        self.base_url = "https://api.inaturalist.org/v1"

    async def fetch_observations(self, taxon_id: int, per_page: int = 10) -> list[dict[str, Any]]:
        """Fetch verified observations for a given taxon from the iNaturalist API.

        Args:
            taxon_id: Target taxon identifier.
            per_page: Number of observation records to return per page.

        Returns:
            List of raw observation dictionaries containing license and image URLs.
        """
        await asyncio.sleep(self.delay_seconds)

        url = f"{self.base_url}/observations"
        params: dict[str, str | int] = {
            "taxon_id": taxon_id,
            "per_page": per_page,
            "has[]": "photos",
            "quality_grade": "research",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

        results = []
        for obs in data.get("results", []):
            photos = obs.get("photos", [])
            if not photos:
                continue

            photo = photos[0]
            results.append(
                {
                    "observation_id": str(obs.get("id", "")),
                    "license_code": photo.get("license_code") or "",
                    "image_url": photo.get("url", "").replace("square", "medium"),
                    "attribution": photo.get("attribution", ""),
                    "license_url": "",
                    "is_open_dataset_verified": True,
                }
            )

        return results
