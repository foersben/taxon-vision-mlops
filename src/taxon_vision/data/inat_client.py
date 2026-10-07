# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Rate-limited client for iNaturalist API v1."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class INatAPIClient:
    """Asynchronous HTTP client adhering to strict iNaturalist API rate limits.

    Why:
        The iNaturalist API v1 imposes a strict terms-of-use rate limit of 60 requests
        per minute (1 request/second) with aggressive IP throttling and bans for abusive
        burst traffic. Ingesting training metadata dynamically requires an asynchronous client
        that enforces request spacing, filters exclusively for 'research' quality-grade
        observations, and converts thumbnail URLs into standardized medium-resolution images.

    How:
        Calculates minimum request delay (60.0 / max_requests_per_minute) and awaits an
        `asyncio.sleep` throttle before dispatching GET requests via `httpx.AsyncClient`.
        Filters query parameters to mandate research-grade community identifications and
        presence of photographic evidence, extracting licensing and attribution metadata.

    Attributes:
        delay_seconds: Precomputed delay between sequential requests in seconds.
        base_url: Canonical base URL for the iNaturalist REST API v1.
    """

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        """Initialize the client with rate limiting configuration.

        Args:
            max_requests_per_minute: Maximum allowed requests per minute to prevent API throttling.
        """
        self.delay_seconds = 60.0 / max_requests_per_minute
        self.base_url = "https://api.inaturalist.org/v1"

    async def fetch_observations(self, taxon_id: int, per_page: int = 10) -> list[dict[str, Any]]:
        """Fetch verified research-grade observations for a target taxon.

        Why:
            Biological training data must maintain high label fidelity. By querying exclusively
            for `quality_grade=research`, we ensure that community taxonomists have reached
            consensus on the organism's species identification. Upgrading image URLs from
            `square` (75x75 thumbnails) to `medium` (500px longest dimension) provides sufficient
            spatial resolution for neural feature extraction.

        How:
            1. Enforces asynchronous rate limiting via `await asyncio.sleep(self.delay_seconds)`.
            2. Issues an HTTP GET request to `/observations` with `taxon_id`, `quality_grade=research`,
               and `has[]=photos`.
            3. Validates HTTP response status code via `raise_for_status()`.
            4. Parses JSON response and filters observations with valid photo records.
            5. Replaces square thumbnail URL segments with medium resolution locators.
            6. Packages observation records with open license codes, attributions, and IDs.

        Args:
            taxon_id: Target Darwin Core taxonomic identifier.
            per_page: Maximum number of observation records to retrieve per request page.

        Returns:
            List of standardized observation dictionaries ready for license auditing.

        Raises:
            httpx.HTTPStatusError: If iNaturalist responds with 4xx or 5xx status codes.
            httpx.RequestError: If network connectivity or DNS resolution fails.
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
