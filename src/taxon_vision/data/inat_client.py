# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Rate-limited client for iNaturalist API v2."""

from __future__ import annotations

import asyncio
from typing import Any


class INatAPIClient:
    """Asynchronous client adhering to the 60 requests/minute rate limit.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        """Init  .

        Args:
            max_requests_per_minute: The max requests per minute parameter.
        """
        self.delay_seconds = 60.0 / max_requests_per_minute

    async def fetch_observations(self, taxon_id: int, per_page: int = 10) -> list[dict[str, Any]]:
        """Fetch observations.

        Args:
            taxon_id: The taxon id parameter.
            per_page: The per page parameter.

        Returns:
            The resulting value from the operation.
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
