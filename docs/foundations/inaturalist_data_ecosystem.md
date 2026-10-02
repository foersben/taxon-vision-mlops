---
type: Concept
title: iNaturalist Data Ecosystem & Open Licensing
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Overview of iNaturalist API rate limits, AWS Open Data buckets, and CC licensing.
tags: [inaturalist, licensing, open-data]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# iNaturalist Data Ecosystem & Open Licensing

* **API Limits:** 60 requests per minute with a download cap of 5 GB per hour and 24 GB per day.
* **AWS Open Data:** Stored under `s3://inaturalist-open-data/` with open licensing metadata.
* **License Filtering:** Only CC0, CC-BY, and CC-BY-NC observations are admitted into training datasets. Images hosted on static domains without verified licenses are rejected.
