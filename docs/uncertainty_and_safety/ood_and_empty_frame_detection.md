---
type: Specification
title: Out-Of-Distribution & Empty Frame Detection
status: stable
stale_after: "2027-01-01T00:00:00Z"
version: 1.0
description: Energy-based scoring for identifying non-organism and uninformative images.
tags: [ood, energy-score, safety]
generated: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
verified: {by: process:scaffold-init, at: "2026-10-02T10:00:00Z"}
---

# Out-Of-Distribution & Empty Frame Detection

The system computes Helmholtz free energy over model logits:

$$E(x; T) = -T \cdot \log \sum_{y \in \mathcal{Y}} \exp\left( \frac{f_y(x)}{T} \right)$$

Observations where $E(x) > \tau_{\text{energy}}$ indicate absent organisms or extreme noise, triggering automated human referral.
