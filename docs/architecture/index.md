# Chapter 3: Software & Config Architecture

This chapter details the codebase design, package hierarchy, and configuration engine of the TaxonVision-MLOps platform.

The system is engineered as a clean 4-tier layered architecture that eliminates dynamic reflection (`hasattr`/`getattr`), prevents namespace pollution during CLI execution, enforces strict cognitive complexity bounds ($< 15$), and centralizes all operational settings into typed Pydantic models.

## Thematic Sections

* **[Repository & Config Encyclopedia](repository_and_config_encyclopedia.md):** Deep-dive into root `params.yaml`, `config/default_config.yaml`, taxonomic catalogs, and environment variable overrides.
* **[Package & Call Graph](package_and_call_graph.md):** Architectural map of the 10 core subpackages, layered dependency constraints, and circular dependency prevention.
* **[Codebase Explorer](codebase_explorer.md):** Module-by-module navigation index linking source files to domain roles and targeted unit tests.
