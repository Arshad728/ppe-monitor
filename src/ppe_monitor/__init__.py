"""Worker Safety (PPE) Monitoring System.

Package layout (one sub-package per stage of the pipeline, see the book's Chapter 14):

    ingestion/  reading live camera streams (Phase 0 probe, Phase 4 multi-stream readers)
    vision/     detection and tracking (Phases 1-2)
    rules/      PPE matching, restricted zones, event confirmation (Phases 2-3)
    backend/    database, API and alert workers (Phase 5)
    dashboard/  the safety officer's web dashboard (Phase 5)
    sim/        development tools: synthetic clips and the fake camera network (Phase 0)
"""

__version__ = "0.1.0"
