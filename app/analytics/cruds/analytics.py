from sqlalchemy.orm import Session
from sqlalchemy import func
from app.evaluation.models.evaluation import Evaluation


def get_analytics(db: Session):
    """Aggregate metrics for chart-friendly analytics payloads."""
    scatter_data = db.query(
        Evaluation.model,
        Evaluation.temperature,
        Evaluation.top_p,
        func.avg(Evaluation.lexical_diversity).label("avg_lexical_diversity"),
        func.avg(Evaluation.query_coverage).label("avg_query_coverage"),
        func.avg(Evaluation.flesch_kincaid_grade).label("avg_fk_grade"),
        func.avg(Evaluation.repetition_penalty).label("avg_repetition_penalty"),
        func.count(Evaluation.id).label("run_count"),
    ).group_by(
        Evaluation.model,
        Evaluation.temperature,
        Evaluation.top_p,
    ).all()

    model_comparison = db.query(
        Evaluation.model,
        func.avg(Evaluation.lexical_diversity).label("avg_lexical_diversity"),
        func.avg(Evaluation.query_coverage).label("avg_query_coverage"),
        func.avg(Evaluation.flesch_kincaid_grade).label("avg_fk_grade"),
        func.avg(Evaluation.repetition_penalty).label("avg_repetition_penalty"),
    ).group_by(Evaluation.model).all()

    overall = db.query(
        func.avg(Evaluation.lexical_diversity),
        func.avg(Evaluation.query_coverage),
        func.avg(Evaluation.flesch_kincaid_grade),
        func.avg(Evaluation.repetition_penalty),
    ).one()
    # Convert to dicts for manipulation
    scatter = [dict(row._mapping) for row in scatter_data]
    compare = [dict(row._mapping) for row in model_comparison]

    # Dataset-aware normalization (min-max per returned set)
    def _min_max(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return (0.0, 0.0)
        return (min(vals), max(vals))

    def _norm_high_is_better(x, vmin, vmax):
        rng = (vmax - vmin) if (vmax - vmin) != 0 else 1e-9
        return 100.0 * (x - vmin) / rng

    def _norm_low_is_better(x, vmin, vmax):
        rng = (vmax - vmin) if (vmax - vmin) != 0 else 1e-9
        return 100.0 * (vmax - x) / rng

    # Scatter normalization based on aggregated rows
    if scatter:
        ld_vals = [row["avg_lexical_diversity"] for row in scatter]
        qc_vals = [row["avg_query_coverage"] for row in scatter]
        fk_vals = [row["avg_fk_grade"] for row in scatter]
        rp_vals = [row["avg_repetition_penalty"] for row in scatter]

        # For percent metrics already in 0..100, do not normalize (pass through)
        # Keep normalization only for FK grade and repetition.
        ld_min, ld_max = _min_max(ld_vals)
        qc_min, qc_max = _min_max(qc_vals)
        fk_min, fk_max = _min_max(fk_vals)
        rp_min, rp_max = _min_max(rp_vals)

        for row in scatter:
            # Pass-through for percent metrics
            row["norm_lexical_diversity"] = round(row["avg_lexical_diversity"], 2)
            row["norm_query_coverage"] = round(row["avg_query_coverage"], 2)
            row["norm_fk_grade"] = round(_norm_high_is_better(row["avg_fk_grade"], fk_min, fk_max), 2)
            row["norm_repetition_penalty"] = round(_norm_low_is_better(row["avg_repetition_penalty"], rp_min, rp_max), 2)

    # Model comparison normalization separately on that set
    if compare:
        ld_vals = [row["avg_lexical_diversity"] for row in compare]
        qc_vals = [row["avg_query_coverage"] for row in compare]
        fk_vals = [row["avg_fk_grade"] for row in compare]
        rp_vals = [row["avg_repetition_penalty"] for row in compare]

        # Skip normalization for percent metrics
        ld_min, ld_max = _min_max(ld_vals)
        qc_min, qc_max = _min_max(qc_vals)
        fk_min, fk_max = _min_max(fk_vals)
        rp_min, rp_max = _min_max(rp_vals)

        for row in compare:
            row["norm_lexical_diversity"] = round(row["avg_lexical_diversity"], 2)
            row["norm_query_coverage"] = round(row["avg_query_coverage"], 2)
            row["norm_fk_grade"] = round(_norm_high_is_better(row["avg_fk_grade"], fk_min, fk_max), 2)
            row["norm_repetition_penalty"] = round(_norm_low_is_better(row["avg_repetition_penalty"], rp_min, rp_max), 2)

    return {
        "scatter_data": scatter,
        "model_comparison": compare,
        "kpi": {
            "overall_avg_lexical_diversity": round(overall[0], 2) if overall[0] else 0,
            "overall_avg_query_coverage": round(overall[1], 2) if overall[1] else 0,
            "overall_avg_fk_grade": round(overall[2], 2) if overall[2] else 0,
            "overall_avg_repetition_penalty": round(overall[3], 2) if overall[3] else 0,
        },
    }
