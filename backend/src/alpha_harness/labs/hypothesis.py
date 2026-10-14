"""Autonomous hypothesis generation, template expansion, and outcome-informed refinement.

Bridges template design, systematic search spaces, and genetic refinement:
- Expands single templates or concrete hypotheses into distinct, validated candidates.
- Aggregates multiple templates into broad, deduplicated candidate pools.
- Preserves full research provenance: originating template, parent hypothesis, generation
  method, and simulation configuration.
- Learns progressively from recorded experiment outcomes (via seed_score and validation
  stability) to prioritize subsequent generations without penalizing missing validation data.
"""

import math
import random
import uuid
from dataclasses import dataclass
from typing import Any

from ..brain.schemas import SimulationRequest, SimulationSettings, SimulationType
from .fastexpr import (
    MAX_FIELDS,
    MAX_OPERATORS,
    Node,
    ParseError,
    canonical_formula,
    data_fields,
    operator_count,
    parse,
    validate,
)
from .fastexpr import (
    render as render_node,
)
from .ga import (
    Market,
    Parent,
    child,
    mutate,
    seed_score,
)
from .template import PRESETS
from .template import values as parse_values


@dataclass(frozen=True, slots=True)
class HypothesisCandidate:
    """A generated alpha hypothesis candidate carrying complete research provenance."""

    id: str
    expression: str
    originating_template: str | None
    parent_id: str | None
    generation_method: str
    generation: int
    settings: dict[str, Any]
    request: SimulationRequest
    hypothesis_rationale: str | None = None


def _substitute(node: Node, params: dict[str, Any]) -> Node:
    """Recursively substitute variable nodes and operator calls from params."""
    if node.kind == "name" and node.value in params:
        val = params[node.value]
        return parse(str(val))
    if node.kind == "call" and node.value in params:
        op_name = params[node.value]
        return Node(
            "call",
            op_name,
            tuple(_substitute(a, params) for a in node.args),
            tuple((k, _substitute(a, params)) for k, a in node.kwargs),
        )
    return Node(
        node.kind,
        node.value,
        tuple(_substitute(a, params) for a in node.args),
        tuple((k, _substitute(a, params)) for k, a in node.kwargs),
    )


class HypothesisGenerator:
    """Generates, expands, and progressively refines alpha hypotheses."""

    def __init__(
        self,
        market: Market,
        default_settings: SimulationSettings | None = None,
    ) -> None:
        self.market = market
        self.default_settings = default_settings or SimulationSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="INDUSTRY",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
        )

    def _build_request(self, expression: str, settings_dict: dict[str, Any]) -> SimulationRequest:
        settings = self.default_settings.model_copy(
            update={
                "neutralization": settings_dict.get(
                    "neutralization", self.default_settings.neutralization
                ),
                "decay": int(settings_dict.get("decay", self.default_settings.decay)),
                "truncation": float(
                    settings_dict.get("truncation", self.default_settings.truncation)
                ),
                "universe": settings_dict.get("universe", self.default_settings.universe),
            }
        )
        return SimulationRequest(
            type=SimulationType.REGULAR,
            settings=settings,
            regular=expression,
        )

    def _auto_space(self, template_text: str) -> dict[str, Any]:
        """Infer variable choices from presets and market fields for template variables."""
        variables: list[dict[str, Any]] = []
        words = set(template_text.replace("(", " ").replace(")", " ").replace(",", " ").split())
        var_names = [w[1:] for w in words if w.startswith("$")]

        avail_fields = sorted(self.market.names - {"sector", "industry", "subindustry", "market"})
        if not avail_fields:
            avail_fields = ["close", "open", "high", "low", "volume", "vwap"]

        for v in var_names:
            v_lower = v.lower()
            if "field" in v_lower:
                variables.append(
                    {
                        "name": v,
                        "kind": "fields",
                        "fields": dict.fromkeys(avail_fields, "MATRIX"),
                    }
                )
            elif "lookback" in v_lower or v_lower == "d":
                vals = parse_values(PRESETS["lookback"])
                variables.append({"name": v, "kind": "values", "values": vals})
            elif "group" in v_lower or v_lower == "g":
                vals = parse_values(PRESETS["group"])
                variables.append({"name": v, "kind": "values", "values": vals})
            elif "op" in v_lower:
                vals = parse_values(PRESETS["ts_op"])
                variables.append({"name": v, "kind": "values", "values": vals})
            else:
                variables.append(
                    {
                        "name": v,
                        "kind": "values",
                        "values": ["1", "5", "10", "20"],
                    }
                )

        neuts = list(self.market.neutralizations or [self.default_settings.neutralization])
        return {
            "universes": [self.default_settings.universe],
            "neutralizations": neuts,
            "investability": ["none"],
            "variables": variables,
        }

    def generate_from_template(
        self,
        template_text: str,
        space: dict[str, Any] | None = None,
        count: int = 3,
        rng: random.Random | None = None,
    ) -> list[HypothesisCandidate]:
        """Generate at least `count` distinct, validated candidate hypotheses from a template."""
        rng = rng or random.Random()
        candidates: list[HypothesisCandidate] = []
        seen_expressions: set[str] = set()

        has_variables = "$" in template_text
        effective_space = space or (self._auto_space(template_text) if has_variables else None)

        max_attempts = max(50, count * 15)
        attempts = 0

        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            expression: str | None = None
            settings_dict = {
                "neutralization": rng.choice(
                    effective_space["neutralizations"]
                    if effective_space and effective_space.get("neutralizations")
                    else (self.market.neutralizations or [self.default_settings.neutralization])
                ),
                "decay": rng.choice([0, 3, 5, 7, 10]),
                "truncation": 0.08,
            }

            if has_variables and effective_space:
                drawn_params: dict[str, Any] = {}
                for var_def in effective_space["variables"]:
                    name = var_def["name"]
                    if var_def["kind"] == "fields":
                        field_choice = rng.choice(list(var_def["fields"].keys()))
                        drawn_params[f"${name}"] = field_choice
                    else:
                        drawn_params[f"${name}"] = rng.choice(var_def["values"])

                try:
                    tree = parse(template_text)
                    tree = _substitute(tree, drawn_params)
                    expression = render_node(tree)
                except ParseError, ValueError:
                    continue
            else:
                try:
                    tree = parse(template_text)
                    mutated_tree, mutated_settings = mutate(
                        rng, tree, settings_dict, self.market, rate=0.4, force=(attempts > 1)
                    )
                    expression = render_node(mutated_tree)
                    settings_dict = mutated_settings
                except ParseError, ValueError:
                    continue

            if not expression:
                continue

            canonical = canonical_formula(expression)
            if not canonical:
                continue
            try:
                parsed_tree = parse(canonical)
            except ParseError, ValueError:
                continue

            val_errors = validate(parsed_tree, self.market.table, self.market.names)
            if val_errors:
                continue
            if operator_count(parsed_tree) > MAX_OPERATORS:
                continue
            if len(data_fields(parsed_tree)) > MAX_FIELDS:
                continue

            neut = settings_dict.get("neutralization")
            decay = settings_dict.get("decay")
            dedup_key = f"{canonical}::{neut}::{decay}"
            if dedup_key in seen_expressions:
                continue
            seen_expressions.add(dedup_key)

            candidate_id = f"hyp-{uuid.uuid4().hex[:8]}"
            req = self._build_request(canonical, settings_dict)

            candidate = HypothesisCandidate(
                id=candidate_id,
                expression=canonical,
                originating_template=template_text,
                parent_id=None,
                generation_method="template_expansion" if has_variables else "structural_mutation",
                generation=0,
                settings=settings_dict,
                request=req,
                hypothesis_rationale=f"Expanded from template: {template_text[:60]}",
            )
            candidates.append(candidate)

        return candidates

    def expand_templates(
        self,
        templates: list[str],
        count_per_template: int = 3,
        rng: random.Random | None = None,
    ) -> list[HypothesisCandidate]:
        """Expand multiple templates into a unified, deduplicated candidate pool."""
        rng = rng or random.Random()
        all_candidates: list[HypothesisCandidate] = []
        seen_keys: set[str] = set()

        for tmpl in templates:
            tmpl_candidates = self.generate_from_template(tmpl, count=count_per_template, rng=rng)
            for cand in tmpl_candidates:
                neut = cand.settings.get("neutralization")
                decay = cand.settings.get("decay")
                key = f"{cand.expression}::{neut}::{decay}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_candidates.append(cand)

        return all_candidates

    def regenerate_generation(
        self,
        previous_candidates: list[HypothesisCandidate],
        outcomes: dict[str, dict[str, Any]],
        count: int = 3,
        rng: random.Random | None = None,
    ) -> list[HypothesisCandidate]:
        """Prioritize top candidates using experiment outcomes and breed the next generation."""
        rng = rng or random.Random()
        if not previous_candidates:
            return []

        scored_candidates: list[tuple[float, HypothesisCandidate]] = []
        for cand in previous_candidates:
            outcome = outcomes.get(cand.id, {})
            if not outcome:
                score = 1.0
            else:
                score = seed_score(outcome)
                if not math.isfinite(score):
                    score = -math.inf
            if score > -math.inf:
                scored_candidates.append((score, cand))

        if not scored_candidates:
            scored_candidates = [(1.0, c) for c in previous_candidates]

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [c for _, c in scored_candidates[: max(2, len(scored_candidates) // 2)]]

        parents: list[Parent] = []
        for cand in top_candidates:
            p = Parent.of(cand.id, cand.expression, cand.settings)
            if p is not None:
                parents.append(p)

        if not parents:
            return []

        next_generation_num = max(c.generation for c in previous_candidates) + 1
        new_candidates: list[HypothesisCandidate] = []
        seen_keys = {
            f"{c.expression}::{c.settings.get('neutralization')}::{c.settings.get('decay')}"
            for c in previous_candidates
        }

        attempts = 0
        max_attempts = max(40, count * 12)

        while len(new_candidates) < count and attempts < max_attempts:
            attempts += 1
            p_a = rng.choice(parents)
            p_b = rng.choice(parents)

            child_res = child(rng, p_a, p_b, self.market, rate=0.25)
            if child_res is None:
                continue

            c_tree, c_settings = child_res
            c_expr = canonical_formula(render_node(c_tree))
            if not c_expr:
                continue

            val_err = validate(c_tree, self.market.table, self.market.names)
            if val_err:
                continue
            if operator_count(c_tree) > MAX_OPERATORS:
                continue
            if len(data_fields(c_tree)) > MAX_FIELDS:
                continue

            neut = c_settings.get("neutralization")
            decay = c_settings.get("decay")
            dedup_key = f"{c_expr}::{neut}::{decay}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            c_id = f"hyp-{uuid.uuid4().hex[:8]}"
            req = self._build_request(c_expr, c_settings)

            parent_cand = next((c for c in previous_candidates if c.id == p_a.alpha_id), None)
            orig_tmpl = parent_cand.originating_template if parent_cand else None

            candidate = HypothesisCandidate(
                id=c_id,
                expression=c_expr,
                originating_template=orig_tmpl,
                parent_id=p_a.alpha_id,
                generation_method="genetic_refinement",
                generation=next_generation_num,
                settings=c_settings,
                request=req,
                hypothesis_rationale=(
                    f"Generation {next_generation_num} refined from parent {p_a.alpha_id}"
                ),
            )
            new_candidates.append(candidate)

        return new_candidates
