"""Small regression checks for the reference biography prompt distribution."""

from __future__ import annotations

import random

from tasks.multi_doc.data.biography import _format_question


def test_reference_question_templates_were_not_truncated():
    name = "Ada"
    observed = {
        _format_question("current_location", name, random.Random(seed))
        for seed in range(100)
    }
    assert len(observed) == 5
    assert (
        "Where is Ada residing at present? Respond with the answer only. Answer:"
        in observed
    )
