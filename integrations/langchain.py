"""Optional LangChain-style wrapper (no hard dependency).

Only imported if the user explicitly pulls it in. It exposes the mock backend
as a simple callable that an agent loop can use for decision scoring.
"""


class OlmoforgeScorer:
    """Wrap the mock backend so an agent can score candidate actions."""

    def __init__(self, skill=0.8, **kwargs):
        from olmoforge.backends.mock import MockBackend

        self.backend = MockBackend(skill=skill, **kwargs)

    def score(self, context, candidates):
        out = []
        for c in candidates:
            s = self.backend.score(context, c)
            out.append((c, s))
        return sorted(out, key=lambda x: -x[1])

    def choose(self, context, candidates):
        ranked = self.score(context, candidates)
        return ranked[0][0] if ranked else None
