"""澄清时保留有原文支撑的实体写法，提交后才规范化键。"""

import pytest
from test_memory_extraction import subject
from test_memory_interaction import after, confirmation, model, turn, value
from test_memory_interaction_extraction import quote

from memrecagent.entity_aliases import install_entity_aliases
from memrecagent.memory_interaction import ConversationMemorySession
from memrecagent.memory_store import JsonlMemoryStore


@pytest.mark.parametrize(
    "category,canonical,alias",
    [
        ("phone", "品牌A", "BrandA"),
        ("musical_instruments", "ASIN-EXACT-001", "Concert Harp"),
    ],
)
def test_subject_confirmation_does_not_replace_quoted_alias_with_unquoted_id(
    tmp_path, category, canonical, alias
):
    install_entity_aliases(
        tmp_path,
        dict(
            schema_version=1,
            entries=[
                dict(
                    category=category,
                    canonical=canonical,
                    aliases=[alias],
                    source="synthetic:catalog",
                )
            ],
        ),
    )
    store = JsonlMemoryStore(tmp_path)
    host = ConversationMemorySession(store, user_id="u", session_id="s", category=category)
    t1 = turn(1, f"有人长期喜欢{alias}。我还没说这个人是我还是叔叔。")
    candidate = value(t1, entity=alias, relation="prefer")
    row = candidate["candidate"]["assertions"][0]
    row.update(
        category=category, subject=subject("unknown"), evidence=[quote(t1, f"有人长期喜欢{alias}")]
    )
    candidate["candidate"]["unresolved"] = [dict(field="subject", evidence=[quote(t1, "有人")])]
    first = host.prepare_user(t1, complete=model(candidate), as_of=after(t1))
    assert first.response_action == "clarify"
    assert store.list_memories("u", as_of=after(t1)) == ()
    q = turn(2, first.memory_preview.proposal.decision.clarification, "assistant")
    host.observe_assistant(q)
    t3 = turn(3, "这个人是我，请记住我的长期偏好。")
    final = host.prepare_user(t3, complete=model(confirmation(t3, q)), as_of=after(t3))
    host.commit(final.ticket_id, as_of=after(t3))
    saved = store.list_memories("u", as_of=after(t3))
    assert len(saved) == 1 and saved[0].entity == canonical
    assert alias in saved[0].evidence_text and canonical not in saved[0].evidence_text
    assert t3.content in saved[0].evidence_text
