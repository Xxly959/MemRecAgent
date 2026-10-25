"""有限明确句型的本次展示数量；不推断人物/品类，不存长期默认。"""

import re
from dataclasses import dataclass

from memrecagent.request_attributes import numbers_in_text

_NUMBER = r"(?:\d+|[零〇一二两三四五六七八九十百千万]+)"
_PAIR = re.compile(rf"(?P<number>{_NUMBER})\s*(?:款|部|台|个|件)")
_MARKER = re.compile(
    r"不要超过|不超过|别超过|不多于|至多|最多|推荐|展示|列出|挑选|看看|改为|改成|换成|再给|只要|仅要|就要|挑|选|找|看|给|来|要"
)
_PAST = re.compile(r"上次|之前|以前|曾经|过去|去年|昨天|前天")
_NEGATIVE = re.compile(r"不要|不想|不必|不用|无需|不是|不推荐|不看|不选|不找|不止|不只|不光|别|没")
_QUOTED = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"')
_SEPARATOR = re.compile(r"[，,。；;！？!?\n]")
_RANGE = re.compile(r"到|至|或|、|[~～—/-]")
_CLARIFY = (
    "请完整说明本次要推荐几款（1至10款）、给谁选以及其他要求；"
    "数量范围或多个不同数量暂不能合并处理。"
)


@dataclass(frozen=True)
class ResultCount:
    count: int | None = None
    spans: tuple[tuple[int, int], ...] = ()
    clarification: str | None = None


def resolve_result_count(text: str) -> ResultCount:
    """只在已接受的推荐请求使用；范围/超限保守澄清，返回原文字符区间。"""
    quoted = [m for m in _QUOTED.finditer(text) if _MARKER.search(m.group())]
    counts, spans = [], []
    ambiguous = quoted_only = negated = False
    for match in _PAIR.finditer(text):
        if any(q.start() <= match.start() < q.end() for q in quoted):
            quoted_only = True
            continue
        if re.match(r"\s*(?:[gG][bB]?|兆|斤|克|元)", text[match.end() :]):
            continue
        separators = [m.end() for m in _SEPARATOR.finditer(text[: match.start()])]
        start = separators[-1] if separators else 0
        following = _SEPARATOR.search(text, match.end())
        end = following.start() if following else len(text)
        prefix, suffix = text[start : match.start()], text[match.end() : end]
        markers = list(_MARKER.finditer(prefix))
        if markers:
            marker = markers[-1]
            before = prefix[max(0, marker.start() - 8) : marker.start()]
            between = prefix[marker.end() :]
            negative_context = before + (
                "" if marker.group() in {"不要超过", "别超过"} else marker.group()
            )
            # 体标记只绑定最近请求动词，不让“看过很多，请推荐两款”被过去背景吞掉。
            if _PAST.search(before) or re.match(r"\s*(?:过|了)", between):
                continue
            if _NEGATIVE.search(negative_context):
                negated = True
                continue
            if len(between) > 20:
                continue
        else:
            between = ""
            # 独立“3款/3款就行”能绑定已接受的推荐意图；“我有3台”不能。
            if prefix.strip(' “「『"') or not re.fullmatch(
                r"\s*(?:手机|商品|产品)?(?:就行|就好|就够|即可|够了)?[”」』\"]?\s*", suffix
            ):
                continue
        raw = match.group("number")
        span = (match.start(), match.end())
        spans.append(span)
        if (
            _RANGE.search(between)
            or (match.start() and text[match.start() - 1] in ".-−")
            or re.search(r"[一二两三四五六七八九]{2}", raw)
        ):
            ambiguous = True
            continue
        try:
            values = numbers_in_text(raw)
            if len(values) != 1:
                raise ValueError
            value = next(iter(values))
            if not value.is_integer() or not 1 <= value <= 10:
                raise ValueError
            counts.append(int(value))
        except (ValueError, OverflowError):
            ambiguous = True
    if ambiguous or len(set(counts)) > 1 or ((quoted_only or negated) and not counts):
        return ResultCount(None, tuple(spans), _CLARIFY)
    return ResultCount(counts[0], tuple(spans)) if counts else ResultCount()
