from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Common wording that carries little retrieval value in Chinese exam questions.
QUESTION_STOPWORDS = {
    "请", "简述", "简答", "说明", "叙述", "试述", "比较", "分析", "解释", "指出",
    "下列", "关于", "其中", "正确", "错误", "的是", "不正确", "属于", "主要", "哪些",
    "什么", "为什么", "如何", "有何", "具有", "是否", "以及", "分别", "以下", "选出",
    "描述", "特点", "作用", "组成", "结构", "位置", "关系", "包括", "可以", "能够",
    "的", "和", "与", "及", "或", "是", "有", "为", "中", "其", "容易", "发生", "导致", "因为", "由于", "通常",
}

# Connectives the stopword splitter can leave attached to a domain term, as
# in “肩关节由哪些结构组成” -> “肩关节由”. They are trimmed only at the end
# and only when at least two characters remain, so terms such as “自由基”
# and “并殖吸虫” stay intact.
TRAILING_CONNECTIVES = set("由使令把被将所该此那这并而且则")

# 缩写匹配时忽略的通用词表已被显式别名表（DEFAULT_BOOK_ALIASES）取代：
# 旧实现要在这里逐个屏蔽“医学/临床/实验”等词，仍然拦不住“系统/组织”这类
# 普通医学词；现在只认白名单，不再需要黑名单。

# 教材缩写白名单：缩写 → 能指代的书名子串（可多个）。
#
# 旧实现根据书名自动生成任意 2–4 字缩写，结果把普通医学词当成书名：
#   “神经系统的组成” → “系统” → 误限《系统解剖学》
#   “组织的分类”     → “组织” → 误限《组织学与胚胎学》
# 现在只认下面的显式别名；不在表里的缩写一律当作没有点名教材，保持全库检索。
DEFAULT_BOOK_ALIASES: dict[str, tuple[str, ...]] = {
    "系解": ("系统解剖学",),
    "局解": ("局部解剖学",),
    "神解": ("神经解剖学",),
    "组胚": ("组织学与胚胎学", "组织胚胎学"),
    "解胚": ("解剖学与胚胎学",),
    "生理": ("生理学",),
    "病生": ("病理生理学",),
    "生化": ("生物化学",),
    "病理": ("病理学",),
    "药理": ("药理学",),
    "微生": ("微生物学",),
    "免疫": ("免疫学",),
    "寄生": ("寄生虫学",),
    "诊断": ("诊断学",),
    "内科": ("内科学",),
    "外科": ("外科学",),
    "妇产": ("妇产科学",),
    "儿科": ("儿科学",),
}
ALIASES_FILE = "aliases.json"

# 缩写必须作为独立引用出现，否则“生理功能”“免疫应答”这类常见词会被当成书名。
SCOPE_MARKERS = ("里", "中", "内", "的", "这", "那", "该", "本", "书", "教材", "课")
REFERENCE_LEADS = ("请问", "依据", "按照", "参见", "参考", "根据", "翻到", "见")

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
# 低于这个置信度就不自动限定教材范围，改回全库检索（安全侧）
SCOPE_MIN_CONFIDENCE = "medium"


def load_book_aliases(base_dir: Path | str | None = None) -> dict[str, tuple[str, ...]]:
    """内置缩写表 + 可选自定义 ``aliases.json``（路径可用 MEDICAL_RAG_ALIASES 覆盖）。

    自定义项覆盖同名内置项，方便学生自己加教材：
    ``{"影像": ["医学影像学"], "口组": ["口腔组织病理学"]}``
    """
    aliases = {alias: tuple(targets) for alias, targets in DEFAULT_BOOK_ALIASES.items()}
    override = os.environ.get("MEDICAL_RAG_ALIASES")
    path = Path(override).expanduser() if override else (Path(base_dir) / ALIASES_FILE if base_dir else None)
    if path is None or not path.exists():
        return aliases
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return aliases
    if isinstance(data, dict):
        for alias, targets in data.items():
            if isinstance(targets, str):
                targets = [targets]
            if isinstance(targets, (list, tuple)) and targets:
                aliases[str(alias).strip()] = tuple(str(target) for target in targets)
    return aliases

QUESTION_PATTERNS: list[tuple[str, str]] = [
    ("choice", r"(?:^|[\n\s])\s*[A-DＡ-Ｄ][\.、．\)]"),
    ("fill_blank", r"_{2,}|（\s*）|\(\s*\)|填空"),
    ("definition", r"什么是|名词解释|解释一下|定义"),
    ("compare", r"比较|区别|异同|鉴别"),
    ("why", r"为什么|原因|机制|原理"),
]


def classify_question(question: str) -> str:
    for kind, pattern in QUESTION_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return kind
    return "short_answer"


def extract_options(question: str) -> list[dict[str, str]]:
    """Extract A-D options without trying to decide which option is correct."""
    pattern = re.compile(
        r"(?:^|[\n\s])([A-DＡ-Ｄ])[\.、．\)]\s*(.*?)(?=(?:[\n\s]+[A-DＡ-Ｄ][\.、．\)]\s*)|$)",
        flags=re.DOTALL,
    )
    matches = list(pattern.finditer(question))
    return [{"label": _normalise_option_label(m.group(1)), "text": m.group(2).strip()} for m in matches]


def _normalise_option_label(value: str) -> str:
    return value.translate(str.maketrans("ＡＢＣＤ", "ABCD"))


def _trim_trailing_connectives(term: str) -> str:
    while len(term) >= 3 and term[-1] in TRAILING_CONNECTIVES:
        term = term[:-1]
    return term


def extract_concepts(question: str) -> list[str]:
    """Extract multi-character Chinese terms and Latin/number terms.

    This is deliberately dependency-free. It is a recall helper, not a medical
    tokenizer; the answer layer can later replace it with a domain dictionary.
    """
    text = re.sub(r"[A-DＡ-Ｄ][\.、．\)]", " ", question)
    raw_terms = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z][A-Za-z0-9_-]{1,}|\d+(?:\.\d+)?", text)
    result: list[str] = []
    stopwords = sorted(QUESTION_STOPWORDS, key=len, reverse=True)
    for raw in raw_terms:
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            # Split a contiguous Chinese phrase around exam filler words. This
            # keeps domain terms such as “肩关节” while dropping “的/和/特点”.
            pieces = [raw]
            for stopword in stopwords:
                next_pieces: list[str] = []
                for piece in pieces:
                    next_pieces.extend(x for x in piece.split(stopword) if x)
                pieces = next_pieces
            candidates = [_trim_trailing_connectives(piece) for piece in pieces]
        else:
            candidates = [raw]
        for term in candidates:
            if term in QUESTION_STOPWORDS:
                continue
            if term not in result:
                result.append(term)
    return result


def build_search_queries(question: str) -> list[str]:
    """Build a small set of complementary queries for multi-pass retrieval."""
    question = re.sub(r"\s+", " ", question).strip()
    if not question:
        return []
    concepts = extract_concepts(question)
    queries = [question]
    if concepts:
        queries.append(" ".join(concepts))
        # Searching the first few concepts separately helps when OCR omitted a
        # word or when a question contains generic exam phrasing.
        queries.extend(concepts[:4])
    options = extract_options(question)
    for option in options:
        option_terms = extract_concepts(option["text"])
        if option_terms:
            queries.append(" ".join(option_terms[:4]))
    result: list[str] = []
    for query in queries:
        if query and query not in result:
            result.append(query)
    return result


def _core_title(title: str) -> str:
    """书名主体：去掉括号说明、版本号与出版方词，如“组织学与胚胎学 人卫 第10版” → “组织学与胚胎学”。"""
    core = re.sub(r"[（(][^）)]*[）)]", " ", str(title or ""))
    core = re.sub(r"第\s*[0-9０-９一二三四五六七八九十]+\s*版", " ", core)
    core = re.sub(r"(人民卫生出版社|人卫|高等教育出版社|高教|规划教材|教材)", " ", core)
    return re.sub(r"\s+", "", core).strip()


def _book_candidates(book: dict) -> list[str]:
    """书名候选：完整标题、书名主体、目录名（及其主体），用于题干中的强匹配。"""
    names: list[str] = []
    title = str(book.get("title") or "")
    for value in (title, _core_title(title)):
        value = value.strip()
        if len(value) >= 2 and value not in names:
            names.append(value)
    path = str(book.get("path") or "")
    if path:
        directory = Path(path).parent.name
        for value in (directory, _core_title(directory)):
            value = value.strip()
            if len(value) >= 2 and value not in names:
                names.append(value)
    return names


def _is_subsequence(needle: str, haystack: str) -> bool:
    """needle 的字符是否按顺序出现在 haystack 中。

    保留给测试与外部调用；**不再用于教材识别**（旧实现靠它把“系统”“组织”这类
    普通医学词当成书名缩写）。
    """
    iterator = iter(haystack)
    return all(char in iterator for char in needle)


def _aliases_for(book: dict, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    """这本书能被哪些显式缩写指代（缩写表按书名/目录名子串匹配）。"""
    candidates = _book_candidates(book)
    if not candidates:
        return []
    return [
        alias
        for alias, targets in aliases.items()
        if any(target in candidate for candidate in candidates for target in targets)
    ]


def _alias_is_standalone(text: str, index: int, length: int) -> bool:
    """缩写是否作为**独立引用**出现，而不是嵌在“生理功能”这类普通词里。

    以下算独立引用：
      - 后面紧跟范围标记：“组胚里…”“系解中…”“组胚的上皮组织”
      - 两侧都不是汉字：“《组胚》”“用 组胚 查”、行首或行尾
      - 前面是引用语：“请问组胚…”“依据系解…”
    其余情况（如“生理功能有哪些”）一律不算——宁可退回全库检索，也不能错误限域。
    """
    before = text[index - 1] if index > 0 else ""
    after = text[index + length] if index + length < len(text) else ""

    def is_han(char: str) -> bool:
        return bool(char) and bool(re.fullmatch(r"[\u4e00-\u9fff]", char))

    if after and after in SCOPE_MARKERS:
        return True
    if not is_han(before) and not is_han(after):
        return True
    prefix = text[:index]
    return any(prefix.endswith(lead) for lead in REFERENCE_LEADS)


@dataclass(frozen=True)
class BookMatch:
    """题干里识别到的教材及其置信度。"""

    book: dict
    mention: str
    confidence: str  # high（完整书名/主体/目录名）| medium（显式缩写）
    reason: str

    def as_dict(self) -> dict:
        return {
            "book_id": self.book.get("id"),
            "title": self.book.get("title"),
            "mention": self.mention,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def match_book_ex(
    text: str,
    books: list[dict],
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> BookMatch | None:
    """从题干里识别教材，返回带置信度的 :class:`BookMatch`。

    两级匹配：
      1) **强匹配**（high）：完整书名 / 书名主体 / 目录名作为子串出现，
         如“系统解剖学中骨的构造”；
      2) **缩写匹配**（medium）：只有 ``aliases`` 白名单里的缩写才算，
         且必须作为独立引用出现（见 :func:`_alias_is_standalone`）。
    多个教材同时命中时不强行路由（返回 None），交给全局检索排序。

    不再有低置信度的自动子序列匹配：匹配不到就是 None，由调用方全库检索。
    """
    text = str(text or "")
    if not text or not books:
        return None
    aliases = DEFAULT_BOOK_ALIASES if aliases is None else aliases

    strong: list[tuple[int, dict, str]] = []
    for book in books:
        for name in _book_candidates(book):
            if name in text:
                strong.append((len(name), book, name))
    if strong:
        matched_ids = {item[1].get("id") for item in strong}
        if len(matched_ids) == 1:
            strong.sort(key=lambda item: -item[0])
            _, book, name = strong[0]
            return BookMatch(book, name, "high", f"书名/目录名「{name}」完整出现")
        return None  # 题干里出现多本教材，交给全局检索

    alias_hits: dict[object, BookMatch] = {}
    for book in books:
        best = ""
        for alias in _aliases_for(book, aliases):
            index = text.find(alias)
            if index == -1 or not _alias_is_standalone(text, index, len(alias)):
                continue
            if len(alias) > len(best):
                best = alias
        if best:
            alias_hits[book.get("id")] = BookMatch(
                book, best, "medium", f"显式缩写「{best}」"
            )
    if len(alias_hits) == 1:
        return next(iter(alias_hits.values()))
    return None


def match_book(text: str, books: list[dict]) -> tuple[dict, str] | None:
    """兼容旧接口：只返回 (book, 命中片段)。新代码请用 :func:`match_book_ex`。"""
    match = match_book_ex(text, books)
    if match is None:
        return None
    return match.book, match.mention


def strip_book_mention(question: str, mention: str) -> str:
    """移除题干里独立引用的教材名（含后置虚词），便于概念抽取。

    只在教材名不是更长医学名词的一部分时才剥离：
    “组胚里上皮组织…” → “上皮组织…”；而“上皮组织有哪些分类”里的“组织”
    嵌在“上皮组织”中，保留原句，避免把概念拆坏。
    """
    if not mention:
        return question
    index = question.find(mention)
    if index == -1:
        return question
    before = question[index - 1] if index > 0 else ""
    after = question[index + len(mention)] if index + len(mention) < len(question) else ""

    def is_han(char: str) -> bool:
        return bool(char) and bool(re.fullmatch(r"[\u4e00-\u9fff]", char))

    if is_han(before) or (is_han(after) and after not in {"里", "中", "内", "的"}):
        return question  # 嵌在更长的词里，保留原句

    remainder = question[:index] + " " + question[index + len(mention):]
    remainder = re.sub(
        r"^[\s《》〈〉「」【】（）()，,、：:；;]*(?:里|中|内|的|这本|那本|该书|本书|教材)*[\s《》〈〉「」【】（）()，,、：:；;]*",
        "",
        remainder,
    )
    return remainder.strip() or question


@dataclass(frozen=True)
class QuestionPlan:
    question: str
    question_type: str
    concepts: list[str]
    options: list[dict[str, str]]
    queries: list[str]

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "question_type": self.question_type,
            "concepts": self.concepts,
            "options": self.options,
            "queries": self.queries,
        }


def plan_question(question: str) -> QuestionPlan:
    question = question.strip()
    return QuestionPlan(
        question=question,
        question_type=classify_question(question),
        concepts=extract_concepts(question),
        options=extract_options(question),
        queries=build_search_queries(question),
    )
