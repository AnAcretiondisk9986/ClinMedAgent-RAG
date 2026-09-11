from __future__ import annotations

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

# 缩写匹配时忽略的通用词，避免“医学统计学”误命中《医学免疫学》之类
BOOK_MENTION_STOPWORDS = {
    "医学", "临床", "基础", "现代", "实用", "中国", "实验", "大学",
    "教材", "图谱", "人卫", "科学", "技术", "规划", "高等", "院校",
}

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
    """needle 的字符是否按顺序出现在 haystack 中（如“组胚” → “组织学与胚胎学”）。"""
    iterator = iter(haystack)
    return all(char in iterator for char in needle)


def match_book(text: str, books: list[dict]) -> tuple[dict, str] | None:
    """从题干里识别教材，返回 (book, 命中的原文片段)。

    两级匹配：
      1) 强匹配：完整书名 / 书名主体 / 目录名作为子串出现，如同 “系统解剖学中骨的构造”；
      2) 弱匹配（仅当强匹配无结果）：以书名主体首字开头的 2–4 字缩写，
         如 “组胚” → 《组织学与胚胎学》，“系解” → 《系统解剖学》。
    多个教材同时命中时不强行路由（返回 None），交给全局检索排序。
    """
    text = str(text or "")
    if not text or not books:
        return None

    strong: list[tuple[int, dict, str]] = []
    for book in books:
        for name in _book_candidates(book):
            if name in text:
                strong.append((len(name), book, name))
    if strong:
        matched_ids = {item[1].get("id") for item in strong}
        if len(matched_ids) == 1:
            strong.sort(key=lambda item: -item[0])
            return strong[0][1], strong[0][2]
        return None  # 题干里出现多本教材，交给全局检索

    weak: dict[object, tuple[dict, str]] = {}
    for book in books:
        core = _core_title(str(book.get("title") or "")) or str(book.get("title") or "")
        if len(core) < 3:
            continue
        best_fragment = ""
        for length in (4, 3, 2):
            for index in range(0, len(text) - length + 1):
                fragment = text[index:index + length]
                if not re.fullmatch(r"[\u4e00-\u9fff]+", fragment):
                    continue
                if fragment in BOOK_MENTION_STOPWORDS:
                    continue
                if fragment[0] != core[0] or not _is_subsequence(fragment, core):
                    continue
                if len(fragment) > len(best_fragment):
                    best_fragment = fragment
            if best_fragment:
                break
        if best_fragment:
            weak[book.get("id")] = (book, best_fragment)
    if len(weak) == 1:
        book, fragment = next(iter(weak.values()))
        return book, fragment
    return None


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
