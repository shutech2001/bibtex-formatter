import argparse
import re
from pathlib import Path

import bibtexparser  # type: ignore
from bibtexparser.bwriter import BibTexWriter  # type: ignore

import string

# 固有名詞候補を手作業で前もって登録してもOK（特殊文字は \ を逃がす）
USER_PROPER_NOUNS = {
    "Cox", "P\\'olya-Gamma", "Bayesian", "Monte", "Carlo", "Markov", "Python", "Cox's",
}


def protect_proper_nouns(title: str) -> str:
    def needs_brace(word: str) -> bool:
        plain = word.strip(string.punctuation)
        if plain in USER_PROPER_NOUNS:
            return True
        return False

    parts, i = [], 0
    while i < len(title):
        if title[i] == "{":
            j = title.find("}", i) + 1
            parts.append(title[i:j])
            i = j
            continue

        j = i
        while j < len(title) and not title[j].isspace():
            j += 1
        token = title[i:j]

        if needs_brace(token):
            parts.append("{" + token + "}")
        else:
            parts.append(token)
        while j < len(title) and title[j].isspace():
            parts.append(title[j])
            j += 1
        i = j
    return "".join(parts)


INITIAL_RE = re.compile(r"^[A-Za-z]")


def _abbr_given(given: str) -> str:
    given = given.strip()
    if not given:
        return ""
    m = re.search(r"[A-Za-z]", given)
    if not m:
        return given
    idx = m.start()
    head = given[:idx]
    core = given[idx:]
    parts = re.split(r"[-\s]+", core)
    initials = []
    for p in parts:
        if INITIAL_RE.match(p):
            initials.append(p[0] + ".")
    return head + "-".join(initials)


def normalize_author_field(author_field: str) -> str:
    normalized_authors = []
    for raw_author in re.split(r"\s+and\s+", author_field):
        if "," in raw_author:
            last, given = [p.strip() for p in raw_author.split(",", 1)]
        else:
            parts = raw_author.strip().split()
            last = parts[-1]
            given = " ".join(parts[:-1])
        if given:
            tokens = given.split()
            initials = "~".join(_abbr_given(tok) for tok in tokens if tok.strip())
        else:
            initials = ""
        normalized_authors.append(f"{last}, {initials}")
    return " and ".join(normalized_authors)


# name of journal
SMALL = {"a", "an", "and", "as", "at", "but", "by", "for",
         "in", "nor", "of", "on", "or", "per", "the", "to", "vs", "via"}

ACRONYMS = {
    "ACM", "SIGKDD",
}


def title_case(text: str) -> str:
    words = re.split(r"(\s+|-)", text)
    result = []
    for idx, w in enumerate(words):
        if not w.strip() or re.match(r"\s+|-", w):
            result.append(w)
            continue

        if w.upper() in ACRONYMS:
            result.append(w.upper())
            continue

        if w[0].isdigit():
            result.append(w)
            continue

        lw = w.lower()
        if idx == 0 or lw not in SMALL:
            m = re.search(r"[A-Za-z]", w)
            if m:
                pos = m.start()
                w = w[:pos] + w[pos].upper() + w[pos+1:].lower()
            result.append(w)
        else:
            result.append(lw)
    return "".join(result)


# pages
def normalize_pages(p: str) -> str:
    s = p.strip()
    # すでに "--" が含まれていたら、そのまま返す
    if "--" in s:
        return s
    return re.sub(r"\s*[--]\s*", "--", s)


ARTICLE_KEEP = ["ID", "author", "title", "journal", "volume", "number", "pages", "year"]
BOOK_KEEP = ["ID", "author", "editor", "title", "publisher", "year"]
INPROC_KEEP = ["ID", "author", "title", "booktitle", "pages", "year"]
OTHER_KEEP = ["ID", "author", "title", "year"]


def clean_entry(entry):
    if "author" in entry:
        entry["author"] = normalize_author_field(entry["author"])

    if "title" in entry:
        entry["title"] = protect_proper_nouns(entry["title"])

    for field in ["journal", 'booktitle']:
        if field in entry:
            entry[field] = title_case(entry[field])

    if "pages" in entry:
        entry["pages"] = normalize_pages(entry["pages"])

    if "arxiv" in entry.get("journal", "").lower():
        entry["ENTRYTYPE"] = "article"
        entry["journal"] = "arXiv"

    entry_type = entry.get("ENTRYTYPE", "").lower()
    if entry_type == "article":
        keep = ARTICLE_KEEP
    elif entry_type == "book":
        keep = BOOK_KEEP
    elif entry_type == "inproceedings":
        keep = INPROC_KEEP
    else:
        keep = OTHER_KEEP

    for k in list(entry.keys()):
        if k not in keep and k != "ENTRYTYPE":
            del entry[k]

    ordered = {}
    for k in ["ENTRYTYPE", "ID"] + keep:
        if k in entry:
            ordered[k] = entry[k]
    return ordered


def main(infile: Path, outfile: Path) -> None:
    with infile.open(encoding="utf-8") as f:
        db = bibtexparser.load(f)

    # print(db.entries)
    # print([e["ID"] for e in db.entries])
    db.entries = [clean_entry(e) for e in db.entries]
    # print(db.entries)

    writer = BibTexWriter()
    writer.indent = "    "
    writer.order_entries_by = ("ID",)
    writer.comma_first = False
    writer.align_values = True

    with outfile.open("w", encoding="utf-8") as f:
        f.write(writer.write(db))

    print(f"[+] Cleaned .bib written to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BibTex formatter")
    parser.add_argument(
        "bibfile", type=Path, help="Input .bib file",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output .bib file (default: <infile>_clean.bib)"
    )
    args = parser.parse_args()

    out = args.output or args.bibfile.with_stem(args.bibfile.stem + "_clean")
    main(args.bibfile, out)
