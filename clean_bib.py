import argparse
import json
import re
from pathlib import Path
import string
from typing import Dict

import bibtexparser  # type: ignore
from bibtexparser.bwriter import BibTexWriter  # type: ignore


def protect_proper_nouns(title: str) -> str:
    """Protect proper nouns in an input by enclosing them in '{}'.
    Proper nouns are identified based on user specified list.

    Args:
        title (str): string where proper nouns need to be protected.

    Returns:
        str: string with proper nouns enclosed in '{}'
    """
    def needs_brace(word: str) -> bool:
        """Check whether '{}' is needed

        Args:
            word (str)

        Returns:
            bool: 'True' if needed, otherwise 'False'
        """
        plain = word.strip(string.punctuation)
        lower = plain.lower()
        # match with USER_SPECIFIED_TITLE
        if plain in USER_SPECIFIED_TITLE:
            return True
        # match with USER_SPECIFIED_TITLE + "'s" (not in EXCLUDE_APOS_S)
        if lower.endswith("'s"):
            base = lower[:-2]
            if (base in PROPER_LOWER_MAP) and (base not in EXCLUDE_APOS_S):
                return True
        return False

    parts, i = [], 0
    while i < len(title):
        if title[i] == "{":
            j = i + 1
            brace = 1
            while j < len(title) and brace:
                if title[j] == "{":
                    brace += 1
                elif title[j] == "}":
                    brace -= 1
                j += 1
            block = title[i+1:j-1]
            # remove '{}' when exclude-mode and not need braces
            if EXCLUDE_BRACE and not needs_brace(block):
                parts.append(protect_proper_nouns(block))
            else:
                parts.append("{" + block + "}")
            i = j
            continue

        # extract the continuous portion of text outside '{}'
        j = i
        while j < len(title) and not title[j].isspace():
            j += 1
        token = title[i:j]

        if needs_brace(token):
            # whether the word ends with "'s", and protect it
            plain = token.strip(string.punctuation)
            lower = plain.lower()
            if plain in USER_SPECIFIED_TITLE:
                proper = plain
            else:
                base = lower[:-2]
                proper = PROPER_LOWER_MAP[base] + "'s"
            parts.append("{" + proper + "}")
        else:
            parts.append(token)

        while j < len(title) and title[j].isspace():
            parts.append(title[j])
            j += 1
        i = j

    return "".join(parts)


# modules for formatting author
def _abbr_given(given: str) -> str:
    """Abbreviate the input by extracting the initials of each part of the name.

    Args:
        given (str): input name, which may include multiple parts separated by spaces or hyphens.

    Returns:
        str: abbreviated version of the input name
    """
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
    """Normalize the author field in a BibTeX entry.

    Args:
        author_field (str): string containing the author names

    Returns:
        str: string where each author's name is formatted as "Last, Initials"
    """
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


# module for formatting pages
def normalize_pages(p: str) -> str:
    """Normalize page range format

    Args:
        p (str): string representing a page range (e.g., 10 -- 20, 10-20)

    Returns:
        str: formatted page range
    """
    s = p.strip()
    # if '--' is already present, return it as is.
    if "--" in s:
        return s
    return re.sub(r"\s*[--]\s*", "--", s)


# modules for formatting title
def title_case(text: str) -> str:
    """Convert input to title case with specific rules for capitalizing certain words and preserving proper nouns.

    Args:
        text (str): string to be converted to title case

    Returns:
        str: converted string
    """
    words = re.split(r"(\s+|-)", text)
    result = []
    after_colon = False

    for idx, w in enumerate(words):
        if not w.strip() or re.match(r"\s+|-", w):
            result.append(w)
            if ":" in w:
                after_colon = True
            continue

        lower_w = w.lower().strip(string.punctuation)

        # check proper noun end with "'s", and included EXCLUDE_APOS_S
        if lower_w.endswith("'s"):
            base = lower_w[:-2]
            if base in PROPER_LOWER_MAP and base not in EXCLUDE_APOS_S:
                result.append(PROPER_LOWER_MAP[base] + "'s")
                after_colon = w.endswith(":")
                continue

        # check proper noun
        if w.strip(string.punctuation) in USER_SPECIFIED_TITLE:
            result.append(w.strip(string.punctuation))
            after_colon = w.endswith(":")
            continue

        # check acronyms
        if w.upper() in ACRONYMS:
            result.append(w.upper())
            after_colon = w.endswith(":")
            continue

        # check length of character is equal to 1, or upper case in original
        if len(w) == 1 and w.isalpha() and w.isupper():
            result.append(w)
            after_colon = w.endswith(":")
            continue

        # check whether the initial character is digit
        if w[0].isdigit():
            result.append(w)
            after_colon = w.endswith(":")
            continue

        lw = w.lower()
        # capitalize the first word and any word immediately after ':'
        if idx == 0 or after_colon:
            m = re.search(r"[A-Za-z]", w)
            if m:
                pos = m.start()
                new_w = w[:pos] + w[pos].upper() + w[pos+1:].lower()
            else:
                new_w = w
            result.append(new_w)
            after_colon = w.endswith(":")
            continue

        # check included LOWER
        if lw in LOWER:
            result.append(lw)
            after_colon = w.endswith(":")
            continue

        # otherwise, Aaaa
        m = re.search(r"[A-Za-z]", w)
        if m:
            pos = m.start()
            new_w = w[:pos] + w[pos].upper() + w[pos+1:].lower()
        else:
            new_w = w
        result.append(new_w)
        after_colon = w.endswith(":")

    return "".join(result)


def book_title_case(text: str) -> str:
    """Convert input to title case, with special rules for book titles

    Args:
        text (str): string to be converted to title case

    Returns:
        str: converted string
    """
    words = re.split(r"(\s+|-)", text)
    result = []
    word_pos = 0
    after_colon = False

    for w in words:
        if not w.strip() or re.match(r"\s+|-", w):
            result.append(w)
            if ":" in w:
                after_colon = True
            continue

        # keep word in {}
        if w.startswith("{") and w.endswith("}"):
            result.append(w)
            word_pos += 1
            after_colon = w.endswith(":")
            continue

        # check acronyms
        if w.upper() in ACRONYMS:
            result.append(w.upper())
            word_pos += 1
            after_colon = w.endswith(":")
            continue

        # check whether the initial character is digit
        if w[0].isdigit():
            result.append(w)
            word_pos += 1
            after_colon = w.endswith(":")
            continue

        # check length of character is equal to 1, or upper case in original
        if len(w) == 1 and w.isalpha() and w.isupper():
            result.append(w)
            word_pos += 1
            after_colon = w.endswith(":")
            continue

        # extract core by splitting signal
        prefix = ""
        suffix = ""
        core = w

        m1 = re.match(r"^(\W+)", core)
        if m1:
            prefix = m1.group(1)
            core = core[len(prefix):]

        m2 = re.match(r"^(.*?)(\W+)$", core)
        if m2:
            core = m2.group(1)
            suffix = m2.group(2)

        lw = core.lower()

        # capitalize the first word and any word immediately after ':'
        if word_pos == 0 or after_colon:
            if core:
                m = re.search(r"[A-Za-z]", core)
                if m:
                    pos = m.start()
                    first_char = core[pos].upper()
                    rest = core[pos+1:].lower()
                    new_core = core[:pos] + first_char + rest
                else:
                    new_core = core
            else:
                new_core = core

            result.append(prefix + new_core + suffix)
            word_pos += 1
            after_colon = w.endswith(":")
            continue

        # for words after the first one
        if lw in LOWER:
            result.append(prefix + lw + suffix)
        else:
            if core:
                m = re.search(r"[A-Za-z]", core)
                if m:
                    pos = m.start()
                    first_char = core[pos].upper()
                    rest = core[pos+1:].lower()
                    new_core = core[:pos] + first_char + rest
                else:
                    new_core = core
            else:
                new_core = core

            result.append(prefix + new_core + suffix)

        word_pos += 1
        after_colon = w.endswith(":")

    return "".join(result)


def sentence_case(text: str) -> str:
    """Convert the given text to sentence case

    Args:
        text (str): string that needs to be converted to sentence case

    Returns:
        str: formatted text in sentence case
    """
    words = re.split(r"(\s+|-)", text)
    result = []
    after_colon = False

    for idx, w in enumerate(words):
        if not w.strip() or re.match(r"\s+|-", w):
            result.append(w)
            after_colon = ":" in w
            continue

        # already protected by '{', '}'
        if w.startswith("{") and w.endswith("}"):
            result.append(w)
            after_colon = w.endswith(":")
            continue

        lw = w.lower().strip(string.punctuation)

        # apostrophe-s proper noun
        if lw.endswith("'s"):
            base = lw[:-2]
            if base in PROPER_LOWER_MAP and base not in EXCLUDE_APOS_S:
                result.append(PROPER_LOWER_MAP[base] + "'s")
                after_colon = w.endswith(":")
                continue

        # proper noun in USER_SPECIFIED_PROPER_NOUN/JOURNAL_CONFERENCE | initial character
        if lw in PROPER_LOWER_MAP:
            result.append(PROPER_LOWER_MAP[lw])
            after_colon = w.endswith(":")
            continue
        if w.upper() in ACRONYMS:
            result.append(w.upper())
            after_colon = w.endswith(":")
            continue

        # do not modify words starting with a digit or single letters
        if w[0].isdigit() or (len(w) == 1 and w.isupper()):
            result.append(w)
            after_colon = w.endswith(":")
            continue

        # capitalize the first word and any word immediately after ':'
        if idx == 0 or after_colon:
            m = re.search(r"[A-Za-z]", w)
            if m:
                pos = m.start()
                new_w = w[:pos] + w[pos].upper() + w[pos+1:].lower()
            else:
                new_w = w
            result.append(new_w)
        else:
            result.append(w.lower())

        after_colon = w.endswith(":")

    return "".join(result)


def format_entry(entry: Dict) -> Dict:
    """Formats a BibTeX entry by applying specific normalization and case formatting rules.

    Args:
        entry (Dict): A dictionary representing a single BibTeX entry.
            keys are the field names (e.g., "author", "title")
            values are the corresponding field values

    Returns:
        Dict: A dictionary representing the formatted BibTeX entry with specific fields.
    """
    # process for author field
    if "author" in entry:
        entry["author"] = normalize_author_field(entry["author"])

    # process for journal, booktitle field
    for field in ["journal", 'booktitle']:
        if field in entry:
            entry[field] = title_case(entry[field])
    # if it is arXiv, a separate conversion process will be applied.
    if "arxiv" in entry.get("journal", "").lower():
        entry["ENTRYTYPE"] = "article"
        entry["journal"] = entry["journal"].replace("Arxiv", "arXiv")

    # process for pages field
    if "pages" in entry:
        entry["pages"] = normalize_pages(entry["pages"])

    # process for title field
    entry_type = entry.get("ENTRYTYPE", "").lower()
    if "title" in entry:
        entry["title"] = protect_proper_nouns(entry["title"])
        if entry_type == "book":
            # processing regardless of the case
            entry["title"] = book_title_case(entry["title"])
        else:
            if CASE == "sentence":
                entry["title"] = sentence_case(entry["title"])
            else:
                entry["title"] = title_case(entry["title"])

    # specify the fields to keep according to the bibitem type.
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

    # arrange the order
    ordered = {}
    for k in ["ENTRYTYPE", "ID"] + keep:
        if k in entry:
            ordered[k] = entry[k]
    return ordered


def main(infile: Path, outfile: Path) -> None:
    """execute formatter

    Args:
        infile (Path): path to the input BibTeX file
        outfile (Path): path to the output BibTeX file
    """
    with infile.open(encoding="utf-8") as f:
        db = bibtexparser.load(f)

    db.entries = [format_entry(e) for e in db.entries]

    writer = BibTexWriter()
    writer.indent = "    "
    writer.order_entries_by = ("ID",)
    writer.comma_first = False
    writer.align_values = True

    with outfile.open("w", encoding="utf-8") as f:
        f.write(writer.write(db))

    print(f"[+] Cleaned .bib written to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BibTeX formatter (title-case / sentence-case switchable)"
    )
    parser.add_argument(
        "bibfile", type=Path, help="Input .bib file",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output .bib file (default: <infile>_clean.bib)"
    )
    parser.add_argument(
        "-c", "--case", choices=["title", "sentence"], default="sentence",
        help="title capitalisation style (default: sentence)"
    )
    parser.add_argument(
        "-eb", "--exclude-brace",
        choices=[True, False],
        default=True,
        help="remove braces around words that are *not* in USER_SPECIFIED_TITLE"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=f"{Path(__file__).parent}/config.json",
        help=f"path to the JSON file in which the configure is set (default: {Path(__file__).parent}/config.json)"
    )
    args = parser.parse_args()

    # load config
    with args.config.open(encoding="utf-8") as f:
        config = json.load(f)

    global USER_SPECIFIED_TITLE, EXCLUDE_APOS_S, LOWER, ACRONYMS, PROPER_LOWER_MAP
    USER_SPECIFIED_TITLE = set(config.get("USER_SPECIFIED_TITLE", []))
    EXCLUDE_APOS_S = set(config.get("EXCLUDE_APOS_S", []))
    LOWER = set(config.get("LOWER", []))
    ACRONYMS = set(config.get("ACRONYMS", []))
    PROPER_LOWER_MAP = {noun.lower(): noun for noun in USER_SPECIFIED_TITLE}

    global ARTICLE_KEEP, BOOK_KEEP, INPROC_KEEP, OTHER_KEEP
    KEEP_ELEMENTS = config.get("KEEP_ELEMENTS", {})
    ARTICLE_KEEP = KEEP_ELEMENTS.get("ARTICLE_KEEP", [])
    BOOK_KEEP = KEEP_ELEMENTS.get("BOOK_KEEP", [])
    INPROC_KEEP = KEEP_ELEMENTS.get("INPROC_KEEP", [])
    OTHER_KEEP = KEEP_ELEMENTS.get("OTHER_KEEP", [])

    # set info
    global INITIAL_RE
    INITIAL_RE = re.compile(r"^[A-Za-z]")
    # set info from argument
    global CASE, EXCLUDE_BRACE
    CASE = args.case
    EXCLUDE_BRACE = args.exclude_brace

    # execute formatter
    out = args.output or args.bibfile.with_stem(args.bibfile.stem + "_clean")
    main(args.bibfile, out)
