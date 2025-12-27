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
            block = title[i + 1 : j - 1]  # noqa: E203
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

    # If already abbreviated with '~', preserve all initials (e.g., "A.~B.")
    if "~" in core:
        subs = [p for p in core.split("~") if p.strip()]
        return head + "~".join(_abbr_given(p) for p in subs)

    # If already packed initials like "A.B." keep both initials
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", core):
        # keep as-is (or return "~".join(re.findall(r"[A-Za-z]\.", core)) if you prefer)
        return head + core

    parts = re.split(r"[-\s]+", core)
    initials = []
    for p in parts:
        if INITIAL_RE.match(p):
            initials.append(p[0] + ".")
    return head + "-".join(initials)


def _split_first_outside_braces(s: str, sep: str = ",") -> tuple[str, str] | None:
    """Split at the first separator that appears at brace depth 0."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
        elif ch == sep and depth == 0:
            return s[:i], s[i + 1 :]  # noqa: E203
    return None


def _tokenize_outside_braces(s: str) -> list[str]:
    """
    Split by whitespace, but keep brace groups as single tokens.
    Example: "Mark {van der Laan}" -> ["Mark", "{van der Laan}"]
    """
    tokens: list[str] = []
    i = 0
    n = len(s)

    while i < n:
        # skip spaces and BibTeX non-breaking space (~)
        if s[i].isspace() or s[i] == "~":
            i += 1
            continue

        if s[i] == "{":
            # parse balanced brace group (with nesting)
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            tokens.append(s[i:j])  # keep braces
            i = j
            continue

        # normal token until whitespace or '~'
        j = i
        while j < n and (not s[j].isspace()) and s[j] != "~":
            # do not start a brace group here; it becomes its own token in next loop
            if s[j] == "{":
                break
            j += 1
        tokens.append(s[i:j])
        i = j

    return tokens


def _is_corporate_author(raw_author: str) -> bool:
    """
    Heuristic: treat a fully braced author string with no comma outside braces
    as a corporate author and keep as-is.
    """
    s = raw_author.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    return _split_first_outside_braces(s, ",") is None


def _first_alpha_char(s: str) -> str | None:
    """Return the first alphabetic character in s (ignoring braces/punct), or None."""
    # keep TeX accents as-is; just look for a-z/A-Z
    for ch in s:
        if ch.isalpha():
            return ch
    return None


def _is_lowercase_particle_token(tok: str) -> bool:
    """
    Decide whether tok should be treated as a surname particle (e.g., van, der, de).
    Works even if tok is braced like "{van}".
    """
    core = tok.strip(string.punctuation).strip()

    # remove one pair of surrounding braces if present
    if core.startswith("{") and core.endswith("}"):
        core = core[1:-1].strip()

    ch = _first_alpha_char(core)
    if ch is None:
        return False
    return ch.islower()


def normalize_author_field(author_field: str) -> str:
    """Normalize the author field in a BibTeX entry.

    - Keeps brace-protected name parts (e.g., "{van der Laan}", "{Andersen}") unchanged.
    - Abbreviates given names to initials for non-braced tokens.
    - Supports multi-word surname particles (e.g., "van der Laan") in "First Last" form.
    """
    normalized: list[str] = []

    for raw in re.split(r"\s+and\s+", author_field):
        raw = raw.strip()
        if not raw:
            continue

        # Corporate author: keep exactly as-is (e.g., "{World Health Organization}")
        if _is_corporate_author(raw):
            normalized.append(raw)
            continue

        # 1) "Last, First" (comma outside braces)
        split = _split_first_outside_braces(raw, ",")
        if split is not None:
            last, given = split[0].strip(), split[1].strip()

            # tokenize given while preserving brace groups
            given_tokens = _tokenize_outside_braces(given)
            initials_parts: list[str] = []
            for gt in given_tokens:
                # if a token is brace-protected as a whole, keep as-is
                if gt.startswith("{") and gt.endswith("}"):
                    initials_parts.append(gt)
                else:
                    ab = _abbr_given(gt)
                    if ab:
                        initials_parts.append(ab)
            initials = "~".join(initials_parts)
            normalized.append(f"{last}, {initials}".rstrip().rstrip(","))
            continue

        # 2) "First ... Last" (no comma outside braces)
        tokens = _tokenize_outside_braces(raw)
        if not tokens:
            continue

        # surname = last token + preceding lowercase particles (including braced particles)
        idx = len(tokens) - 1
        last_parts = [tokens[idx]]
        idx -= 1
        while idx >= 0 and _is_lowercase_particle_token(tokens[idx]):
            last_parts.insert(0, tokens[idx])
            idx -= 1

        last = " ".join(last_parts)
        given_tokens = tokens[: idx + 1]

        initials_parts = []
        for gt in given_tokens:
            if gt.startswith("{") and gt.endswith("}"):
                # brace-protected given part: keep as-is
                initials_parts.append(gt)
            else:
                ab = _abbr_given(gt)
                if ab:
                    initials_parts.append(ab)
        initials = "~".join(initials_parts)

        if initials:
            normalized.append(f"{last}, {initials}")
        else:
            normalized.append(last)

    return " and ".join(normalized)


# module for formatting journal name
def _canonical_pattern(s: str) -> re.Pattern:
    """Build a case-insensitive regex pattern for matching a canonical journal string.

    The input string is normalized by removing BibTeX braces and splitting on
    whitespace. The resulting pattern allows flexible whitespace (one or more
    spaces) between tokens and is compiled with IGNORECASE. If the canonical
    string starts/ends with an alphanumeric character, word boundaries are added
    to avoid partial matches inside longer words.

    Args:
        s (str): Canonical journal string (e.g., "PLoS One") defined in the config.

    Returns:
        re.Pattern: Compiled regular expression used for substring canonicalization.
    """
    s = s.replace("{", "").replace("}", "").strip()
    parts = re.split(r"\s+", s)
    pat = r"\s+".join(re.escape(p) for p in parts)

    # add word boundaries if it starts/ends with alnum
    if s and s[0].isalnum():
        pat = r"\b" + pat
    if s and s[-1].isalnum():
        pat = pat + r"\b"

    return re.compile(pat, flags=re.IGNORECASE)


def _journal_key(s: str) -> str:
    """Generate a normalized lookup key for a journal name.

    This helper creates a canonical comparison key by:
      - removing BibTeX braces ('{' and '}'),
      - collapsing consecutive whitespace into a single space,
      - trimming leading/trailing whitespace,
      - lowercasing the result.

    The key is intended for case-insensitive, formatting-insensitive matching
    against a predefined journal canonicalization map.

    Args:
        s (str): Raw journal name as it appears in a BibTeX entry.

    Returns:
        str: Normalized key used for journal name matching.
    """
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def canonicalize_journal(name: str) -> str:
    """Canonicalize a journal name using exact and substring-based rules.

    If the journal name contains BibTeX braces, it is treated as user-protected
    and returned unchanged. Otherwise, the function first checks an exact-match
    canonical map (whitespace/braces/case normalized). If no exact match is found,
    it applies title-casing and then performs case-insensitive substring
    replacements based on configured canonical journal strings.

    Args:
        name (str): Journal name as it appears in the BibTeX entry.

    Returns:
        str: Canonicalized journal name with preferred capitalization, or the
        original name if it was brace-protected.
    """
    # 1) already brace-protected by user -> do not touch
    if "{" in name or "}" in name:
        return name

    # 2) exact match -> canonical
    key = _journal_key(name)
    if key in JOURNAL_CANONICAL_MAP:
        return JOURNAL_CANONICAL_MAP[key]

    # 3) generic title case + substring canonicalization
    out = title_case(name)
    for pat, canon in JOURNAL_CANONICAL_PATTERNS:
        out = pat.sub(canon, out)
    return out


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
    if "--" in s and not re.search(r"\s--\s", s):
        return s
    return re.sub(r"\s*-+\s*", "--", s)


# modules for formatting title
def _is_fully_braced_token(token: str) -> bool:
    """
    Return True if the token is essentially "{...}" possibly wrapped by
    leading/trailing punctuation (e.g., "{Wuhan},", "({United})", "{US}."),
    and has no letters/digits/backslashes outside the brace block.

    This is used to preserve user-protected capitalization inside braces.
    """
    if "{" not in token:
        return False

    # 1) scan from left: allow punctuation until we see '{'
    i = 0
    n = len(token)
    while i < n:
        ch = token[i]
        if ch == "{":
            break
        # if we see alnum or TeX backslash before '{', it's not a fully-braced token (e.g., "Fr{\'e}chet")
        if ch.isalnum() or ch == "\\":
            return False
        i += 1

    if i >= n or token[i] != "{":
        return False

    # 2) parse matching brace block with nesting
    j = i + 1
    depth = 1
    while j < n and depth > 0:
        if token[j] == "{":
            depth += 1
        elif token[j] == "}":
            depth -= 1
        j += 1

    if depth != 0:
        return False  # unbalanced braces

    # 3) after the brace block, only allow punctuation (no alnum/backslash)
    k = j
    while k < n:
        ch = token[k]
        if ch.isalnum() or ch == "\\":
            return False
        k += 1

    return True


def _split_words_keep_seps(text: str) -> list[str]:
    """
    Split *text* into tokens while preserving separators (whitespace and hyphen runs),
    but NEVER split inside balanced {...} groups.

    This replaces: re.split(r"(\\s+|-)", text)
    """
    out: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # separator: whitespace (depth is always 0 here because we don't track global depth;
        # separators only matter outside braces, and inside braces we are in "word" mode)
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # separator: hyphen run (e.g., "-", "--")
        if ch == "-":
            j = i + 1
            while j < n and text[j] == "-":
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # word token: read until next whitespace/hyphen at brace depth 0
        j = i
        depth = 0
        while j < n:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth = max(depth - 1, 0)
            elif depth == 0 and (c.isspace() or c == "-"):
                break
            j += 1

        out.append(text[i:j])
        i = j

    return out


def _is_sep_token(tok: str) -> bool:
    return bool(tok) and (tok.isspace() or set(tok) == {"-"})


def _ends_with_colon(tok: str) -> bool:
    """
    Consider ':' as a boundary even if followed by trailing punctuation, e.g. "Title:,"
    """
    t = tok.rstrip()
    k = len(t) - 1
    while k >= 0 and t[k] in string.punctuation and t[k] != ":":
        k -= 1
    return k >= 0 and t[k] == ":"


def title_case(text: str) -> str:
    words = _split_words_keep_seps(text)
    result: list[str] = []
    after_colon = False
    word_pos = 0  # count only non-separator tokens

    for w in words:
        if _is_sep_token(w):
            result.append(w)
            continue

        # Fully brace-protected token => keep EXACTLY as-is
        if _is_fully_braced_token(w):
            result.append(w)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        # Preserve leading/trailing punctuation (but don't treat braces as punctuation here)
        prefix = ""
        suffix = ""
        core = w

        while core and core[0] in string.punctuation and core[0] not in "{}":
            prefix += core[0]
            core = core[1:]
        while core and core[-1] in string.punctuation and core[-1] not in "{}":
            suffix = core[-1] + suffix
            core = core[:-1]

        plain = core.strip(string.punctuation)
        lower_plain = plain.lower()

        # proper noun with "'s"
        if lower_plain.endswith("'s"):
            base = lower_plain[:-2]
            if base in PROPER_LOWER_MAP and base not in EXCLUDE_APOS_S:
                result.append(prefix + PROPER_LOWER_MAP[base] + "'s" + suffix)
                after_colon = _ends_with_colon(w)
                word_pos += 1
                continue

        # proper noun (case-insensitive canonicalization)
        if lower_plain in PROPER_LOWER_MAP:
            result.append(prefix + PROPER_LOWER_MAP[lower_plain] + suffix)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        # acronyms
        if plain.upper() in ACRONYMS:
            result.append(prefix + plain.upper() + suffix)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        # digits / single-letter uppercase
        if core and (core[0].isdigit() or (len(core) == 1 and core.isalpha() and core.isupper())):
            result.append(w)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        lw = core.lower()

        # first word OR right after colon => capitalize
        if word_pos == 0 or after_colon:
            m = re.search(r"[A-Za-z]", core)
            if m:
                pos = m.start()
                new_core = core[:pos] + core[pos].upper() + core[pos + 1 :].lower()  # noqa: E203
            else:
                new_core = core
            result.append(prefix + new_core + suffix)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        # small words => lower
        if lw in LOWER:
            result.append(prefix + lw + suffix)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        # default: Capitalize
        m = re.search(r"[A-Za-z]", core)
        if m:
            pos = m.start()
            new_core = core[:pos] + core[pos].upper() + core[pos + 1 :].lower()  # noqa: E203
        else:
            new_core = core
        result.append(prefix + new_core + suffix)
        after_colon = _ends_with_colon(w)
        word_pos += 1

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

        # already brace-protected token
        if _is_fully_braced_token(w):
            result.append(w)
            after_colon = w.rstrip().endswith(":")
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
            core = core[len(prefix) :]  # noqa: E203

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
                    rest = core[pos + 1 :].lower()  # noqa: E203
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
                    rest = core[pos + 1 :].lower()  # noqa: E203
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
    words = _split_words_keep_seps(text)
    result: list[str] = []
    after_colon = False
    word_pos = 0

    for w in words:
        if _is_sep_token(w):
            result.append(w)
            continue

        # Fully brace-protected token => keep EXACTLY as-is
        if _is_fully_braced_token(w):
            result.append(w)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        prefix = ""
        suffix = ""
        core = w
        while core and core[0] in string.punctuation and core[0] not in "{}":
            prefix += core[0]
            core = core[1:]
        while core and core[-1] in string.punctuation and core[-1] not in "{}":
            suffix = core[-1] + suffix
            core = core[:-1]

        plain = core.strip(string.punctuation)
        lower_plain = plain.lower()

        # apostrophe-s proper noun
        if lower_plain.endswith("'s"):
            base = lower_plain[:-2]
            if base in PROPER_LOWER_MAP and base not in EXCLUDE_APOS_S:
                result.append(prefix + PROPER_LOWER_MAP[base] + "'s" + suffix)
                after_colon = _ends_with_colon(w)
                word_pos += 1
                continue

        # proper noun / acronyms
        if lower_plain in PROPER_LOWER_MAP:
            result.append(prefix + PROPER_LOWER_MAP[lower_plain] + suffix)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue
        if plain.upper() in ACRONYMS:
            result.append(prefix + plain.upper() + suffix)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        # digits / single-letter uppercase -> keep
        if core and (core[0].isdigit() or (len(core) == 1 and core.isupper())):
            result.append(w)
            after_colon = _ends_with_colon(w)
            word_pos += 1
            continue

        # first word OR after colon => capitalize, else lower
        if word_pos == 0 or after_colon:
            m = re.search(r"[A-Za-z]", core)
            if m:
                pos = m.start()
                new_core = core[:pos] + core[pos].upper() + core[pos + 1 :].lower()  # noqa: E203
            else:
                new_core = core
            result.append(prefix + new_core + suffix)
        else:
            result.append(prefix + core.lower() + suffix)

        after_colon = _ends_with_colon(w)
        word_pos += 1

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
    if "journal" in entry:
        entry["journal"] = canonicalize_journal(entry["journal"])
    if "booktitle" in entry:
        entry["booktitle"] = title_case(entry["booktitle"])
    # if it is arXiv, a separate conversion process will be applied.
    if entry.get("ENTRYTYPE", "").lower() == "misc" or "arxiv" in entry.get("journal", "").lower():
        # extracted arXiv id
        arx_id = None
        # extract arXiv id from note field
        m = re.search(r"arXiv:([0-9.]+)", entry.get("note", ""), flags=re.I)
        if m:
            arx_id = m.group(1)

        entry["ENTRYTYPE"] = "article"
        if arx_id:
            entry["journal"] = f"arXiv preprint arXiv:{arx_id}"
        else:
            # if no arXiv id, then only 'arXiv preprint'
            entry["journal"] = "arXiv preprint"

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

    print(f"[+] Formatted .bib written to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BibTeX formatter (title-case / sentence-case switchable)")
    parser.add_argument(
        "bibfile",
        type=Path,
        help="Input .bib file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .bib file (default: <infile>_formatted.bib)",
    )
    parser.add_argument(
        "-c",
        "--case",
        choices=["title", "sentence"],
        default="sentence",
        help="title capitalisation style (default: sentence)",
    )
    parser.add_argument(
        "-eb",
        "--exclude-brace",
        action="store_true",
        help="If given, remove braces around words *not* in USER_SPECIFIED_TITLE",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=f"{Path(__file__).parent}/config.json",
        help=f"path to the JSON file in which the configure is set (default: {Path(__file__).parent}/config.json)",
    )
    args = parser.parse_args()

    # load config
    with args.config.open(encoding="utf-8") as f:
        config = json.load(f)

    global USER_SPECIFIED_TITLE, PROPER_LOWER_MAP
    USER_SPECIFIED_TITLE = set(config.get("USER_SPECIFIED_TITLE", []))
    PROPER_LOWER_MAP = {noun.lower(): noun for noun in USER_SPECIFIED_TITLE}

    global EXCLUDE_APOS_S, LOWER, ACRONYMS, JOURNAL_CANONICAL_MAP, JOURNAL_CANONICAL_PATTERNS
    EXCLUDE_APOS_S = set(config.get("EXCLUDE_APOS_S", []))
    LOWER = set(config.get("LOWER", []))
    ACRONYMS = set(config.get("ACRONYMS", []))
    JOURNAL_CANONICAL = config.get("JOURNAL_CANONICAL", [])
    JOURNAL_CANONICAL_MAP = {_journal_key(x): x for x in JOURNAL_CANONICAL}
    JOURNAL_CANONICAL_PATTERNS = [(_canonical_pattern(x), x) for x in sorted(JOURNAL_CANONICAL, key=len, reverse=True)]

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
    out = args.output or args.bibfile.with_stem(args.bibfile.stem + "_formatted")
    main(args.bibfile, out)
