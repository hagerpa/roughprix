"""MkDocs hook: include README sections in the pages.

A page line  --8<-- "README.md:NAME"  is replaced by the README text between the markers
<!-- --8<-- [start:NAME] -->  and  <!-- --8<-- [end:NAME] -->.  The README's in-page links
are pointed to the pages that hold those sections.  A missing marker fails the build.
"""
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
INCLUDE = re.compile(r'^--8<-- "README\.md:(\w+)"$', re.MULTILINE)
ANCHORS = {
    "](#reproducing-the-paper)": "](paper.md)",
    "](#accuracy)": "](expansion.md#accuracy)",
}


def _section(text, name):
    start = f"<!-- --8<-- [start:{name}] -->\n"
    i = text.index(start) + len(start)
    return text[i:text.index(f"<!-- --8<-- [end:{name}] -->", i)]


def on_page_markdown(markdown, page, config, files):
    readme = README.read_text(encoding="utf-8")

    def include(match):
        body = _section(readme, match.group(1))
        for old, new in ANCHORS.items():
            body = body.replace(old, new)
        return body

    return INCLUDE.sub(include, markdown)
