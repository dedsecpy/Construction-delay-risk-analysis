"""Convert ordinary figure floats to in-flow captionof blocks (keep figure*)."""
from pathlib import Path
import re

path = Path("main.tex")
text = path.read_text(encoding="utf-8")

pattern = re.compile(
    r"\\begin\{figure\}\[!ht\]\s*"
    r"\\centering\s*"
    r"(\\includegraphics\[[^\]]*\]\{[^}]+\})\s*"
    r"\\caption\{(.*?)\}\s*"
    r"\\label\{(.*?)\}\s*"
    r"\\end\{figure\}",
    re.S,
)


def repl(m: re.Match) -> str:
    include, caption, label = m.group(1), m.group(2), m.group(3)
    return (
        "\\begin{center}\n"
        f"{include}\n"
        f"\\captionof{{figure}}{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{center}"
    )


new, n = pattern.subn(repl, text)
path.write_text(new, encoding="utf-8")
print(f"converted {n} ordinary figures to in-flow captionof")
