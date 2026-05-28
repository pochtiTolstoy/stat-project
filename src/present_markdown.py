import argparse
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Slide:
    title: str
    body: list[str]


def clear_screen() -> None:
    os.system("clear")


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_slides(markdown: str) -> list[Slide]:
    lines = markdown.splitlines()
    deck_title = "Markdown Presentation"
    slides: list[Slide] = []
    current_title = "Intro"
    current_body: list[str] = []

    for line in lines:
        if line.startswith("# "):
            deck_title = line[2:].strip()
            if not slides and not current_body:
                current_title = deck_title
            else:
                current_body.append(line)
            continue

        if line.startswith("## "):
            if current_body:
                slides.append(Slide(current_title, current_body))
            current_title = line[3:].strip()
            current_body = []
            continue

        current_body.append(line)

    if current_body or not slides:
        slides.append(Slide(current_title, current_body))

    if slides and slides[0].title == "Intro":
        slides[0].title = deck_title

    return slides


def normalize_line(line: str) -> str:
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    return line.rstrip()


def wrap_slide(slide: Slide, width: int) -> list[str]:
    wrapped = [slide.title, "=" * min(len(slide.title), width), ""]

    for raw_line in slide.body:
        line = normalize_line(raw_line)

        if not line.strip():
            wrapped.append("")
            continue

        indent = ""
        content = line
        if line.startswith("- "):
            indent = "  "
            content = "• " + line[2:].strip()
        elif re.match(r"\d+\.\s", line):
            indent = "   "
            content = line

        wrapped.extend(
            textwrap.wrap(
                content,
                width=width,
                initial_indent="",
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )

    return wrapped


def render_slide(slides: list[Slide], index: int, source: Path) -> None:
    width = max(60, shutil.get_terminal_size((100, 30)).columns - 4)
    height = max(20, shutil.get_terminal_size((100, 30)).lines - 6)
    lines = wrap_slide(slides[index], width)

    clear_screen()
    print(f"{source.name}  [{index + 1}/{len(slides)}]")
    print()

    if len(lines) > height:
        visible = lines[: height - 1]
        visible.append("...")
    else:
        visible = lines

    for line in visible:
        print(line)

    print()
    print("[Enter/n] next   [p] previous   [q] quit   [number] jump")


def run_presentation(source: Path, slides: list[Slide]) -> None:
    index = 0
    while True:
        render_slide(slides, index, source)
        try:
            command = input("> ").strip().lower()
        except EOFError:
            print()
            return

        if command in {"", "n", "next"}:
            if index < len(slides) - 1:
                index += 1
            continue

        if command in {"p", "prev", "previous"}:
            if index > 0:
                index -= 1
            continue

        if command in {"q", "quit", "exit"}:
            return

        if command.isdigit():
            jump = int(command) - 1
            if 0 <= jump < len(slides):
                index = jump


def list_slides(slides: list[Slide]) -> None:
    for i, slide in enumerate(slides, start=1):
        print(f"{i:>2}. {slide.title}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show a Markdown file as a simple slide deck in the terminal."
    )
    parser.add_argument("file", nargs="?", default="defense_notes.md")
    parser.add_argument("--list", action="store_true", help="Print slide titles and exit.")
    args = parser.parse_args()

    source = Path(args.file).resolve()
    if not source.exists():
        print(f"File not found: {source}", file=sys.stderr)
        return 1

    slides = split_slides(read_markdown(source))
    if args.list:
        list_slides(slides)
        return 0

    run_presentation(source, slides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
