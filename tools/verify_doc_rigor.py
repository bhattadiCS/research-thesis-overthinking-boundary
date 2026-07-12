#!/usr/bin/env python3
"""Programmatic document verification script to check link validity, 
Mermaid diagram syntax, LaTeX formatting, and alert blocks across all markdown docs.
"""
import re
import sys
from pathlib import Path

# Paths to search
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
THESIS_DOCS_ROOT = WORKSPACE_ROOT / "ThesisDocs"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check_markdown_file(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    warnings = []

    # 1. Check Links
    # Extract links like [text](path) or [text](file:///path)
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
    for text, url in links:
        # Ignore external web links, only check local file references
        if url.startswith("http://") or url.startswith("https://") or url.startswith("mailto:"):
            continue
            
        # Clean file:/// absolute urls for check
        clean_url = url
        if clean_url.startswith("file:///"):
            clean_url = clean_url.replace("file:///", "")
            
        # Ignore math false positives (containing spaces, backslashes, math symbols)
        if any(char in clean_url for char in ['\\', ' ', '+', '-', '*', '|', '=', '<', '>', '{', '}']):
            continue
            
        # Check if relative to the file_path, or relative to workspace root, or absolute
        path_to_check = file_path.parent / clean_url
        if not path_to_check.exists():
            path_to_check_workspace = WORKSPACE_ROOT / clean_url
            if not path_to_check_workspace.exists():
                path_to_check_absolute = Path(clean_url)
                if not path_to_check_absolute.exists():
                    errors.append(f"Broken link: [{text}]({url}) - target path does not exist")

    # 2. Check Mermaid Blocks
    # Extract mermaid code blocks
    mermaid_blocks = re.findall(r"```mermaid\s*\n(.*?)\n```", content, re.DOTALL)
    for idx, block in enumerate(mermaid_blocks):
        # Basic checks on node labels to prevent formatting crashes
        # e.g., node[Label (With Parentheses)] is invalid in Mermaid, must be node["Label (With Parentheses)"]
        invalid_labels = re.findall(r"\w+\[[^\]]*\([^)]*\)[^\]]*\]", block)
        if invalid_labels:
            for label in invalid_labels:
                warnings.append(
                    f"Mermaid Warning (Block {idx+1}): Node label contains raw parentheses: {label}. "
                    "Recommend wrapping label text in double quotes inside brackets, e.g., node[\"Label (with parens)\"]"
                )
        
        # Check graph syntax validity
        lines = [line.strip() for line in block.splitlines() if line.strip() and not line.strip().startswith("%%")]
        if lines:
            first_word = lines[0].split()[0]
            valid_starts = {"graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram", "erDiagram", "gantt", "pie", "gitGraph"}
            if first_word not in valid_starts:
                errors.append(f"Invalid Mermaid graph type start: '{first_word}' in block {idx+1}")

    # 3. Check LaTeX Blocks
    # Extract $...$ and $$...$$
    double_dollar_blocks = re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL)
    for idx, block in enumerate(double_dollar_blocks):
        # Check for unmatched curly braces
        open_braces = block.count("{")
        close_braces = block.count("}")
        if open_braces != close_braces:
            errors.append(f"LaTeX Error: Unmatched curly braces in block {idx+1} ({open_braces} open vs {close_braces} close)")

    # 4. Check GitHub Alert Blocks
    # Legacy check: warn if alert blocks use old markdown blockquotes without modern tags
    blockquote_lines = re.findall(r"^>\s*(.*)$", content, re.MULTILINE)
    for line in blockquote_lines:
        # Check if it has a warning or alert-like text but doesn't have [!NOTE], [!TIP] etc.
        lowercase_line = line.lower()
        if any(keyword in lowercase_line for keyword in ["warning:", "note:", "tip:", "important:", "caution:"]):
            if not any(tag in line for tag in ["[!NOTE]", "[!TIP]", "[!IMPORTANT]", "[!WARNING]", "[!CAUTION]"]):
                warnings.append(f"Alert Block Suggestion: Line '{line}' contains alert-like keyword. Consider upgrading to modern GitHub alerts: '> [!NOTE]'")

    return {
        "errors": errors,
        "warnings": warnings,
        "links_count": len(links),
        "mermaid_count": len(mermaid_blocks),
        "latex_count": len(double_dollar_blocks)
    }


def main():
    print(f"Scanning directory: {THESIS_DOCS_ROOT}\n")
    if not THESIS_DOCS_ROOT.exists():
        print(f"{RED}Error: ThesisDocs directory not found at {THESIS_DOCS_ROOT}{RESET}")
        sys.exit(1)

    md_files = list(THESIS_DOCS_ROOT.rglob("*.md"))
    total_errors = 0
    total_warnings = 0

    for file_path in md_files:
        rel_path = file_path.relative_to(WORKSPACE_ROOT)
        result = check_markdown_file(file_path)
        
        errors = result["errors"]
        warnings = result["warnings"]
        
        if errors or warnings:
            print(f"[DOC] {YELLOW}{rel_path}{RESET}")
            print(f"   (Found {result['links_count']} links, {result['mermaid_count']} diagrams, {result['latex_count']} LaTeX blocks)")
            for err in errors:
                clean_err = err.replace('\u2212', '-').encode('ascii', 'replace').decode('ascii')
                print(f"   {RED}X Error:{RESET} {clean_err}")
                total_errors += 1
            for warn in warnings:
                clean_warn = warn.replace('\u2212', '-').encode('ascii', 'replace').decode('ascii')
                print(f"   {YELLOW}! Warning:{RESET} {clean_warn}")
                total_warnings += 1
            print()
        else:
            print(f"[DOC] {GREEN}OK {rel_path}{RESET} (Pass - {result['links_count']} links, {result['mermaid_count']} diagrams, {result['latex_count']} LaTeX blocks)")

    print("-" * 50)
    if total_errors > 0:
        print(f"[FAIL] Completed with {RED}{total_errors} errors{RESET} and {YELLOW}{total_warnings} warnings{RESET}.")
        sys.exit(1)
    else:
        print(f"[SUCCESS] All {len(md_files)} documents checked. {GREEN}0 errors{RESET}, {YELLOW}{total_warnings} warnings{RESET}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
