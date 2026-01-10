#!/usr/bin/env python3
"""
Script to remove trailing whitespace from lines with non-whitespace content
in the current git repository, and optionally update/add Copyright notices.

This script:
1. Gets all tracked files from git
2. Optionally filters to only files in a specified subdirectory (--path)
3. Skips third-party directories (test/googletest/, ompi/)
4. Processes only text files (excludes binary files)
5. Removes trailing whitespace from lines that contain non-whitespace characters
6. Preserves empty lines and lines that are only whitespace
7. Optionally updates NVIDIA Copyright year notices to current year (third-party copyrights are preserved)
8. Optionally adds missing NVIDIA Copyright notices using current year
9. Shows a summary of changes made
"""


# To configure vim to remove trailing whitespace upon saving, add this to .vimrc:
#     autocmd BufWritePre * %s/\+$//e

import os
import sys
import subprocess
import mimetypes
from pathlib import Path
import argparse
import re
from datetime import datetime


def get_git_tracked_files():
    """Get all tracked files from git."""
    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError as e:
        print(f"Error getting git tracked files: {e}")
        return []
    except FileNotFoundError:
        print("Error: git command not found. Make sure you're in a git repository.")
        return []


def is_text_file(file_path):
    """
    Check if a file is likely a text file.
    Uses mimetypes and also checks for common binary file extensions.
    """
    # Common binary file extensions to exclude
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
        '.exe', '.dll', '.so', '.dylib', '.a', '.o',
        '.class', '.pyc', '.pyo',
        '.db', '.sqlite', '.sqlite3',
        '.bin', '.dat', '.obj', '.pdb',
        '.swp', '.swo', '.tmp', '.bak',
        '.git', '.gitignore',
        '.tags',  # ctags file
    }

    # Check file extension
    file_ext = Path(file_path).suffix.lower()
    if file_ext in binary_extensions:
        return False

    # Check mimetype
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and not (mime_type.startswith('text/') or file_ext=='.cu'):
        return False

    # Additional check: try to read first few bytes
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            # Check if it contains null bytes (indicates binary)
            if b'\x00' in chunk:
                return False
            # Check if it's mostly printable ASCII
            try:
                chunk.decode('utf-8')
            except UnicodeDecodeError:
                return False
    except (IOError, OSError) as e:
        return False

    return True


def should_have_copyright(file_path):
    """
    Determine if a file should have a copyright notice based on its extension.
    Matches the extensions supported by get_copyright_notice_for_file.
    """
    source_extensions = {
        # C/C++/CUDA
        '.c', '.cc', '.cpp', '.cxx', '.c++',
        '.h', '.hh', '.hpp', '.hxx', '.h++',
        '.cu', '.cuh',
        # Python/Shell
        '.py', '.sh',
        # Makefile
        '.mk',
        # Fortran
        '.f90',
    }

    file_ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    return file_ext in source_extensions or filename in ['Makefile', 'makefile']


def has_copyright_notice(content):
    """
    Check if content contains an NVIDIA copyright notice.
    """
    # Look for copyright in the first 50 lines
    lines = content.split('\n')[:50]
    first_lines = '\n'.join(lines)

    # Check for NVIDIA copyright specifically
    pattern = r'Copyright\s+(?:\([cC]\)\s+)?\d{4}.*NVIDIA'
    return bool(re.search(pattern, first_lines, re.IGNORECASE | re.DOTALL))


def get_copyright_notice_for_file(file_path, year):
    """
    Generate the appropriate copyright notice based on file extension.
    Does NOT include shebangs - those are handled separately.
    Returns None for extensions that should be ignored.
    Raises ValueError for unrecognized source file extensions.
    """
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    # C/C++/CUDA style (block comment)
    if ext in ['.c', '.cc', '.cpp', '.cxx', '.c++', '.cu', '.cuh',
               '.h', '.hh', '.hpp', '.hxx', '.h++']:
        return (
            f"/*************************************************************************\n"
            f" * Copyright (c) {year}, NVIDIA CORPORATION. All rights reserved.\n"
            f" *\n"
            f" * See LICENSE.txt for license information\n"
            f" ************************************************************************/\n\n"
        )

    # Python/Shell/Makefile style (line comments with #)
    elif ext in ['.py', '.sh', '.mk'] or filename in ['Makefile', 'makefile']:
        return (
            f"# Copyright (c) {year}, NVIDIA CORPORATION. All rights reserved.\n"
            f"#\n"
            f"# See LICENSE.txt for license information\n\n"
        )

    # Fortran style
    elif ext == '.f90':
        return (
            f"!* Copyright (c) {year}, NVIDIA CORPORATION. All rights reserved.\n"
            f"!*\n"
            f"!* See LICENSE.txt for license information\n\n"
        )

    # Explicitly ignored extensions (non-source files that are tracked in git)
    elif ext in [
        # Documentation and data files
        '.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.xml', '.html', '.css',
        # Build/config files
        '.in', '.cmake', '.m4', '.am', '.ac',
        # Git/IDE files
        '.gitignore', '.gitattributes', '.gitmodules',
        # Binary/compiled
        '.o', '.a', '.so', '.dylib', '.dll', '.exe',
        # Images
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
        # Archives
        '.zip', '.tar', '.gz', '.bz2', '.xz',
        # Special
        '', '.sample'  # No extension or special files
    ] or filename in ['.clang-format', 'LICENSE', 'README', 'CHANGELOG', 'AUTHORS']:
        return None  # Explicitly ignored

    # Unrecognized extension that might be a source file
    else:
        raise ValueError(f"Unrecognized file extension '{ext}' for file: {file_path}")


def add_copyright_notice(file_path, year):
    """
    Add a copyright notice to the beginning of a file.
    Preserves existing shebang lines if present.
    Returns True if copyright was added, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check if already has a copyright
        if has_copyright_notice(content):
            return False

        try:
            copyright_notice = get_copyright_notice_for_file(file_path, year)
        except ValueError as e:
            print(f"ERROR: {e}")
            return False

        if copyright_notice is None:
            return False  # File extension is explicitly ignored

        # Handle shebang lines - keep them at the top for script files
        if content.startswith('#!'):
            lines = content.split('\n', 1)
            if len(lines) == 2:
                shebang, rest = lines
                new_content = f"{shebang}\n{copyright_notice}{rest}"
            else:
                new_content = copyright_notice + content
        else:
            new_content = copyright_notice + content

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True

    except (IOError, OSError) as e:
        print(f"Error adding copyright to {file_path}: {e}")
        return False


def update_copyright_year(file_path, target_year):
    """
    Update copyright notices in a file to use the target year.
    Only updates NVIDIA copyrights, skips third-party copyrights.

    Handles various copyright formats:
    - Copyright (c) YYYY, NVIDIA CORPORATION. All rights reserved.
    - Copyright YYYY  NVIDIA Corporation.  All rights reserved.
    - Copyright (C) YYYY-YYYY, NVIDIA CORPORATION. All rights reserved.

    Returns (modified, copyrights_updated) tuple.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        original_content = content
        copyrights_updated = 0

        # Pattern to match various copyright formats
        # This pattern captures everything including text after the year
        # Group 1: prefix (Copyright (c) or Copyright)
        # Group 2: start year
        # Group 3: end year (optional)
        # Group 4: suffix (everything after the year/range)
        pattern = r'(Copyright\s+(?:\([cC]\)\s+)?)(\d{4})(?:-(\d{4}))?([,\s].*?)(?=\n|$)'

        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        for match in reversed(matches):  # Process in reverse to maintain positions
            prefix = match.group(1)
            start_year = int(match.group(2))
            end_year = match.group(3)
            suffix = match.group(4)

            # Only update NVIDIA copyrights, skip third-party ones
            if 'NVIDIA' not in suffix:
                continue

            if end_year:
                end_year = int(end_year)

            original_text = match.group(0)

            # Determine new text
            if end_year:
                if target_year > end_year:
                    new_text = f"{prefix}{start_year}-{target_year}{suffix}"
                else:
                    new_text = original_text  # No change needed
            else:
                if target_year > start_year:
                    new_text = f"{prefix}{start_year}-{target_year}{suffix}"
                else:
                    new_text = original_text  # No change needed

            if new_text != original_text:
                content = content[:match.start()] + new_text + content[match.end():]
                copyrights_updated += 1

        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, copyrights_updated

        return False, 0

    except (IOError, OSError) as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def remove_trailing_whitespace(file_path):
    """
    Remove trailing whitespace from lines that contain non-whitespace characters.
    Returns (modified, lines_changed) tuple.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        original_lines = lines.copy()
        lines_changed = 0

        for i, line in enumerate(lines):
            # Only process lines that contain non-whitespace characters
            if line.strip():  # Line has non-whitespace content
                # Remove trailing whitespace
                new_line = line.rstrip() + '\n'
            elif line:
                new_line = '\n'
            if new_line != line:
                lines[i] = new_line
                lines_changed += 1

        # Write back if changes were made
        if lines_changed > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True, lines_changed

        return False, 0

    except (IOError, OSError) as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def main():
    parser = argparse.ArgumentParser(
        description='Remove trailing whitespace from lines with non-whitespace content in git tracked files, '
                    'and optionally update/add Copyright notices'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--update-copyright',
        action='store_true',
        help='Update NVIDIA Copyright year notices to current year in addition to removing trailing whitespace '
             '(third-party copyrights are not modified)'
    )
    parser.add_argument(
        '--add-missing-copyright',
        action='store_true',
        help='Add copyright notices to files that are missing them '
             '(uses current year)'
    )
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='Only process files in the specified subdirectory '
             '(e.g., "src/" or "test/perf/")'
    )

    args = parser.parse_args()

    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("Error: Not in a git repository")
        sys.exit(1)

    # Get tracked files
    tracked_files = get_git_tracked_files()
    if not tracked_files:
        print("No tracked files found or not in a git repository")
        sys.exit(1)

    # Filter by path if specified
    if args.path:
        # Normalize path to ensure it ends with / for directory matching
        filter_path = args.path
        if not filter_path.endswith('/'):
            filter_path += '/'

        original_count = len(tracked_files)
        tracked_files = [f for f in tracked_files if f.startswith(filter_path)]

        if not tracked_files:
            print(f"No tracked files found in path: {args.path}")
            sys.exit(1)

        print(f"Found {len(tracked_files)} tracked files in {args.path} (filtered from {original_count} total)")
    else:
        print(f"Found {len(tracked_files)} tracked files")

    current_year = datetime.now().year
    if args.update_copyright:
        print(f"Copyright updating enabled (year: {current_year})")
    if args.add_missing_copyright:
        print(f"Adding missing copyright notices enabled (year: {current_year})")

    # Process files
    processed_files = 0
    modified_files = 0
    total_lines_changed = 0
    total_copyrights_updated = 0
    total_copyrights_added = 0
    files_missing_copyright = []

    for file_path in tracked_files:
        if not os.path.exists(file_path):
            continue

        # Skip third-party directories
        third_party_dirs = ['test/googletest/', 'ompi/']
        if any(file_path.startswith(d) for d in third_party_dirs):
            if args.verbose:
                print(f"Skipping third-party file: {file_path}")
            continue

        # Skip if not a text file
        if not is_text_file(file_path):
            if args.verbose:
                print(f"Skipping binary file: {file_path}")
            continue

        processed_files += 1

        if args.dry_run:
            # In dry-run mode, just check what would be changed
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.splitlines(keepends=True)

                lines_with_trailing = 0
                for i,line in enumerate(lines):
                    if line.rstrip() + '\n' != line:
                        if args.verbose:
                            print(f'   Found line {i+1}')
                        lines_with_trailing += 1

                copyrights_would_update = 0
                copyrights_would_add = 0

                # Check for missing copyright
                missing_copyright = False
                if should_have_copyright(file_path):
                    if not has_copyright_notice(content):
                        missing_copyright = True
                        if args.update_copyright and not args.add_missing_copyright:
                            print(f"WARNING: {file_path} should have a copyright notice but doesn't")
                            files_missing_copyright.append(file_path)
                        if args.add_missing_copyright:
                            copyrights_would_add = 1

                if args.update_copyright and not missing_copyright:
                    # Check for copyright notices that would be updated (NVIDIA only)
                    pattern = r'(Copyright\s+(?:\([cC]\)\s+)?)(\d{4})(?:-(\d{4}))?([,\s].*?)(?=\n|$)'
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        suffix = match.group(4)
                        # Only count NVIDIA copyrights
                        if 'NVIDIA' not in suffix:
                            continue

                        start_year = int(match.group(2))
                        end_year = match.group(3)
                        if end_year:
                            end_year = int(end_year)
                            if current_year > end_year:
                                copyrights_would_update += 1
                        else:
                            if current_year > start_year:
                                copyrights_would_update += 1

                if lines_with_trailing > 0 or copyrights_would_update > 0 or copyrights_would_add > 0:
                    changes_desc = []
                    if lines_with_trailing > 0:
                        changes_desc.append(f"{lines_with_trailing} lines with trailing whitespace")
                    if copyrights_would_update > 0:
                        changes_desc.append(f"{copyrights_would_update} copyright(s) to update")
                    if copyrights_would_add > 0:
                        changes_desc.append(f"copyright to add")
                    print(f"Would modify {file_path}: {', '.join(changes_desc)}")
                    modified_files += 1
                    total_lines_changed += lines_with_trailing
                    total_copyrights_updated += copyrights_would_update
                    total_copyrights_added += copyrights_would_add

            except (IOError, OSError) as e:
                print(f"Error reading {file_path}: {e}")
        else:
            # Actually process the file
            file_modified = False
            lines_changed = 0
            copyrights_updated = 0
            copyrights_added = 0

            # Check if file needs copyright
            missing_copyright = False
            if should_have_copyright(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if not has_copyright_notice(content):
                        missing_copyright = True
                        if args.update_copyright and not args.add_missing_copyright:
                            print(f"WARNING: {file_path} should have a copyright notice but doesn't")
                            files_missing_copyright.append(file_path)
                except (IOError, OSError):
                    pass

            # Add missing copyright if requested
            if args.add_missing_copyright and missing_copyright:
                if add_copyright_notice(file_path, current_year):
                    copyrights_added = 1
                    file_modified = True

            # Remove trailing whitespace
            modified, lines_changed = remove_trailing_whitespace(file_path)
            if modified:
                file_modified = True

            # Update copyright if requested (and not just added)
            if args.update_copyright and not missing_copyright:
                modified_copyright, copyrights_updated = update_copyright_year(file_path, current_year)
                if modified_copyright:
                    file_modified = True

            if file_modified:
                changes_desc = []
                if copyrights_added > 0:
                    changes_desc.append(f"copyright added")
                if lines_changed > 0:
                    changes_desc.append(f"{lines_changed} lines")
                if copyrights_updated > 0:
                    changes_desc.append(f"{copyrights_updated} copyright(s) updated")
                print(f"Modified {file_path}: {', '.join(changes_desc)}")
                modified_files += 1
                total_lines_changed += lines_changed
                total_copyrights_updated += copyrights_updated
                total_copyrights_added += copyrights_added
            elif args.verbose:
                print(f"No changes needed: {file_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Files processed: {processed_files}")
    print(f"  Files {'would be ' if args.dry_run else ''}modified: {modified_files}")
    print(f"  Total lines {'would be ' if args.dry_run else ''}changed: {total_lines_changed}")
    if args.update_copyright:
        print(f"  Total copyrights {'would be ' if args.dry_run else ''}updated: {total_copyrights_updated}")
        if files_missing_copyright:
            print(f"  Files missing copyright notices: {len(files_missing_copyright)}")
    if args.add_missing_copyright:
        print(f"  Total copyrights {'would be ' if args.dry_run else ''}added: {total_copyrights_added}")

    if files_missing_copyright and args.verbose:
        print(f"\nFiles missing copyright notices:")
        for f in files_missing_copyright:
            print(f"  - {f}")

    if args.dry_run and modified_files > 0:
        print(f"\nRun without --dry-run to apply these changes")

    sys.exit( 0 if modified_files==0 else 1 )


if __name__ == '__main__':
    main()
