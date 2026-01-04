#!/usr/bin/env python3
"""
Script to convert Jupyter notebook to Python script.
"""

import json
import re
import subprocess
import tempfile
import os


def remove_comments_from_python(content):
    """
    Remove all comments from Python content.

    Args:
        content (str): Python code content

    Returns:
        str: Python code without comments
    """
    import re

    # Remove triple quote comments (docstrings and multiline comments)
    # This regex removes content between triple quotes
    content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
    content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)

    # Remove single line comments
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        # Check if the line is not empty and doesn't start with a comment
        stripped = line.strip()
        if stripped.startswith('#'):
            # Skip comment lines
            continue
        else:
            # Remove inline comments but keep the code
            # Find the first # that's not in a string
            in_string = False
            quote_char = None
            new_line = []
            i = 0
            while i < len(line):
                char = line[i]
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == '#' and not in_string:
                    # Found a comment, stop processing this line
                    break
                new_line.append(char)
                i += 1

            cleaned_line = ''.join(new_line).rstrip()
            if cleaned_line or (stripped and not stripped.startswith('#')):
                cleaned_lines.append(cleaned_line)

    # Remove extra empty lines
    final_lines = []
    prev_was_empty = False
    for line in cleaned_lines:
        if not line.strip():
            if not prev_was_empty:
                final_lines.append(line)
                prev_was_empty = True
        else:
            final_lines.append(line)
            prev_was_empty = False

    return '\n'.join(final_lines)


def remove_colab_notebook_code(content):
    """
    Remove Colab and notebook-specific code from Python content.

    Args:
        content (str): Python code content

    Returns:
        str: Python code without Colab/notebook specific code
    """
    lines = content.split('\n')
    filtered_lines = []

    # Find and remove COLAB-related if/else blocks
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line is part of a COLAB-related block
        if 'COLAB_' in line or any(skip_text in line for skip_text in [
            'google.colab',
            'Runtime',
            'Tesla T4 Google Colab',
            'free Tesla T4',
            'drive.mount',
            'ipywidgets',
            'widget'
        ]):
            # This is part of a COLAB block, skip it
            i += 1
        else:
            # Check if this line is an else/elif that might be part of a COLAB block
            stripped_line = line.strip()
            if stripped_line.startswith(('else:', 'elif:')):
                # Look back to see if there was a COLAB-related if statement
                found_colab_if = False
                j = i - 1
                while j >= 0:
                    prev_line = lines[j].strip()
                    if prev_line.startswith('if ') and 'COLAB_' in prev_line:
                        found_colab_if = True
                        break
                    elif prev_line.startswith('if '):
                        # Found a different if statement, stop looking
                        break
                    j -= 1

                if found_colab_if:
                    # This else/elif is part of a COLAB block, skip it and all lines
                    # that belong to this block based on indentation
                    else_indent = len(lines[i]) - len(lines[i].lstrip())
                    i += 1  # Move to the next line
                    while i < len(lines):
                        current_line = lines[i].strip()
                        if not current_line:
                            i += 1
                            continue
                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        # If the indentation is less than or equal to the else statement's indentation
                        # and it's not an elif/else, we've reached the end of the block
                        if current_indent <= else_indent and not current_line.startswith(('elif:', 'else:')):
                            break
                        i += 1
                    continue  # Don't add anything from this block
            elif stripped_line.startswith('if ') and 'COLAB_' in stripped_line:
                # This is a COLAB-related if statement, skip the entire if/elif/else block
                # Find the end of this block by looking at indentation
                if_indent = len(lines[i]) - len(lines[i].lstrip())
                i += 1  # Move to the next line
                while i < len(lines):
                    current_line = lines[i].strip()
                    if not current_line:
                        i += 1
                        continue
                    current_indent = len(lines[i]) - len(lines[i].lstrip())
                    # If the indentation is less than or equal to the if statement's indentation
                    # and it's not an elif/else, we've reached the end of the block
                    if current_indent <= if_indent and not current_line.startswith(('elif:', 'else:')):
                        break
                    i += 1
                continue  # Don't add anything from this block

            # Skip lines that set up notebook environments
            if any(skip_pattern in line for skip_pattern in [
                'get_ipython()',
                'ipython',
                '%',
                'jupyter',
                'notebook'
            ]) and not any(keep_pattern in line for keep_pattern in [
                'import',  # Don't remove import statements
            ]):
                i += 1
                continue

            # Skip lines that install packages in specific ways for notebooks
            if 'pip install' in line and any(skip_text in line for skip_text in [
                'pip install --upgrade',
                'pip install -U',
                'pip install git+',
            ]):
                i += 1
                continue

            # Keep the line if it doesn't match any skip patterns
            filtered_lines.append(line)
            i += 1

    return '\n'.join(filtered_lines)


def format_with_ruff(file_path):
    """
    Format a Python file using ruff check --fix first, then ruff format.

    Args:
        file_path (str): Path to the Python file to format
    """
    try:
        # Run ruff check --fix first to fix import-related issues
        result = subprocess.run(['ruff', 'check', '--fix', file_path],
                                capture_output=True, text=True, check=True)
        print(f"Successfully checked and fixed {file_path} with ruff")
    except subprocess.CalledProcessError as e:
        print(f"Error checking with ruff: {e}")
        print(f"Error output: {e.stderr}")
    except FileNotFoundError:
        print("Ruff is not installed. Please install it with 'pip install ruff'")

    try:
        # Run ruff format on the file after fixing issues
        result = subprocess.run(['ruff', 'format', file_path],
                                capture_output=True, text=True, check=True)
        print(f"Successfully formatted {file_path} with ruff")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting with ruff format: {e}")
        print(f"Error output: {e.stderr}")
    except FileNotFoundError:
        print("Ruff is not installed. Please install it with 'pip install ruff'")


def convert_notebook_to_python(notebook_path, output_path):
    """
    Convert a Jupyter notebook to a Python script with formatting and cleaning.

    Args:
        notebook_path (str): Path to the input notebook file
        output_path (str): Path to the output Python file
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Process cells one by one to handle multi-cell if/else blocks
    all_code_lines = []

    for cell in notebook['cells']:
        cell_type = cell['cell_type']

        if cell_type == 'markdown':
            # Skip markdown cells since we're removing comments
            continue

        elif cell_type == 'code':
            # Add code cells directly
            source_lines = cell['source']
            for line in source_lines:
                # Handle shell commands in code cells
                if line.strip().startswith('!'):
                    # Skip shell commands entirely since we're removing os.system calls
                    continue
                else:
                    all_code_lines.append(line.rstrip())

            # Add an empty line after each code cell for readability
            all_code_lines.append("")

    # Join all the code
    full_code = '\n'.join(all_code_lines)

    # Remove comments
    full_code = remove_comments_from_python(full_code)

    # Remove Colab/notebook specific code
    full_code = remove_colab_notebook_code(full_code)

    # Extract imports and move them to the top
    full_code = move_imports_to_top(full_code)

    # Write the Python code to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_code)

    # Format the code with ruff
    format_with_ruff(output_path)

    print(f"Successfully converted {notebook_path} to {output_path}")
    print("  - Converted notebook to Python")
    print("  - Removed comments")
    print("  - Removed Colab/notebook specific code")
    print("  - Removed os.system calls")
    print("  - Moved imports to top")
    print("  - Formatted with ruff")


def move_imports_to_top(content):
    """
    Move all import statements to the top of the file.

    Args:
        content (str): Python code content

    Returns:
        str: Python code with imports moved to the top
    """
    lines = content.split('\n')

    imports = []
    other_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this is a simple import statement (not combined with os.system)
        if stripped.startswith('import ') and not ';' in line and not stripped.startswith('import os'):
            imports.append(line)
        elif stripped.startswith('from ') and not ';' in line:
            # This might be a multiline import, so we need to collect all lines
            # that belong to this import statement
            import_block = [line]
            j = i + 1
            # Check if the import continues on the next lines (for multiline imports)
            # Look for the closing parenthesis
            open_parens_count = line.count('(') - line.count(')')
            while j < len(lines) and open_parens_count > 0:
                next_line = lines[j]
                import_block.append(next_line)
                open_parens_count += next_line.count('(') - next_line.count(')')
                j += 1

            imports.extend(import_block)
            i = j
            continue
        elif stripped.startswith('import os') and ';' in line and 'os.system' in line:
            # This is an import os; os.system(...) line - treat as other code
            other_lines.append(line)
        else:
            # This is not an import statement
            other_lines.append(line)

        i += 1

    # Remove duplicate imports while preserving order
    seen_imports = set()
    unique_imports = []
    for imp in imports:
        if imp not in seen_imports:
            seen_imports.add(imp)
            unique_imports.append(imp)

    # Combine imports at the top and other code below
    result_lines = unique_imports + [''] + other_lines

    # Remove any empty lines at the beginning (except after imports)
    while result_lines and result_lines[0].strip() == '':
        result_lines.pop(0)

    return '\n'.join(result_lines)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 3:
        print("Usage: python convert_notebook.py <input_notebook.ipynb> <output_script.py>")
        sys.exit(1)

    input_notebook = sys.argv[1]
    output_script = sys.argv[2]

    if not os.path.exists(input_notebook):
        print(f"Error: Input notebook {input_notebook} does not exist")
        sys.exit(1)

    convert_notebook_to_python(input_notebook, output_script)