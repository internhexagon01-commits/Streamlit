#!/usr/bin/env python3
"""
Script to restore the correct main.py from your colleague's version.
This replaces the current main.py with the correct agent logic.
"""

import shutil
import os

# The correct main.py content is too large to paste here
# Instead, you should:
# 1. Get the correct main.py file from your colleague
# 2. Place it in the project root as "main_correct.py"
# 3. Run this script to replace src/main.py

correct_file = "main_correct.py"
target_file = "src/main.py"
backup_file = "src/main.py.backup"

if not os.path.exists(correct_file):
    print(f"❌ Error: {correct_file} not found!")
    print("\nPlease:")
    print("1. Get the correct main.py from your colleague")
    print("2. Save it as 'main_correct.py' in the project root")
    print("3. Run this script again")
    exit(1)

# Create backup
print(f"Creating backup: {backup_file}")
shutil.copy2(target_file, backup_file)

# Replace with correct version
print(f"Replacing {target_file} with correct version...")
shutil.copy2(correct_file, target_file)

print("\n✅ Done! The correct main.py has been restored.")
print(f"Backup saved as: {backup_file}")
print("\nNow restart your Streamlit app:")
print("  run_streamlit.bat")
