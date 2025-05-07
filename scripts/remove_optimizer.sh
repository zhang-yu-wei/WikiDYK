#!/bin/bash

# Check if a directory path was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 /path/to/your/folder"
    exit 1
fi

DIRECTORY="$1"

# Check if the provided path is a valid directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: '$DIRECTORY' is not a valid directory"
    exit 1
fi

echo "Searching for 'optimizer.pt' files in '$DIRECTORY'..."

# Find all optimizer.pt files and list them before removal
FOUND_FILES=$(find "$DIRECTORY" -name "optimizer.pt" -type f)

# Check if any files were found
if [ -z "$FOUND_FILES" ]; then
    echo "No 'optimizer.pt' files found in '$DIRECTORY'"
    exit 0
fi

# Count the number of files found
FILE_COUNT=$(echo "$FOUND_FILES" | wc -l)
echo "Found $FILE_COUNT 'optimizer.pt' files:"
echo "$FOUND_FILES"

# Ask for confirmation before deletion
read -p "Do you want to delete these files? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Operation cancelled"
    exit 0
fi

# Remove the files
find "$DIRECTORY" -name "optimizer.pt" -type f -delete

echo "All 'optimizer.pt' files have been successfully removed"