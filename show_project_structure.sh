#!/bin/bash

# Script to display complete project file structure
# Includes hidden files and all contents for development purposes

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default settings
SHOW_SIZE=false
EXCLUDE_GIT_INTERNALS=false
OUTPUT_FILE=""

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -s    Show file sizes"
    echo "  -g    Exclude .git internal files (still shows .gitignore, etc.)"
    echo "  -o    Output to file (example: -o structure.txt)"
    echo "  -h    Show this help message"
    echo ""
    echo "Example: $0 -s -g -o project_structure.txt"
}

# Parse command line options
while getopts "sgo:h" opt; do
    case $opt in
        s)
            SHOW_SIZE=true
            ;;
        g)
            EXCLUDE_GIT_INTERNALS=true
            ;;
        o)
            OUTPUT_FILE="$OPTARG"
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done

# Function to check if tree command is available
check_tree_command() {
    if command -v tree &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to generate tree using the tree command
generate_tree_with_command() {
    local tree_cmd="tree -a -F --dirsfirst"
    
    # Add size option if requested
    if [ "$SHOW_SIZE" = true ]; then
        tree_cmd="$tree_cmd -h"
    fi
    
    # Add git exclusion if requested
    if [ "$EXCLUDE_GIT_INTERNALS" = true ]; then
        tree_cmd="$tree_cmd -I '.git'"
    fi
    
    # Execute tree command
    if [ -n "$OUTPUT_FILE" ]; then
        $tree_cmd > "$OUTPUT_FILE"
        echo -e "${GREEN}Project structure saved to: $OUTPUT_FILE${NC}"
    else
        $tree_cmd
    fi
}

# Fallback function to generate tree structure without tree command
generate_tree_manual() {
    local prefix="$1"
    local dir="$2"
    local is_last="$3"
    
    # Skip .git internals if requested
    if [ "$EXCLUDE_GIT_INTERNALS" = true ] && [[ "$dir" == *"/.git/"* ]]; then
        return
    fi
    
    # Get the basename of the directory
    local basename=$(basename "$dir")
    
    # Print the current directory
    if [ "$dir" != "." ]; then
        echo -n "$prefix"
        if [ "$is_last" = true ]; then
            echo -n "└── "
            prefix="${prefix}    "
        else
            echo -n "├── "
            prefix="${prefix}│   "
        fi
        
        # Add directory marker and size if requested
        if [ -d "$dir" ]; then
            echo -n -e "${BLUE}${basename}/${NC}"
        else
            echo -n "$basename"
        fi
        
        if [ "$SHOW_SIZE" = true ] && [ -f "$dir" ]; then
            local size=$(ls -lh "$dir" 2>/dev/null | awk '{print $5}')
            echo " ($size)"
        else
            echo
        fi
    fi
    
    # If it's a directory, recurse into it
    if [ -d "$dir" ]; then
        # Get all entries (including hidden)
        local entries=()
        while IFS= read -r -d '' entry; do
            entries+=("$entry")
        done < <(find "$dir" -maxdepth 1 -mindepth 1 -print0 | sort -z)
        
        local count=${#entries[@]}
        local index=0
        
        for entry in "${entries[@]}"; do
            ((index++))
            local is_last_entry=false
            if [ $index -eq $count ]; then
                is_last_entry=true
            fi
            
            if [ "$dir" = "." ]; then
                generate_tree_manual "" "$entry" "$is_last_entry"
            else
                generate_tree_manual "$prefix" "$entry" "$is_last_entry"
            fi
        done
    fi
}

# Main execution
echo -e "${GREEN}=== Fantasy Football AI Project Structure ===${NC}"
echo -e "${YELLOW}Generated on: $(date)${NC}"
echo ""

# Check if we're in a git repository
if [ -d .git ]; then
    echo -e "${BLUE}Git repository detected${NC}"
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo -e "${BLUE}Current branch: $CURRENT_BRANCH${NC}"
    echo ""
fi

# Generate the tree structure
if check_tree_command; then
    echo -e "${GREEN}Using 'tree' command...${NC}"
    echo ""
    generate_tree_with_command
else
    echo -e "${YELLOW}Warning: 'tree' command not found. Using fallback method...${NC}"
    echo -e "${YELLOW}For better output, install tree: ${NC}"
    echo -e "${YELLOW}  Ubuntu/Debian: sudo apt-get install tree${NC}"
    echo -e "${YELLOW}  macOS: brew install tree${NC}"
    echo ""
    
    if [ -n "$OUTPUT_FILE" ]; then
        # Redirect output to file
        {
            echo "."
            generate_tree_manual "" "." false
        } > "$OUTPUT_FILE"
        echo -e "${GREEN}Project structure saved to: $OUTPUT_FILE${NC}"
    else
        echo "."
        generate_tree_manual "" "." false
    fi
fi

# Show summary statistics
echo ""
echo -e "${GREEN}=== Summary Statistics ===${NC}"
echo "Total directories: $(find . -type d | wc -l)"
echo "Total files: $(find . -type f | wc -l)"
echo "Python files: $(find . -name "*.py" -type f | wc -l)"
echo "Hidden files/dirs: $(find . -name ".*" | wc -l)"

# Show size of major directories if they exist
echo ""
echo -e "${GREEN}=== Directory Sizes ===${NC}"
for dir in "src" "scripts" "tests" "data" "models" ".git" "__pycache__" ".venv" "venv"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "$dir/: $size"
    fi
done


# Show only essential project files

echo "=== Essential Fantasy Football AI Project Files ==="
echo ""
echo "Backend Python files:"
find backend -name "*.py" -not -path "*/__pycache__/*" -not -path "*/.ipynb_checkpoints/*" | sort
echo ""
echo "Frontend Python files:"
find frontend -name "*.py" -not -path "*/__pycache__/*" -not -path "*/.ipynb_checkpoints/*" | sort
echo ""
echo "Configuration files:"
ls -la *.yml *.md *.sh *.sql Makefile .gitignore .env 2>/dev/null | grep -v ".ipynb_checkpoints"
echo ""
echo "Infrastructure:"
find infrastructure -type f -not -path "*/.ipynb_checkpoints/*" | sort