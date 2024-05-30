#!/bin/bash

# Get the current directory
CURRENT_DIR="$(pwd)"

# Find all .py files in the current directory
SCRIPTS=$(find "$CURRENT_DIR" -maxdepth 1 -name "*.py")

# Loop through each .py file and open it in a new terminal
for SCRIPT in $SCRIPTS; 
do
    echo "Starting $SCRIPT in a new terminal..."
    
    # Check for the terminal emulator and open a new terminal window for each script
    if command -v gnome-terminal &> /dev/null
    then
        gnome-terminal -- bash -c "python3 '$SCRIPT'; exec bash"
    elif command -v xterm &> /dev/null
    then
        xterm -hold -e "python3 '$SCRIPT'"
    elif command -v konsole &> /dev/null
    then
        konsole --noclose -e "python3 '$SCRIPT'"
    else
        echo "No supported terminal emulator found. Please install gnome-terminal, xterm, or konsole."
        exit 1
    fi
done

echo "All scripts are running in separate terminals."

