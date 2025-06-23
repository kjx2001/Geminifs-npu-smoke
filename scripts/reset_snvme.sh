#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"

echo "Starting snvme driver reset..."

# 1. Call unbind.sh to unbind all devices
echo "Step 1: Unbinding all snvme devices..."
bash "$SCRIPT_DIR/unbind.sh"

# 2. Check if there are still snvme related devices in /dev/
echo "Step 2: Checking for remaining snvme devices..."
remaining_devices=$(ls /dev/snvm_control /dev/ssnvme* 2>/dev/null || true)

if [ -n "$remaining_devices" ]; then
    echo "Found remaining snvme devices:"
    echo "$remaining_devices"
    
    # 3. Unload and reinstall kernel module
    echo "Step 3: Reloading kernel module..."
    
    if [ ! -d "$BUILD_DIR" ]; then
        echo "Error: Build directory $BUILD_DIR does not exist"
        exit 1
    fi
    
    cd "$BUILD_DIR"
    
    echo "Unloading kernel module..."
    if ! make rmmod; then
        echo "Warning: Failed to unload kernel module, continuing with installation..."
    fi
    
    echo "Installing kernel module..."
    if ! make insmod; then
        echo "Error: Failed to reinstall kernel module!"
        echo "Please restart the computer and try again."
        exit 1
    fi
    
    echo "Kernel module reloaded successfully."
else
    echo "No remaining snvme devices found, no need to reload kernel module."
fi

echo "snvme driver reset completed!"
