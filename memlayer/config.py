"""
Configuration settings for MemLayer.

This module provides global configuration for controlling behavior across the library.
"""

# Debug mode - controls verbosity of output
# Set to True to see detailed debugging information
# Set to False for production use (only shows key events)
DEBUG_MODE = False

def set_debug_mode(enabled: bool):
    """
    Enable or disable debug mode globally.
    
    When DEBUG_MODE is True:
    - Shows detailed trace events
    - Prints background consolidation progress
    - Shows salience check details
    - Displays search tier selection
    
    When DEBUG_MODE is False:
    - Only shows key initialization messages
    - Only shows final extraction results
    - Minimal output for production use
    
    Args:
        enabled (bool): True to enable debug mode, False to disable
    """
    global DEBUG_MODE
    DEBUG_MODE = enabled

def is_debug_mode() -> bool:
    """
    Check if debug mode is currently enabled.
    
    Returns:
        bool: True if debug mode is enabled, False otherwise
    """
    return DEBUG_MODE
