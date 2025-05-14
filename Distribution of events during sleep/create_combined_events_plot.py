#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a combined events plot showing sleep stages and respiratory events.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
from analyze_arousal_file import extract_events_from_arousal_file, check_and_install_dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_combined_events_plot(data, output_file=None):
    """
    Create a combined plot showing all events (sleep stages and respiratory events) on the same timeline.

    Args:
        data: Data dictionary from extract_events_from_arousal_file
        output_file: Path to save the visualization (optional)
    """
    if 'error' in data:
        logger.error(f"Cannot visualize data with errors: {data['error']}")
        return

    # Use default font settings to avoid Chinese font issues
    try:
        # Reset to default parameters
        plt.rcParams.update(plt.rcParamsDefault)
        # Set basic plotting parameters
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        # Ensure we're using English locale
        import locale
        locale.setlocale(locale.LC_ALL, 'C')
    except Exception as e:
        logger.warning(f"Error setting font: {str(e)}")

    record_info = data['record_info']
    sleep_stages = data['sleep_stages']
    respiratory_events = data['respiratory_events']

    # Create figure with a more compact aspect ratio
    plt.figure(figsize=(8, 6))  # Made width even smaller for more compact display

    # Define event types and their y-positions and colors
    # Use consistent spacing between all event types
    event_types = {
        'Wake': {'y': 1, 'color': 'blue', 'count': 0},
        'N1': {'y': 2, 'color': 'orange', 'count': 0},
        'N2': {'y': 3, 'color': 'green', 'count': 0},
        'N3': {'y': 4, 'color': 'red', 'count': 0},
        'REM': {'y': 5, 'color': 'purple', 'count': 0},
        'hypopnea': {'y': 6, 'color': 'brown', 'count': 0},
        'apnea': {'y': 7, 'color': 'pink', 'count': 0}  # Changed from 8 to 7 for consistent spacing
    }

    # Plot sleep stages
    for stage in sleep_stages:
        stage_name = stage['stage']
        if stage_name in event_types:
            time_sec = stage['time_seconds']
            y_pos = event_types[stage_name]['y']
            color = event_types[stage_name]['color']
            plt.plot(time_sec, y_pos, 'o', color=color, markersize=5)  # Increased marker size from 3 to 5
            event_types[stage_name]['count'] += 1

    # Plot respiratory events
    for event in respiratory_events:
        event_type = event['type']
        if event_type in event_types:
            # Use the midpoint of the event
            time_sec = (event['start_time'] + event['end_time']) / 2
            y_pos = event_types[event_type]['y']
            color = event_types[event_type]['color']
            plt.plot(time_sec, y_pos, 'o', color=color, markersize=5)  # Increased marker size from 3 to 5
            event_types[event_type]['count'] += 1

    # Set y-axis ticks and labels
    y_ticks = []
    y_labels = []
    legend_elements = []

    for event_name, info in event_types.items():
        if info['count'] > 0:  # Only include events that are present
            y_ticks.append(info['y'])
            y_labels.append(event_name)
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=info['color'], markersize=10,  # Increased from 8 to 10
                                         label=f"{event_name} ({info['count']})"))

    # Ensure y-ticks are in the correct order (from bottom to top)
    y_ticks_sorted = sorted(y_ticks)
    y_labels_sorted = [next(event_name for event_name, info in event_types.items()
                          if info['y'] == y_tick and info['count'] > 0) for y_tick in y_ticks_sorted]

    plt.yticks(y_ticks_sorted, y_labels_sorted)
    plt.ylabel('Event ID')

    # Set x-axis to show time in seconds and limit the range
    plt.xlabel('Time (s)')

    # Calculate appropriate x-axis limits
    # Get the maximum time from events
    max_time = 0
    for stage in sleep_stages:
        max_time = max(max_time, stage['time_seconds'])

    for event in respiratory_events:
        max_time = max(max_time, event['end_time'])

    # Add a small margin (5%) to the maximum time
    max_time = max_time * 1.05

    # Set x-axis limits
    plt.xlim(0, max_time)

    # Add title
    plt.title(f'Combined Events for {record_info["record_name"]}')

    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Adjust layout
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Combined events plot saved to {output_file}")
    else:
        plt.show()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create combined events plot from .arousal file')
    parser.add_argument('--file', '-f', default='/root/autodl-tmp/physionet.org/files/challenge-2018/1.0.0/training/tr13-0076/tr13-0076.arousal',
                        help='Path to the .arousal file (default: %(default)s)')
    parser.add_argument('--output-dir', '-d', default='/root/autodl-tmp/processed_data/pc18/sdb',
                        help='Output directory for results (default: %(default)s)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename based on input file
    record_name = os.path.basename(args.file).split('.')[0]
    output_file = os.path.join(args.output_dir, f"{record_name}_combined_events.png")

    # Check and install dependencies
    check_and_install_dependencies()

    # Extract events from arousal file
    logger.info(f"Analyzing arousal file: {args.file}")
    data = extract_events_from_arousal_file(args.file)

    # Check for errors
    if 'error' in data:
        logger.error(f"Error analyzing file: {data['error']}")
        return

    # Create combined events plot
    logger.info(f"Creating combined events plot to {output_file}")
    create_combined_events_plot(data, output_file)
    print(f"Combined events plot saved to: {output_file}")

if __name__ == "__main__":
    main()
