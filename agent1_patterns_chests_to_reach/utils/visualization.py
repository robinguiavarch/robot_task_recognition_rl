"""
Timeline visualization of observed events.

This module provides a function to plot event sequences over time or visualize them directly from the environment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
from agent1_patterns_chests_to_reach.utils.data_collectors import collect_observations


def plot_event_timeline(events, start_time=0, end_time=None, env_name=""):
    """
    Visualize events on a timeline using color-coded bars.

    Args:
        events (list): List of dictionaries containing event data.
        start_time (float): Start of the x-axis (time).
        end_time (float or None): End of the x-axis (time). If None, it is inferred from data.
        env_name (str): Optional name to include in the plot title.
    """
    if end_time is None:
        end_time = max(event["end_time"] for event in events)

    fig, ax = plt.subplots(figsize=(15, 5))
    last_event_end_times = []
    height = 1

    for event in events:
        name = event["symbol"]
        start = event["start_time"]
        end = event["end_time"]
        color = event["bg_color"]
        text_color = event["symbol_color"]

        # Find row to avoid overlap
        line = 0
        while line < len(last_event_end_times):
            if start >= last_event_end_times[line]:
                break
            line += 1

        if line == len(last_event_end_times):
            last_event_end_times.append(end)
        else:
            last_event_end_times[line] = end

        y_pos = line * (height + 0.5)
        rect = patches.Rectangle(
            (start, y_pos), max(end - start, 0.1), height, color=color, alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(
            start + (end - start) / 2,
            y_pos + height / 2,
            name,
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlim(start_time, end_time)
    ax.set_ylim(0, len(last_event_end_times) * (height + 0.5))
    ax.set_xlabel("Time")
    ax.set_ylabel("Event Sequences")
    ax.set_title(f"Observed Event Timeline ({env_name})" if env_name else "Event Timeline")
    plt.show()


def visualize_env_timeline(env_name: str, num_steps: int = 30):
    """
    Collect observations and display the timeline for a given environment.

    Args:
        env_name (str): Environment ID.
        num_steps (int): Number of steps to simulate.
    """
    events = collect_observations(env_name, num_steps=num_steps)
    plot_event_timeline(
        events,
        start_time=0,
        end_time=events[-1]["end_time"],
        env_name=env_name
    )
