"""
Event encoding utilities.

Functions to convert raw observations into human-readable event dictionaries.
"""

def event_to_dict_from_context(obs, event_types, attributes):
    """
    Convert raw obs['context'] from OpenTheChestsGym to a readable dictionary.

    Args:
        obs (dict): Environment observation containing a 'context' field.
        event_types (list): List of all event type symbols.
        attributes (dict): Dictionary containing foreground and background color mappings.

    Returns:
        dict: Parsed event dictionary with symbol, colors, and timing info.
    """
    event = obs["context"]
    return {
        "symbol": event_types[event.type],
        "bg_color": attributes["bg"][event.attributes["bg"]],
        "symbol_color": attributes["fg"][event.attributes["fg"]],
        "start_time": event.start,
        "end_time": event.end,
    }


def event_to_dict_from_gym(obs, event_types, attributes):
    """
    Convert observation from Gym-based OpenTheChests environments to a readable dictionary.

    Args:
        obs (dict): Observation containing fields 'e_type', 'bg', 'fg', 'start', and 'end'.
        event_types (list): List of all event type symbols.
        attributes (dict): Dictionary containing foreground and background color mappings.

    Returns:
        dict: Parsed event dictionary with symbol, colors, and timing info.
    """
    return {
        "symbol": event_types[obs["e_type"]],
        "bg_color": attributes["bg"][obs["bg"]],
        "symbol_color": attributes["fg"][obs["fg"]],
        "start_time": obs["start"][0],
        "end_time": obs["end"][0],
    }
