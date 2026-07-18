import os

def parse_input_file(filepath):
    """
    Parses a key-value text configuration file in any order.
    Handles comments (#), blank lines, integers, floats, strings, and tuples.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found at: {filepath}")

    config = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Split key and value (supports 'key = val' or 'key val')
            if '=' in line:
                key, val = line.split('=', 1)
            else:
                # Fallback for projection_tuple ('x','y') style spacing
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, val = parts
                else:
                    continue

            key = key.strip()
            val = val.strip()

            # --- Type Conversion System ---
            # 1. Parse Tuples, e.g., ('x','y')
            if val.startswith('(') and val.endswith(')'):
                inner_content = val[1:-1]
                val = tuple(item.strip().strip("'\"") for item in inner_content.split(',') if item.strip())

            # 2. Parse Strings wrapped in quotes
            elif (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]

            # 3. Parse Numerical values (Int / Float)
            else:
                try:
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass  # Fallback to raw string if numeric conversion fails

            config[key] = val

    return config