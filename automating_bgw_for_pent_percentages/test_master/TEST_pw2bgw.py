import os


def update_parameters(template_path, output_path, params_to_change):
    """
    Reads a template file, updates specified parameters, and writes to a new file.

    Args:
        template_path (str): The path to the input template file.
        output_path (str): The desired path for the output file.
        params_to_change (dict): A dictionary where keys are the parameter names
                                 and values are the new values.
    """
    print(f"Reading template file: '{template_path}'")
    with open(template_path, 'r') as f:
        lines = f.readlines()

    output_lines = []
    updated_keys = set()

    print("Updating parameters...")
    # Loop through each line of the template file
    for line in lines:
        # Strip whitespace and split by '=' to check for a parameter
        parts = [p.strip() for p in line.split('=')]

        # Check if the line contains a parameter we want to change
        if len(parts) == 2 and parts[0] in params_to_change:
            key = parts[0]
            new_value = params_to_change[key]

            # Create the new line, preserving original spacing and comments
            # This assumes the value is the second part of the line.
            original_value_and_comment = parts[1].split('!')[0]
            new_line = line.replace(original_value_and_comment, f" {new_value} ")

            output_lines.append(new_line)
            updated_keys.add(key)
            print(f"  - Changed '{key}' to '{new_value}'")
        else:
            # If the line isn't a parameter we're changing, keep it as is
            output_lines.append(line)

    # --- Write the updated content to the new file ---
    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    print(f"\n✅ Successfully created output file: '{output_path}'")

    # Warn user if some keys they specified were not found in the template
    not_found = set(params_to_change.keys()) - updated_keys
    if not_found:
        print(f"\n⚠️ Warning: The following parameters were not found in the template: {', '.join(not_found)}")


# --- Main execution block ---
if __name__ == "__main__":
    # ===================================================================
    # ==> 1. DEFINE the name of your template and output files here.
    # ===================================================================
    TEMPLATE_FILE = '../mean_field_automation_official/pw2bgw_template.inp'
    OUTPUT_FILE = 'pw2bgw_custom.inp'  # You can change this name

    # ===================================================================
    # ==> 2. DEFINE the parameters and their new values here.
    # ===================================================================
    new_parameters = {
        'wfng_nk1': 4,
        'wfng_nk2': 4,
        'wfng_nk3': 2,
        'wfng_dk1': 0,
        'wfng_dk2': 0,
        'wfng_dk3': 0,
        'vxc_flag': '.true.',
        'rhog_flag': '.false.'
    }
    # ===================================================================

    # Run the function with your defined settings
    update_parameters(
        template_path=TEMPLATE_FILE,
        output_path=OUTPUT_FILE,
        params_to_change=new_parameters
    )