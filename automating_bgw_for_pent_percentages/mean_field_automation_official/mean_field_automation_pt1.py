import os
import subprocess
import pandas as pd

# parameters that can be changed
#NUMBER OF STRUCTURES#
num_of_structs = 2

#kgrid parameters
WFN_header = [
    [4, 4, 2],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]]
WFNq_header = [
    [4, 4, 2],
    [0.0, 0.0, 0.0],
    [0.005, 0.005, 0.01]]
WFN_fi_header = [
        [8, 8, 4],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]]

#pw2bgw parameters
WFN_parameters_pw2bgw = {
    'wfng_nk1': 4,
    'wfng_nk2': 4,
    'wfng_nk3': 2,
    'wfng_dk1': 0,
    'wfng_dk2': 0,
    'wfng_dk3': 0,
    'vxc_flag': '.true.',
    'rhog_flag': '.true.'
}
WFNq_parameters_pw2bgw = {
    'wfng_nk1': 4,
    'wfng_nk2': 4,
    'wfng_nk3': 2,
    'wfng_dk1': 0.02,
    'wfng_dk2': 0.02,
    'wfng_dk3': 0.02,
    'vxc_flag': '.false.',
    'rhog_flag': '.false.'
}
WFN_fi_parameters_pw2bgw = {
    'wfng_nk1': 8,
    'wfng_nk2': 8,
    'wfng_nk3': 4,
    'wfng_dk1': 0,
    'wfng_dk2': 0,
    'wfng_dk3': 0,
    'vxc_flag': '.false.',
    'rhog_flag': '.false.'
}

#FUNCTIONS
def update_scf_coordinates(template_file, csv_file, output_file):
    """
    Reads atomic coordinates from a CSV and updates an SCF template file.

    Args:
        template_file (str): Path to the input template file ('scf_template.in').
        csv_file (str): Path to the CSV file with new coordinates ('91cm-1_struc_1.csv').
        output_file (str): Path for the output file ('scf.in').
    """
    print(f"Reading new coordinates from '{csv_file}'...")
    # [cite_start]Read the specified columns from the CSV into a pandas DataFrame [cite: 1]
    df = pd.read_csv(csv_file)
    new_coords = df[['X_T_dep', 'Y_T_dep', 'Z_T_dep']].values.tolist()

    print(f"Reading structure from template '{template_file}'...")
    # Read all lines from the template file
    with open(template_file, 'r') as f:
        template_lines = f.readlines()

    # Prepare the list for the new file content
    output_lines = []

    try:
        # Find the starting line of the atomic positions block
        atomic_pos_index = template_lines.index('ATOMIC_POSITIONS angstrom\n')
        # The coordinates start on the line immediately after the header
        coord_start_index = atomic_pos_index + 1
    except ValueError:
        print(f"Error: Could not find 'ATOMIC_POSITIONS angstrom' in '{template_file}'.")
        return

    # Add all lines from the template up to the coordinates
    output_lines.extend(template_lines[:coord_start_index])

    print(f"Replacing {len(new_coords)} atomic coordinates...")
    # Iterate through the new coordinates and the corresponding lines in the template
    for i, coords in enumerate(new_coords):
        # Get the atom symbol (e.g., 'C', 'H') from the original line
        original_line = template_lines[coord_start_index + i]
        atom_symbol = original_line.strip().split()[0]

        # Get the new coordinates
        x, y, z = coords

        # Create the new line with the original atom symbol and new coordinates,
        # formatted for alignment.
        new_line = f"  {atom_symbol:<2s} {x:18.16f}  {y:18.16f}  {z:18.16f}\n"
        output_lines.append(new_line)

    # Find the end of the coordinate block in the original file
    coord_end_index = coord_start_index + len(new_coords)
    # Add the remaining lines from the template to the output
    output_lines.extend(template_lines[coord_end_index:])

    # Write the combined content to the new output file
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    print(f"✅ Successfully created '{output_file}' with updated coordinates.")



def generate_kgrid_input(csv_path, template_path, output_path, header_vectors):
    """
    Generates a 'kgrid.inp' file by updating a template with new coordinates
    and user-defined header vectors.

    Args:
        csv_path (str): The file path for the source CSV coordinates.
        template_path (str): The file path for the input template.
        output_path (str): The file path for the final output.
        header_vectors (list): A list of 3 lists, each containing 3 numbers for the new header.
    """
    print(f"Reading coordinates from '{csv_path}'...")
    # Read only the necessary columns from the provided CSV file
    df = pd.read_csv(csv_path)
    new_coords = df[['X_T_dep', 'Y_T_dep', 'Z_T_dep']].values.tolist()

    print(f"Reading template from '{template_path}'...")
    # Read all lines from the template file
    with open(template_path, 'r') as f:
        template_lines = f.readlines()

    # This list will hold the lines for the new file
    output_content = []

    # --- Step 1: Replace the header ---
    for k, vector in enumerate(header_vectors):
        # The first line (index 0) will be formatted as integers
        if k == 0:
            line = f"   {int(vector[0])}   {int(vector[1])}   {int(vector[2])}\n"
        # All other lines will be formatted as floats without extra padding
        else:
            line = f"   {vector[0]}   {vector[1]}   {vector[2]}\n"
        output_content.append(line)

    # --- Step 2: Add the rest of the file and replace coordinates ---
    # We assume the line with the number of atoms is the 4th line (index 3)
    # and the coordinates start on the 5th line.
    coord_start_index_in_template = 9

    # Add the line that specifies the number of atoms
    for j in range(3, 9):
        output_content.append(template_lines[j])

    print(f"Replacing {len(new_coords)} atomic coordinates...")
    for i in range(len(new_coords)):
        # Get the original line from the template to extract the leading integer
        original_line = template_lines[coord_start_index_in_template + i]
        parts = original_line.strip().split()
        leading_integer = parts[0]

        # Get the new coordinate values for this atom
        x, y, z = new_coords[i]

        # Format the new line, preserving the leading integer
        new_line = f"    {leading_integer}     {x:.10f}   {y:.10f}   {z:.10f}\n"
        output_content.append(new_line)
    #add in last lines
    for j in range(81,85):
        output_content.append(template_lines[j])

    # --- Step 3: Write the output file ---
    with open(output_path, 'w') as f:
        f.writelines(output_content)

    print(f"✅ Successfully created '{output_path}'.")


def update_pw2bgw_parameters(template_path, output_path, params_to_change):
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

#goes into SCF, makes scf.in, returns to struct_x directory
def SCF_dir(template_file, csv_file, output_file):
    os.makedirs("SCF", exist_ok=True)
    print("Made SCF inside " + dir_name)
    os.chdir("SCF")
    print(f"Current directory: {os.getcwd()}")
    update_scf_coordinates(template_file, csv_file, output_file)
    os.chdir('../')
    print(f"Current directory: {os.getcwd()}")

#goes into WFN directories, makes kgrid.in and pw2bgw.in, returns to struct_x directory
#must specify three different headers for WFN, WFNq, WFN_fi
def WFNs_dirs(csv_path, template_path_kgrid, template_path_pw2bgw, output_path_kgrid, output_path_pw2bgw, header_vectors, wfn_dir_name,params):
    os.makedirs(wfn_dir_name, exist_ok=True)
    print("Made "+wfn_dir_name+" inside " + dir_name)
    os.chdir(wfn_dir_name)
    print(f"Current directory: {os.getcwd()}")
    generate_kgrid_input(csv_path, template_path_kgrid, output_path_kgrid, header_vectors)
    update_pw2bgw_parameters(template_path=template_path_pw2bgw, output_path=output_path_pw2bgw,params_to_change=params)
    os.chdir('../')
    print(f"Current directory: {os.getcwd()}")



#RUN STARTS#

#loop that:
#   - makes x_num of structure files
#   - after 1 structure file is made:
#       -  Creates SCF folder; Puts new scf.in the folder
#        - Creates WFN, WFNq, and WFN_fi; Puts kgrid.in and pw2bgw.in in the folders
for i in range(1,num_of_structs+1):
    dir_name = f'struct_{i}'
    os.makedirs(dir_name, exist_ok=True)
    print("Directory " + dir_name + " made")
    os.chdir(dir_name)
    print("Inside directory " + dir_name)
    # Making SCF Directory; Putting scf.in files inside; returns to master directory
    SCF_dir(template_file="../../scf_template.in", csv_file="../../91cm-1_struc_"+str(i)+".csv", output_file="scf.in")
    #MAKING WFN DIRECTORIES: Puts kgrid.in and pw2bgw.in files into created WFN, WFNq, and WFN_fi folders; returns to master directory
    ##WFNq
    WFNs_dirs(wfn_dir_name="WFN", template_path_kgrid="../../kgrid_template.inp",
              csv_path="../../91cm-1_struc_"+str(i)+".csv", output_path_kgrid="../master/kgrid.inp", header_vectors=WFN_header,
              template_path_pw2bgw="../../pw2bgw_template.inp", output_path_pw2bgw="./pw2bgw.inp", params=WFN_parameters_pw2bgw)
    ##WFNq
    WFNs_dirs(wfn_dir_name="WFNq", template_path_kgrid="../../kgrid_template.inp",
              csv_path="../../91cm-1_struc_" + str(i) + ".csv", output_path_kgrid="../master/kgrid.inp", header_vectors=WFNq_header,
              template_path_pw2bgw="../../pw2bgw_template.inp", output_path_pw2bgw="./pw2bgw.inp", params=WFNq_parameters_pw2bgw)
    ##WFN_fi
    WFNs_dirs(wfn_dir_name="WFN_fi", template_path_kgrid="../../kgrid_template.inp",
              csv_path="../../91cm-1_struc_" + str(i) + ".csv", output_path_kgrid="../master/kgrid.inp", header_vectors=WFN_fi_header,
              template_path_pw2bgw="../../pw2bgw_template.inp", output_path_pw2bgw="./pw2bgw.inp", params=WFN_fi_parameters_pw2bgw)
    os.chdir('../')

    #print(f"Directory after going back: {os.getcwd()}") #Used to double check which directory we are in

print(f"Current Directory{os.getcwd()}")

#Runs bash_script_1.sub
try:
    # check=True will raise an exception if the script fails (returns a non-zero exit code).
    # capture_output=True saves the script's output.
    # text=True decodes the output as a string (instead of bytes).
    result = subprocess.run(
        ["qsub", "bash_script_1.sub"],
        check=True,
        capture_output=True,
        text=True
    )

    # Print the standard output from the bash script
    print("Script output:")
    print(result.stdout)
except FileNotFoundError:
    print("Error: 'bash' script not found in the current directory.")
except subprocess.CalledProcessError as e:
    print(f"Error: Script failed with exit code {e.returncode}")
    print("Standard Error:")
    print(e.stderr)

#A reminder!
print("Run mean_field_automation_pt2.py after bash_script_1.sub is complete.")
