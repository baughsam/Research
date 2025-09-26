# create_epsilon_input.py

import sys
import os
import subprocess

# --- Configuration ---
#Number of Structures
num_of_structs = 10
# You can change these values
KGRID_FILE = 'kgrid.out'
epsilon_template = 'epsilon_template'
epsilon_output = 'epsilon.inp'
sigma_template = 'sigma_template'
sigma_output = 'sigma.inp'
kernel_template = 'kernel_template'
absorption_template = 'absorption_template'

# This is the marker that the script will look for in the template file.
# It can be changed to match your template.
QPOINTS_BEGIN_MARKER_eps = 'begin qpoints'
QPOINTS_BEGIN_MARKER_sig = 'begin kpoints'
QPOINTS_END_MARKER = 'end'

# This sets the value for the 4th column on the very first line of k-points.
# All other k-points will have 0.0 in this column.
FIRST_QPOINT_WEIGHT_eps = 1.0

#first q_point might need to be shifted
first_q_point_eps = [0.005,0.005,0.01]
first_q_point_sig = [0,0,0]
# --- End of Configuration ---


def parse_kgrid_file(filename):
    """
    Parses the kgrid.out file to extract the number of k-points and their coordinates.
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1) # Exit the script if the file doesn't exist

    num_kpoints = 0
    kpoints_data = []
    reading_kpoints = False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        # Find the line that specifies the number of k-points


        if parts[0].upper() == 'K_POINTS':
            try:
                num_kpoints = int(lines[1]) + 1 #+1 ~ to take into account the reading of the integer at the beginning
                reading_kpoints = True
                continue # Move to the next line to start reading points
            except (ValueError, IndexError):
                print(f"Error: Could not parse number of k-points from line: '{line.strip()}'")
                sys.exit(1)

        # Once we know the number, start reading the k-point lines
        if reading_kpoints and len(kpoints_data) < num_kpoints:
            # We only need the first 3 columns for the coordinates
            kpoints_data.append(parts[0:4])

    if not kpoints_data or len(kpoints_data) != num_kpoints:
        print(f"Error: Found {len(kpoints_data)} k-points, but expected {num_kpoints} from '{filename}'.")
        sys.exit(1)

    #delete integer specifying number of k-points at the beginning of list kpoints_data
    kpoints_data.pop(0)
    #print(kpoints_data)
    print(f"Successfully read {num_kpoints - 1} k-points from '{KGRID_FILE}'.")
    return kpoints_data


def create_epsilon_inp(kpoints_, template_file, output_file, first_q_point_list, first_q_point_weight, begin_xpoints, is_q):
    """
    Creates the epsilon.inp file by inserting k-point data into a template.
    """
    try:
        with open(template_file, 'r') as f:
            template_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The template file '{template_file}' was not found.")
        sys.exit(1)

    try:
        # Find the line numbers for the start and end markers
        start_index = next(i for i, line in enumerate(template_lines) if begin_xpoints in line)
        end_index = next(i for i, line in enumerate(template_lines) if QPOINTS_END_MARKER in line)
    except StopIteration:
        print(f"Error: Could not find '{begin_xpoints}' and/or '{QPOINTS_END_MARKER}' in '{template_file}'.")
        sys.exit(1)

    # --- Build the new block of q-points ---
    new_qpoints_block = []
    for i, point in enumerate(kpoints_):
        x, y, z , integer= point
        #print(point)
        # Set the weight for the 4th column
        weight = first_q_point_weight if i == 0 else 0.0
        # Format the line with proper spacing
        if i == 0:
            x,y,z = first_q_point_list
        if is_q:
            formatted_line = f"    {float(x):.7f} {float(y):.7f} {float(z):.7f} {float(integer):.2} {int(weight)}\n"
            new_qpoints_block.append(formatted_line)
        elif not is_q:
            formatted_line = f"    {float(x):.7f} {float(y):.7f} {float(z):.7f} {float(integer):.2}\n"
            new_qpoints_block.append(formatted_line)


    # --- Combine the template parts with the new block ---
    # Part 1: Everything from the template before the start marker
    final_content = template_lines[:start_index + 1]
    # Part 2: The newly generated q-points block
    final_content.extend(new_qpoints_block)
    # Part 3: Everything from the end marker to the end of the template file
    final_content.extend(template_lines[end_index:])

    # --- Write the final content to the output file ---
    with open(output_file, 'w') as f:
        f.writelines(final_content)

    print(f"Successfully created '{output_file}'.")


if __name__ == '__main__':
    #kpoints = parse_kgrid_file("kgrid.out")
    #create_epsilon_inp(kpoints_= kpoints, template_file= epsilon_template,output_file="epsilon_test", first_q_point_list=first_q_point_eps, first_q_point_weight=1,begin_xpoints=QPOINTS_BEGIN_MARKER_eps, is_q=True)
    #Make folders,copy/link, run

    #Make folders: EPSILON, SIGMA, KERNEL, ABSORPTION
    ##FOLDER_NAMES
    epsilon_folder = "EPSILON"
    sigma_folder = "SIGMA"
    kernel_folder = "KERNEL"
    absorption_folder = "ABSORPTION"
    folders = [epsilon_folder,sigma_folder,kernel_folder,absorption_folder]


    print(os.getcwd())
    #Make input files in respective folders
    for i in range(1, num_of_structs + 1):
        #going inside folder
        dir_name = f'struct_{i}'
        os.chdir(dir_name)
        print("Inside directory " + dir_name)

        ##MAKING FOLDERS
        for created_dir in folders:
            try:
                os.mkdir(created_dir)
                print(f"Folder '{created_dir}' created")
            except FileExistsError:
                continue

        # Linking in each folder
        # EPSILON#
        os.symlink("../WFN/WFN", "./" + epsilon_folder + "/WFN")
        os.symlink("../WFNq/WFN", "./" + epsilon_folder + "/WFNq")
        # SIGMA#
        os.symlink("../WFN/WFN", "./" + sigma_folder + "/WFN_inner")
        os.symlink("../WFN/vxc.dat", "./" + sigma_folder + "/vxc.dat")
        os.symlink("../WFN/RHO", "./" + sigma_folder + "/RHO")
        os.symlink("../WFNq/WFN", "./" + sigma_folder + "/WFNq")
        os.symlink("../" + epsilon_folder + "/eps0mat.h5", "./" + sigma_folder + "/eps0mat.h5")
        os.symlink("../" + epsilon_folder + "/epsmat.h5", "./" + sigma_folder + "/epsmat.h5")
        # KERNEL#
        os.symlink("../WFN/WFN", "./" + kernel_folder + "/WFN_co")
        os.symlink("../" + epsilon_folder + "/eps0mat.h5", "./" + kernel_folder + "/eps0mat.h5")
        os.symlink("../" + epsilon_folder + "/epsmat.h5", "./" + kernel_folder + "/epsmat.h5")
        # ABSORPTION#
        os.symlink("../WFN/WFN", "./" + absorption_folder + "/WFN_co")
        os.symlink("../WFN_fi/WFN", "./" + absorption_folder + "/WFN_fi")
        os.symlink("../WFN_fi/WFN", "./" + absorption_folder + "/WFNq_fi")
        os.symlink("../" + epsilon_folder + "/eps0mat.h5", "./" + absorption_folder + "/eps0mat.h5")
        os.symlink("../" + epsilon_folder + "/epsmat.h5", "./" + absorption_folder + "/epsmat.h5")
        os.symlink("../" + sigma_folder + "/eqp1.dat", "./" + absorption_folder + "/eqp_co.dat")
        os.symlink("../" + kernel_folder + "/bsemat.h5", "./" + absorption_folder + "/bsemat.h5")

        #creating input files
        ##parsing kpoint file
        kpoints = parse_kgrid_file(f"./WFN/{KGRID_FILE}")
        ##epsilon.inp
        create_epsilon_inp(kpoints, f"../{epsilon_template}", f"./{epsilon_folder}/{epsilon_output}", first_q_point_eps,
                           FIRST_QPOINT_WEIGHT_eps, QPOINTS_BEGIN_MARKER_eps, is_q=True)
        ##sigma.inp
        create_epsilon_inp(kpoints, f"../{sigma_template}", f"./{sigma_folder}/{sigma_output}", first_q_point_sig,
                           first_q_point_sig, QPOINTS_BEGIN_MARKER_sig, is_q=False)
        ##kernel.inp
        os.system(f"cp ../{kernel_template} ./{kernel_folder}/kernel.inp")
        ##absorption.inp
        os.system(f"cp ../{absorption_template} ./{absorption_folder}/absorption.inp")

        #leaving directory
        os.chdir("../")
        print(f"Directory after going back: {os.getcwd()}")


    #run bashscript
    # Runs bash_script_3.sub
    try:
        # check=True will raise an exception if the script fails (returns a non-zero exit code).
        # capture_output=True saves the script's output.
        # text=True decodes the output as a string (instead of bytes).
        result = subprocess.run(
            ["qsub", "bash_script_3.sub"],
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


    print("bash_script_3.sub is running!")
