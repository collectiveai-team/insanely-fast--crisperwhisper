import yaml

def extract_python_packages(input_file_path, output_file_path):
    print(f"Extracting Python packages from {input_file_path} and writing to {output_file_path}")

    with open(input_file_path, 'r') as file:
        # Parse the YAML file
        data = yaml.safe_load(file)

    # Extract the list of Python packages
    build = data.get('build', {})
    python_packages = build.get('python_packages', [])

    if not python_packages:
        raise ValueError(f"No Python packages found in {input_file_path}")

    # Write the Python packages to the requirements.txt file
    with open(output_file_path, 'w') as file:
        for package in python_packages:
            file.write(f"{package}\n")

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_file_path", type=str)
    parser.add_argument("output_file_path", type=str)

    args = parser.parse_args()

    extract_python_packages(args.input_file_path, args.output_file_path)
