import re

def sanitize_filename(filename):
    # Remove invalid characters and limit the length
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    return filename[:255]

def split_file(input_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # Split the content based on the delimiter
    sections = content.split('// -----// ')

    # Remove empty sections
    sections = [section for section in sections if section.strip()]

    # Process each section
    for i, section in enumerate(sections):
        # Extract the name from the first line of the section
        first_line = section.split('\n')[0]
        name = first_line.split(' //-----')[0].strip()
        
        # Sanitize the file name
        sanitized_name = sanitize_filename(name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("'", ""))
        
        # Create a new file for each section
        output_file = f'part{i}_{sanitized_name}.mlir'
        with open(output_file, 'w') as out_f:
            out_f.write('// -----// ' + section)

        print(f'Created file: {output_file}')

if __name__ == "__main__":
    input_file = 'dump_tma1_mlir.mlir'  # Replace with your input file name
    split_file(input_file)
