import csv


def remove_prefix(name: str, location_type: str) -> str:
    """Remove administrative prefixes from Vietnamese location names using flexible matching"""
    import re

    # Clean the input name
    name = name.strip()

    # More comprehensive and flexible prefix patterns
    prefix_patterns = {
        "province": [
            r"^(Thành phố|Tỉnh|TP|T\.?)\s*",  # Thành phố, Tỉnh, TP, T.
            r"^(Province|Prov\.?)\s*",  # English variants
        ],
        "district": [
            r"^(Huyện|Quận|Thị xã|Thành phố|Huyện|Q\.?|H\.?|TX\.?|TP\.?)\s*",  # Vietnamese
            r"^(District|Dist\.?)\s*",  # English variants
        ],
        "ward": [
            r"^(Phường|Xã|Thị trấn|P\.?|X\.?|TT\.?)\s*",  # Vietnamese
            r"^(Ward|Commune|Comm\.?)\s*",  # English variants
        ],
    }

    patterns = prefix_patterns.get(location_type, [])

    for pattern in patterns:
        # Try to match and remove prefix
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            result = name[match.end() :].strip()
            if result:  # Only return if there's something left after removing prefix
                return result

    return name


def remove_duplicates_from_file(input_path: str, output_path: str = None):
    """Remove duplicate lines from a text file while preserving order"""
    if output_path is None:
        output_path = input_path

    seen = set()
    unique_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + "\n")

    return len(unique_lines)


def convert_csv_to_txt(csv_path, location_type):
    """Convert CSV file to TXT format with prefixes removed"""
    txt_path = csv_path.replace(".csv", ".txt")

    # Collect all cleaned names first
    cleaned_names = []

    with open(csv_path, "r", encoding="utf-8-sig") as csvfile:  # utf-8-sig handles BOM
        reader = csv.reader(csvfile)

        for row in reader:
            if row:  # Check if row is not empty
                # Get the first column (location name)
                location_name = row[0].strip()

                # Remove prefix
                clean_name = remove_prefix(location_name, location_type)
                cleaned_names.append(clean_name)

    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in cleaned_names:
        if name and name not in seen:
            seen.add(name)
            unique_names.append(name)

    # Write unique names to txt file
    with open(txt_path, "w", encoding="utf-8") as txtfile:
        for name in unique_names:
            txtfile.write(name + "\n")

    print(
        f"Converted {csv_path} to {txt_path} with prefixes removed. Found {len(unique_names)} unique locations."
    )


if __name__ == "__main__":
    # Convert all CSV files
    convert_csv_to_txt("data/provinces.csv", "province")
    convert_csv_to_txt("data/districts.csv", "district")
    convert_csv_to_txt("data/wards.csv", "ward")
