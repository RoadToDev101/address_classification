import csv


def remove_prefix(name: str, location_type: str) -> str:
    """Remove administrative prefixes from Vietnamese location names"""
    # Clean the input name
    name = name.strip()

    prefixes = {
        "province": ["Thành phố ", "Tỉnh "],
        "district": ["Huyện ", "Quận ", "Thị xã ", "Thành phố "],
        "ward": ["Phường ", "Xã ", "Thị trấn "],
    }

    for prefix in prefixes.get(location_type, []):
        if name.startswith(prefix):
            result = name[len(prefix) :].strip()
            return result
    return name


def convert_csv_to_txt(csv_path, location_type):
    """Convert CSV file to TXT format with prefixes removed"""
    txt_path = csv_path.replace(".csv", ".txt")

    with open(csv_path, "r", encoding="utf-8-sig") as csvfile:  # utf-8-sig handles BOM
        reader = csv.reader(csvfile)

        with open(txt_path, "w", encoding="utf-8") as txtfile:
            for row in reader:
                if row:  # Check if row is not empty
                    # Get the first column (location name)
                    location_name = row[0].strip()

                    # Remove prefix
                    clean_name = remove_prefix(location_name, location_type)

                    # Write to txt file
                    txtfile.write(clean_name + "\n")

    print(f"Converted {csv_path} to {txt_path} with prefixes removed")


if __name__ == "__main__":
    # Convert all CSV files
    convert_csv_to_txt("data/provinces.csv", "province")
    convert_csv_to_txt("data/districts.csv", "district")
    convert_csv_to_txt("data/wards.csv", "ward")
