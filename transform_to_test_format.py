import json
import os
import sys
import argparse

# Set UTF-8 encoding for console output
sys.stdout.reconfigure(encoding="utf-8")


def transform_json_to_desired_format(
    input_file, output_file, success_filter=True, include_failed=False
):
    """
    Transform JSON from current format to desired format:
    Current: {lat, lon, address, success, error, province, district, ward}
    Desired: {text: address, result: {province, district, ward}}

    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        success_filter (bool): If True, only include entries with success=True
        include_failed (bool): If True, include failed entries with empty result fields
    """

    print(f"Reading from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} entries")

    transformed_data = []

    for entry in data:
        # Check if entry has required fields
        if not entry.get("address"):
            if include_failed:
                # Include failed entries with empty result
                transformed_entry = {
                    "text": entry.get("address", ""),
                    "result": {
                        "province": "",
                        "district": "",
                        "ward": "",
                    },
                }
                transformed_data.append(transformed_entry)
            continue

        # Apply success filter if enabled
        if success_filter and not entry.get("success", False):
            if include_failed:
                # Include failed entries with empty result
                transformed_entry = {
                    "text": entry["address"],
                    "result": {
                        "province": "",
                        "district": "",
                        "ward": "",
                    },
                }
                transformed_data.append(transformed_entry)
            continue

        # Check if entry has valid data (not all null/empty)
        province = entry.get("province", "")
        district = entry.get("district", "")
        ward = entry.get("ward", "")

        # Skip entries where all three fields are null or empty
        if not province and not district and not ward:
            continue

        # Create the new format
        transformed_entry = {
            "text": entry["address"],
            "result": {
                "province": province,
                "district": district,
                "ward": ward,
            },
        }

        transformed_data.append(transformed_entry)

    print(f"Transformed {len(transformed_data)} valid entries")

    # Write the transformed data
    print(f"Writing to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)

    print("Transformation completed!")
    return len(transformed_data)


def preview_sample(input_file, num_samples=5):
    """Preview a few samples of the transformed data"""
    print(f"\nPreviewing first {num_samples} entries from {input_file}:")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, entry in enumerate(data[:num_samples]):
        print(f"\nSample {i + 1}:")
        print(f"  Text: {entry['text']}")
        print(f"  Result: {json.dumps(entry['result'], ensure_ascii=False)}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Transform JSON data to test format with customizable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transformation
  python transform_to_test_format.py -i input.json -o output.json
  
  # Include failed entries with empty results
  python transform_to_test_format.py -i input.json -o output.json --include-failed
  
  # Disable success filtering (include all entries)
  python transform_to_test_format.py -i input.json -o output.json --no-success-filter
  
  # Preview more samples
  python transform_to_test_format.py -i input.json -o output.json --preview 10
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input JSON file path")

    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")

    parser.add_argument(
        "--no-success-filter",
        action="store_true",
        help="Disable success filtering (include all entries regardless of success status)",
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed entries with empty result fields",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of samples to preview (default: 5, use 0 to disable preview)",
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress output messages")

    return parser.parse_args()


def main():
    """Main function with command line argument support"""
    args = parse_arguments()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)

    # Determine success filter setting
    success_filter = not args.no_success_filter

    if not args.quiet:
        print(f"=== Transforming {args.input} ===")
        print(f"Success filter: {'Enabled' if success_filter else 'Disabled'}")
        print(f"Include failed: {'Yes' if args.include_failed else 'No'}")

    # Transform the data
    count = transform_json_to_desired_format(
        args.input,
        args.output,
        success_filter=success_filter,
        include_failed=args.include_failed,
    )

    # Preview samples if requested
    if args.preview > 0 and not args.quiet:
        preview_sample(args.output, args.preview)

    if not args.quiet:
        print(f"\nTransformation completed! {count} entries processed.")


if __name__ == "__main__":
    main()
