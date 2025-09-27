#!/usr/bin/env python3
"""
Script to generate comprehensive test cases based on full address data from test.json
"""

import json
import argparse
import os
import sys
from collections import defaultdict

# Set UTF-8 encoding for console output
sys.stdout.reconfigure(encoding="utf-8")


def load_address_data(filename):
    """Load the address data from JSON file (supports both original and transformed formats)"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group addresses by their components for better test case generation
    grouped_data = defaultdict(list)

    for entry in data:
        # Handle both original format (address, province, district, ward) and transformed format (text, result)
        if "text" in entry and "result" in entry:
            # Transformed format
            text = entry["text"]
            result = entry["result"]
            province = result.get("province", "")
            district = result.get("district", "")
            ward = result.get("ward", "")
        elif "address" in entry:
            # Original format
            text = entry["address"]
            province = entry.get("province", "")
            district = entry.get("district", "")
            ward = entry.get("ward", "")
            result = {"province": province, "district": district, "ward": ward}
        else:
            # Skip entries that don't match either format
            continue

        # Create a key for grouping similar addresses
        key = (province, district, ward)
        grouped_data[key].append({"text": text, "result": result})

    return grouped_data


def remove_province_prefix(name):
    for prefix in ["Thành phố ", "Tỉnh "]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def remove_district_prefix(name):
    for prefix in ["Huyện ", "Quận ", "Thị xã ", "Thành phố "]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def remove_ward_prefix(name):
    for prefix in ["Phường ", "Xã ", "Thị trấn "]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def generate_vietnamese_spelling_errors(text):
    """Generate realistic Vietnamese spelling error variations"""
    if not text or text is None:
        return []

    errors = []

    # Common Vietnamese character confusions
    common_mistakes = {
        # Diacritic variations
        "ă": "a",
        "â": "a",
        "à": "a",
        "á": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ê": "e",
        "è": "e",
        "é": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ô": "o",
        "ơ": "o",
        "ò": "o",
        "ó": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ư": "u",
        "ù": "u",
        "ú": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ì": "i",
        "í": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ỳ": "y",
        "ý": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
        # Common consonant confusions
        "nh": "n",  # Thinh -> Thin
        "ng": "n",  # Dong -> Don
        "ph": "f",  # Pho -> Fo
        "th": "t",  # Thanh -> Tan
        "tr": "ch",  # Tran -> Chan
        "ch": "c",  # Chau -> Cau
        "gi": "z",  # Giang -> Ziang
        "qu": "kw",  # Quan -> Kwan
        "d": "z",  # Do -> Zo
        "x": "s",  # Xa -> Sa
        "v": "b",  # Van -> Ban
        "r": "l",  # Rong -> Long
    }

    # Apply single character substitutions
    for old, new in common_mistakes.items():
        if old in text.lower():
            errors.append(text.lower().replace(old, new))

    # Character omissions (remove one character) - more realistic
    if len(text) > 4:
        # Only remove characters from the middle, not start/end
        for i in range(1, len(text) - 1):
            if text[i].isalpha():
                errors.append(text[:i] + text[i + 1 :])

    # Character additions (duplicate a character) - more realistic
    for i in range(len(text)):
        if text[i].isalpha() and i > 0 and i < len(text) - 1:
            errors.append(text[:i] + text[i] + text[i:])

    # Character swaps (transpose adjacent characters) - more realistic
    for i in range(len(text) - 1):
        if text[i].isalpha() and text[i + 1].isalpha():
            chars = list(text)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            errors.append("".join(chars))

    # Return unique errors (max 2 to avoid too many)
    return list(set(errors))[:2]


def generate_vietnamese_abbreviation_errors(text):
    """Generate realistic Vietnamese abbreviation and truncation errors"""
    if not text or text is None:
        return []

    errors = []

    # Common Vietnamese administrative unit abbreviations
    abbreviations = {
        "Phường": ["P.", "P", "Ph"],
        "Quận": ["Q.", "Q", "Qu"],
        "Huyện": ["H.", "H", "Hy"],
        "Thị xã": ["TX.", "TX", "Tx"],
        "Thành phố": ["TP.", "TP", "Tp"],
        "Tỉnh": ["T.", "T", "Tinh"],
        "Xã": ["X.", "X"],
        "Thị trấn": ["TT.", "TT", "Tt"],
    }

    # Apply abbreviations
    for full_form, abbrevs in abbreviations.items():
        if full_form in text:
            for abbrev in abbrevs:
                errors.append(text.replace(full_form, abbrev))

    # Truncate administrative units (common mistake)
    if "Phường" in text:
        errors.append(text.replace("Phường", "Phuong"))
    if "Quận" in text:
        errors.append(text.replace("Quận", "Quan"))
    if "Thành phố" in text:
        errors.append(text.replace("Thành phố", "Thanh pho"))

    # Remove spaces between words (common typing error)
    if " " in text:
        errors.append(text.replace(" ", ""))

    # Truncate long names (keep first 2-3 words)
    words = text.split()
    if len(words) > 2:
        errors.append(" ".join(words[:2]))  # First 2 words
    if len(words) > 3:
        errors.append(" ".join(words[:3]))  # First 3 words

    return errors


def remove_accents(text):
    """Simple accent removal for test variations"""
    accent_map = {
        "à": "a",
        "á": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "â": "a",
        "ầ": "a",
        "ấ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "ă": "a",
        "ằ": "a",
        "ắ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "è": "e",
        "é": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ề": "e",
        "ế": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "ì": "i",
        "í": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ò": "o",
        "ó": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ồ": "o",
        "ố": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ờ": "o",
        "ớ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ù": "u",
        "ú": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ừ": "u",
        "ứ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ỳ": "y",
        "ý": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
        "À": "A",
        "Á": "A",
        "Ả": "A",
        "Ã": "A",
        "Ạ": "A",
        "Â": "A",
        "Ầ": "A",
        "Ấ": "A",
        "Ẩ": "A",
        "Ẫ": "A",
        "Ậ": "A",
        "Ă": "A",
        "Ằ": "A",
        "Ắ": "A",
        "Ẳ": "A",
        "Ẵ": "A",
        "Ặ": "A",
        "È": "E",
        "É": "E",
        "Ẻ": "E",
        "Ẽ": "E",
        "Ẹ": "E",
        "Ê": "E",
        "Ề": "E",
        "Ế": "E",
        "Ể": "E",
        "Ễ": "E",
        "Ệ": "E",
        "Ì": "I",
        "Í": "I",
        "Ỉ": "I",
        "Ĩ": "I",
        "Ị": "I",
        "Ò": "O",
        "Ó": "O",
        "Ỏ": "O",
        "Õ": "O",
        "Ọ": "O",
        "Ô": "O",
        "Ồ": "O",
        "Ố": "O",
        "Ổ": "O",
        "Ỗ": "O",
        "Ộ": "O",
        "Ơ": "O",
        "Ờ": "O",
        "Ớ": "O",
        "Ở": "O",
        "Ỡ": "O",
        "Ợ": "O",
        "Ù": "U",
        "Ú": "U",
        "Ủ": "U",
        "Ũ": "U",
        "Ụ": "U",
        "Ư": "U",
        "Ừ": "U",
        "Ứ": "U",
        "Ử": "U",
        "Ữ": "U",
        "Ự": "U",
        "Ỳ": "Y",
        "Ý": "Y",
        "Ỷ": "Y",
        "Ỹ": "Y",
        "Ỵ": "Y",
        "Đ": "D",
    }
    result = ""
    for char in text:
        result += accent_map.get(char, char)
    return result


def generate_test_cases_from_address_data(
    address_data, max_per_group=10, include_errors=False, test_types=None
):
    """Generate comprehensive test cases from existing address data"""
    if test_types is None:
        test_types = [
            "spelling",
            "abbreviation",
            "accent",
            "case",
            "formatting",
            "incomplete",
        ]

    test_cases = []

    # Get unique address combinations for testing
    unique_combinations = list(address_data.keys())

    # Limit to reasonable number for testing (first 20 combinations)
    test_combinations = unique_combinations[:20]

    for i, (province, district, ward) in enumerate(test_combinations):
        # Get original address from the data
        original_addresses = address_data[(province, district, ward)]
        if not original_addresses:
            continue

        # Use the first address as base
        base_address = original_addresses[0]
        original_text = base_address["text"]
        original_result = base_address["result"]

        # Track test cases for this group
        group_test_cases = []

        # 1. Keep original address as baseline
        if "original" in test_types:
            group_test_cases.append(
                {
                    "text": original_text,
                    "result": {
                        "province": original_result.get("province"),
                        "district": original_result.get("district"),
                        "ward": original_result.get("ward"),
                    },
                    "notes": "original address",
                }
            )

        # 2. Generate spelling error variations
        if "spelling" in test_types and ward:
            ward_errors = generate_vietnamese_spelling_errors(ward)
            for j, error_ward in enumerate(ward_errors):
                # Create address with spelling error in ward
                error_text = original_text.replace(ward, error_ward)
                group_test_cases.append(
                    {
                        "text": error_text,
                        "result": {
                            "province": province,
                            "district": district,
                            "ward": ward,
                        },
                        "notes": f"ward spelling error {j + 1}",
                    }
                )

        # 3. Generate district spelling errors
        if "spelling" in test_types and district:
            district_errors = generate_vietnamese_spelling_errors(district)
            for j, error_district in enumerate(district_errors):
                error_text = original_text.replace(district, error_district)
                group_test_cases.append(
                    {
                        "text": error_text,
                        "result": {
                            "province": province,
                            "district": district,
                            "ward": ward,
                        },
                        "notes": f"district spelling error {j + 1}",
                    }
                )

        # 4. Generate province spelling errors
        if "spelling" in test_types and province:
            province_errors = generate_vietnamese_spelling_errors(province)
            for j, error_province in enumerate(province_errors):
                error_text = original_text.replace(province, error_province)
                group_test_cases.append(
                    {
                        "text": error_text,
                        "result": {
                            "province": province,
                            "district": district,
                            "ward": ward,
                        },
                        "notes": f"province spelling error {j + 1}",
                    }
                )

        # 5. Generate abbreviation variations
        if "abbreviation" in test_types:
            abbrev_errors = generate_vietnamese_abbreviation_errors(original_text)
            for j, abbrev_text in enumerate(
                abbrev_errors[:3]
            ):  # Limit to 3 abbreviation errors
                group_test_cases.append(
                    {
                        "text": abbrev_text,
                        "result": {
                            "province": province,
                            "district": district,
                            "ward": ward,
                        },
                        "notes": f"abbreviation variation {j + 1}",
                    }
                )

        # 6. Generate no-accent variations
        if "accent" in test_types:
            no_accent_text = remove_accents(original_text)
            if no_accent_text != original_text:
                group_test_cases.append(
                    {
                        "text": no_accent_text,
                        "result": {
                            "province": province,
                            "district": district,
                            "ward": ward,
                        },
                        "notes": "no accents",
                    }
                )

        # 7. Generate case variations
        if "case" in test_types:
            case_variations = [
                original_text.upper(),
                original_text.lower(),
                original_text.title(),
            ]
            for j, case_text in enumerate(case_variations):
                if case_text != original_text:
                    group_test_cases.append(
                        {
                            "text": case_text,
                            "result": {
                                "province": province,
                                "district": district,
                                "ward": ward,
                            },
                            "notes": f"case variation {j + 1}",
                        }
                    )

        # 8. Generate spacing/punctuation variations
        if "formatting" in test_types:
            spacing_variations = [
                original_text.replace(",", ";"),  # Semicolon instead of comma
                original_text.replace(",", " - "),  # Dash instead of comma
                original_text.replace(" ", "  "),  # Double spaces
                original_text.replace(" ", ""),  # No spaces
                f"  {original_text}  ",  # Extra spaces at start/end
            ]

            for j, spacing_text in enumerate(spacing_variations):
                if spacing_text != original_text:
                    group_test_cases.append(
                        {
                            "text": spacing_text,
                            "result": {
                                "province": province,
                                "district": district,
                                "ward": ward,
                            },
                            "notes": f"spacing/punctuation variation {j + 1}",
                        }
                    )

        # 9. Generate incomplete address variations
        if "incomplete" in test_types:
            if ward:
                # Missing province
                incomplete_text = original_text.replace(f", {province}", "")
                group_test_cases.append(
                    {
                        "text": incomplete_text,
                        "result": {
                            "province": None,
                            "district": district,
                            "ward": ward,
                        },
                        "notes": "missing province",
                    }
                )

                # Ward only
                group_test_cases.append(
                    {
                        "text": ward,
                        "result": {"province": None, "district": None, "ward": ward},
                        "notes": "ward only",
                    }
                )

            # District and province only
            if ward:
                district_province_text = original_text.replace(f"{ward}, ", "")
                group_test_cases.append(
                    {
                        "text": district_province_text,
                        "result": {
                            "province": province,
                            "district": district,
                            "ward": None,
                        },
                        "notes": "district and province only",
                    }
                )

            # Province only
            group_test_cases.append(
                {
                    "text": province,
                    "result": {"province": province, "district": None, "ward": None},
                    "notes": "province only",
                }
            )

        # Limit test cases per group and add to main list
        limited_group_cases = group_test_cases[:max_per_group]
        test_cases.extend(limited_group_cases)

    return test_cases


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive test cases from address data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python generate_test_cases.py -i test.json -o enhanced_test.json
  
  # Generate test cases with custom parameters
  python generate_test_cases.py -i test.json -o test_cases.json --max-per-group 5 --include-errors
  
  # Generate only specific test types
  python generate_test_cases.py -i test.json -o test_cases.json --test-types spelling case formatting
  
  # Quiet mode for batch processing
  python generate_test_cases.py -i test.json -o test_cases.json --quiet
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        default="test.json",
        help="Input JSON file with address data (default: test.json)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="enhanced_test.json",
        help="Output JSON file for test cases (default: enhanced_test.json)",
    )

    parser.add_argument(
        "--max-per-group",
        type=int,
        default=10,
        help="Maximum test cases per address group (default: 10)",
    )

    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include entries with null/empty results in test generation",
    )

    parser.add_argument(
        "--test-types",
        nargs="+",
        choices=[
            "original",
            "spelling",
            "abbreviation",
            "accent",
            "case",
            "formatting",
            "incomplete",
        ],
        default=[
            "original",
            "spelling",
            "abbreviation",
            "accent",
            "case",
            "formatting",
            "incomplete",
        ],
        help="Types of test cases to generate (default: all types)",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of sample test cases to preview (default: 5, use 0 to disable)",
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output messages except errors"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)

    if not args.quiet:
        print(f"=== Generating Test Cases ===")
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output}")
        print(f"Max per group: {args.max_per_group}")
        print(f"Include errors: {'Yes' if args.include_errors else 'No'}")
        print(f"Test types: {', '.join(args.test_types)}")
        print()

    print("Loading address data from test.json...")
    address_data = load_address_data(args.input)

    print(f"Loaded {len(address_data)} unique address combinations")

    print("Generating comprehensive test cases...")
    test_cases = generate_test_cases_from_address_data(
        address_data,
        max_per_group=args.max_per_group,
        include_errors=args.include_errors,
        test_types=args.test_types,
    )

    print(f"Generated {len(test_cases)} test cases")

    # Write to output file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"Test cases saved to {args.output}")

    # Show detailed statistics
    if not args.quiet:
        notes_count = {}

        for tc in test_cases:
            note = tc["notes"]
            notes_count[note] = notes_count.get(note, 0) + 1

        print("\nTest cases by type:")
        for note, count in sorted(notes_count.items()):
            print(f"  {note}: {count} cases")

        print(f"\nTotal test cases: {len(test_cases)}")
        print("Test cases include:")
        print("  - Baseline original addresses")
        if "spelling" in args.test_types:
            print("  - Vietnamese spelling errors (diacritics, consonants)")
        if "abbreviation" in args.test_types:
            print("  - Realistic abbreviation variations")
        if "accent" in args.test_types:
            print("  - Accent removal variations")
        if "case" in args.test_types:
            print("  - Case variations (upper, lower, title)")
        if "formatting" in args.test_types:
            print("  - Formatting variations (spacing, punctuation)")
        if "incomplete" in args.test_types:
            print("  - Incomplete address variations")

    # Preview samples if requested
    if args.preview > 0 and not args.quiet:
        print(f"\nPreview of first {min(args.preview, len(test_cases))} test cases:")
        for i, tc in enumerate(test_cases[: args.preview]):
            print(f"\nTest case {i + 1}:")
            print(f"  Text: {tc['text']}")
            print(f"  Expected: {tc['result']}")
            print(f"  Type: {tc['notes']}")


if __name__ == "__main__":
    main()
