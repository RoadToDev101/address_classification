#!/usr/bin/env python3
"""
Script to generate comprehensive test cases based on hierarchical mapping data
"""

import json
import csv
from collections import defaultdict


def load_mapping_data(filename):
    """Load the mapping CSV and organize by hierarchy"""
    mapping = defaultdict(lambda: defaultdict(list))

    with open(filename, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            province = row["provinces"]
            district = row["districts"]
            ward = row["wards"]

            # Remove administrative prefixes for expected results
            province_clean = remove_province_prefix(province)
            district_clean = remove_district_prefix(district)
            ward_clean = remove_ward_prefix(ward)

            mapping[province_clean][district_clean].append(ward_clean)

    return mapping


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


def generate_spelling_errors(text):
    """Generate common spelling error variations"""
    errors = []

    # Common Vietnamese spelling mistakes
    common_mistakes = {
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
    }

    # Apply single character substitutions
    for old, new in common_mistakes.items():
        if old in text.lower():
            errors.append(text.lower().replace(old, new))

    # Character omissions (remove one character)
    if len(text) > 3:
        for i in range(1, len(text) - 1):
            errors.append(text[:i] + text[i + 1 :])

    # Character additions (duplicate a character)
    for i in range(len(text)):
        if text[i].isalpha():
            errors.append(text[:i] + text[i] + text[i:])

    # Character swaps (transpose adjacent characters)
    for i in range(len(text) - 1):
        if text[i].isalpha() and text[i + 1].isalpha():
            chars = list(text)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            errors.append("".join(chars))

    # Return unique errors (max 3 to avoid too many)
    return list(set(errors))[:3]


def generate_abbreviation_errors(text):
    """Generate abbreviation and truncation errors"""
    errors = []

    # Truncate to first few characters
    if len(text) > 4:
        errors.append(text[:3])  # First 3 chars
        errors.append(text[:4])  # First 4 chars

    # Remove last syllable
    words = text.split()
    if len(words) > 1:
        errors.append(words[0])  # Just first word
        errors.append(" ".join(words[:-1]))  # All but last word

    # Add common abbreviation patterns
    if "Quận" in text:
        errors.append(text.replace("Quận", "Q"))
        errors.append(text.replace("Quận", "Qu"))

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


def generate_test_cases(mapping_data):
    """Generate comprehensive test cases from mapping data"""
    test_cases = []

    # Select some key locations for comprehensive testing
    key_locations = [
        ("Hà Nội", "Ba Đình", "Phúc Xá"),
        ("Hà Nội", "Hoàn Kiếm", "Hàng Mã"),
        ("Hồ Chí Minh", "1", "Bến Nghé"),
        ("Hồ Chí Minh", "1", "Bến Thành"),
        ("Nghệ An", "Đô Lương", "Thịnh Sơn"),
        ("Đà Nẵng", "Hải Châu", None),
        ("Cần Thơ", "Ninh Kiều", None),
    ]

    # Add more locations from the data
    provinces_sample = [
        "Hà Nội",
        "Hồ Chí Minh",
        "Nghệ An",
        "Đà Nẵng",
        "Hải Phòng",
        "Quảng Ninh",
        "Thái Nguyên",
        "Bình Định",
        "Phú Yên",
        "Cần Thơ",
        "Sơn La",
    ]

    for province in provinces_sample:
        if province in mapping_data:
            for district in list(mapping_data[province].keys())[
                :2
            ]:  # Take first 2 districts
                wards = mapping_data[province][district]
                ward = wards[0] if wards else None
                if (province, district, ward) not in key_locations:
                    key_locations.append((province, district, ward))

    # Generate variations for each location
    for i, (province, district, ward) in enumerate(key_locations):
        group = chr(65 + (i % 10))  # A, B, C, ...

        # Complete address variations
        if ward:
            # 1. Full formal address
            test_cases.append(
                {
                    "input": f"Phường {ward}, Quận {district}, Thành phố {province}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "full formal address",
                }
            )

            # 2. No accents
            test_cases.append(
                {
                    "input": f"{remove_accents(ward)}, {remove_accents(district)}, {remove_accents(province)}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "no accents",
                }
            )

            # 3. Abbreviated
            test_cases.append(
                {
                    "input": f"P. {ward}, Q. {district}, {province}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "abbreviated format",
                }
            )

            # 4. Reversed order
            test_cases.append(
                {
                    "input": f"{province} - {district} - {ward}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "reversed order with dashes",
                }
            )

            # 5. Incomplete - missing province
            test_cases.append(
                {
                    "input": f"{ward}, {district}",
                    "expected_province": None,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "missing province",
                }
            )

        # District + Province variations
        # 6. District and province only
        test_cases.append(
            {
                "input": f"{district}, {province}",
                "expected_province": province,
                "expected_district": district,
                "expected_ward": None,
                "group": group,
                "notes": "district and province only",
            }
        )

        # 7. No accents district + province
        test_cases.append(
            {
                "input": f"{remove_accents(district)} {remove_accents(province)}",
                "expected_province": province,
                "expected_district": district,
                "expected_ward": None,
                "group": group,
                "notes": "no accents, no comma",
            }
        )

        # Province only variations
        # 8. Province only
        test_cases.append(
            {
                "input": province,
                "expected_province": province,
                "expected_district": None,
                "expected_ward": None,
                "group": group,
                "notes": "province only",
            }
        )

        # 9. Province with admin prefix
        prefix = (
            "TP"
            if province in ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Cần Thơ", "Hải Phòng"]
            else "Tinh"
        )
        test_cases.append(
            {
                "input": f"{prefix} {province}",
                "expected_province": province,
                "expected_district": None,
                "expected_ward": None,
                "group": group,
                "notes": "province with admin prefix",
            }
        )

        # Error variations if ward exists
        if ward:
            # 10. Ward only
            test_cases.append(
                {
                    "input": ward,
                    "expected_province": None,
                    "expected_district": None,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "ward only",
                }
            )

            # 11. Typo in ward
            ward_typo = (
                ward.replace("ă", "a")
                .replace("â", "a")
                .replace("ê", "e")
                .replace("ô", "o")
            )
            if ward_typo != ward:
                test_cases.append(
                    {
                        "input": f"{ward_typo}, {district}, {province}",
                        "expected_province": province,
                        "expected_district": district,
                        "expected_ward": ward,
                        "group": group,
                        "notes": "ward with diacritic variation",
                    }
                )

            # 12-14. Spelling errors in ward
            ward_errors = generate_spelling_errors(ward)
            for j, error_ward in enumerate(ward_errors[:2]):  # Max 2 errors per ward
                test_cases.append(
                    {
                        "input": f"{error_ward}, {district}, {province}",
                        "expected_province": province,
                        "expected_district": district,
                        "expected_ward": ward,
                        "group": group,
                        "notes": f"ward spelling error {j + 1}",
                    }
                )

            # 15. Spelling error in district
            district_errors = generate_spelling_errors(district)
            if district_errors:
                test_cases.append(
                    {
                        "input": f"{ward}, {district_errors[0]}, {province}",
                        "expected_province": province,
                        "expected_district": district,
                        "expected_ward": ward,
                        "group": group,
                        "notes": "district spelling error",
                    }
                )

            # 16. Multiple spelling errors
            if ward_errors and district_errors:
                test_cases.append(
                    {
                        "input": f"{ward_errors[0]}, {district_errors[0]}, {remove_accents(province)}",
                        "expected_province": province,
                        "expected_district": district,
                        "expected_ward": ward,
                        "group": group,
                        "notes": "multiple spelling errors",
                    }
                )

        # 17-18. Spelling errors in district (without ward)
        district_errors = generate_spelling_errors(district)
        for j, error_district in enumerate(district_errors[:2]):
            test_cases.append(
                {
                    "input": f"{error_district}, {province}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": None,
                    "group": group,
                    "notes": f"district spelling error {j + 1} (no ward)",
                }
            )

        # 19. Province spelling error
        province_errors = generate_spelling_errors(province)
        if province_errors:
            test_cases.append(
                {
                    "input": province_errors[0],
                    "expected_province": province,
                    "expected_district": None,
                    "expected_ward": None,
                    "group": group,
                    "notes": "province spelling error",
                }
            )

        # 20. Abbreviation errors
        district_abbrevs = generate_abbreviation_errors(district)
        if district_abbrevs:
            test_cases.append(
                {
                    "input": f"{district_abbrevs[0]}, {province}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": None,
                    "group": group,
                    "notes": "district abbreviation",
                }
            )

        # 21. Mixed case errors
        if ward:
            test_cases.append(
                {
                    "input": f"{ward.upper()}, {district.lower()}, {province.title()}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "mixed case",
                }
            )

        # 22. Extra spaces and punctuation
        if ward:
            test_cases.append(
                {
                    "input": f"  {ward}  ,  {district}  ,  {province}  ",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "extra spaces",
                }
            )

        # 23. Different delimiters
        if ward:
            test_cases.append(
                {
                    "input": f"{ward}; {district}; {province}",
                    "expected_province": province,
                    "expected_district": district,
                    "expected_ward": ward,
                    "group": group,
                    "notes": "semicolon delimiters",
                }
            )

        # 24. No delimiters (space only)
        test_cases.append(
            {
                "input": f"{district} {province}",
                "expected_province": province,
                "expected_district": district,
                "expected_ward": None,
                "group": group,
                "notes": "space delimiter only",
            }
        )

    return test_cases


def main():
    print("Loading mapping data...")
    mapping_data = load_mapping_data("data/mapping.csv")

    print(f"Loaded {len(mapping_data)} provinces")

    print("Generating test cases...")
    test_cases = generate_test_cases(mapping_data)

    print(f"Generated {len(test_cases)} test cases")

    # Write to test.json
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print("Test cases saved to test.json")

    # Show some statistics
    groups = {}
    for tc in test_cases:
        group = tc["group"]
        groups[group] = groups.get(group, 0) + 1

    print("\nTest cases by group:")
    for group, count in sorted(groups.items()):
        print(f"  Group {group}: {count} cases")


if __name__ == "__main__":
    main()
