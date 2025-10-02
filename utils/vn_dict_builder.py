#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese Dictionary Builder

This script creates a dictionary of Vietnamese words from CSV files containing
districts, provinces, and wards data. The words are converted to lowercase
and duplicates are removed while preserving Vietnamese diacritics.
"""

from pathlib import Path
from typing import Set, List


def read_csv_file(file_path: str) -> List[str]:
    """
    Read a CSV file and return a list of lines.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        List[str]: List of lines from the CSV file
    """
    lines = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    lines.append(line)
        print(f"Successfully read {len(lines)} lines from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return lines


def build_vietnamese_dictionary() -> Set[str]:
    """
    Build a Vietnamese dictionary from districts, provinces, and wards CSV files.

    Returns:
        Set[str]: Set of unique Vietnamese words in lowercase
    """
    data_dir = Path("data")
    csv_files = [
        data_dir / "districts.csv",
        data_dir / "provinces.csv",
        data_dir / "wards.csv",
    ]

    all_words = set()

    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        lines = read_csv_file(str(csv_file))

        for line in lines:
            # Extract words from each line
            words = line.split(" ")
            print(f"Extracted {len(words)} words from line: {line}")
            for word in words:
                # Convert to lowercase and add to set (automatically removes duplicates)
                lowercase_word = word.lower()
                all_words.add(lowercase_word)

    return all_words


def save_dictionary_to_file(
    words: Set[str], output_file: str = "data/vietnamese_dictionary.txt"
):
    """
    Save the Vietnamese dictionary to a text file.

    Args:
        words (Set[str]): Set of Vietnamese words
        output_file (str): Output file path
    """
    try:
        # Sort words alphabetically for better organization
        sorted_words = sorted(words)

        with open(output_file, "w", encoding="utf-8-sig") as file:
            for word in sorted_words:
                file.write(word + "\n")

        print(f"\nDictionary saved to {output_file}")
        print(f"Total unique words: {len(words)}")

    except Exception as e:
        print(f"Error saving dictionary: {e}")


def display_sample_words(words: Set[str], sample_size: int = 20):
    """
    Display a sample of words from the dictionary.

    Args:
        words (Set[str]): Set of Vietnamese words
        sample_size (int): Number of words to display
    """
    sorted_words = sorted(words)
    sample_words = sorted_words[:sample_size]

    print(f"\nSample words (first {sample_size}):")
    for i, word in enumerate(sample_words, 1):
        print(f"{i:2d}. {word}")

    if len(words) > sample_size:
        print(f"... and {len(words) - sample_size} more words")


def main():
    """
    Main function to build and save Vietnamese dictionary.
    """
    print("Vietnamese Dictionary Builder")
    print("=" * 40)

    # Build the dictionary
    vietnamese_words = build_vietnamese_dictionary()

    if vietnamese_words:
        # Display statistics
        print("\nDictionary Statistics:")
        print(f"Total unique words: {len(vietnamese_words)}")

        # Display sample words
        display_sample_words(vietnamese_words)

        # Save to file
        save_dictionary_to_file(vietnamese_words)

        # Also save as index.dic in the data directory (matching existing pattern)
        # save_dictionary_to_file(vietnamese_words, "data/index.dic")

    else:
        print("No words found. Please check the input files.")


if __name__ == "__main__":
    main()
