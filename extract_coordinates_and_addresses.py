#!/usr/bin/env python3
"""
Safe version of coordinate extraction and address lookup that handles Unicode properly.
"""

import json
import time
import requests
import random
import logging
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass
import argparse


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for handling Vietnamese text properly.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("address_extractor")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter with UTF-8 encoding support
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)

    # Set encoding to UTF-8 for console output
    if hasattr(console_handler.stream, "reconfigure"):
        console_handler.stream.reconfigure(encoding="utf-8")

    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@dataclass
class AddressResult:
    """Data class to store address lookup results"""

    lat: float
    lon: float
    address: Optional[str]
    success: bool
    error: Optional[str] = None
    province: Optional[str] = None  # Administrative area level 1 (province)
    district: Optional[str] = None  # Administrative area level 2 (district)
    ward: Optional[str] = None  # Administrative area level 3 (ward)


class OpenStreetMapAddressLookup:
    """Class to handle OpenStreetMap address lookups with rate limiting and error handling"""

    def __init__(self, delay_between_requests: float = 1.1, max_retries: int = 3):
        self.delay = delay_between_requests
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "AddressClassification/1.0 (Educational Project)"}
        )

    def get_address_from_coordinates(self, lat: float, lon: float) -> AddressResult:
        """
        Get address from coordinates using OpenStreetMap Nominatim API.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            AddressResult object with address information
        """
        url = "https://nominatim.openstreetmap.org/reverse"

        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": 1,
            "accept-language": "vi,en",  # Prefer Vietnamese, fallback to English
            "zoom": 18,  # High detail level
        }

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)

                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()

                data = response.json()

                if data and "display_name" in data:
                    return AddressResult(
                        lat=lat, lon=lon, address=data["display_name"], success=True
                    )
                else:
                    return AddressResult(
                        lat=lat,
                        lon=lon,
                        address=None,
                        success=False,
                        error="No address found in response",
                    )

            except requests.exceptions.RequestException as e:
                error_msg = f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                if attempt == self.max_retries - 1:
                    return AddressResult(
                        lat=lat, lon=lon, address=None, success=False, error=error_msg
                    )

        return AddressResult(
            lat=lat, lon=lon, address=None, success=False, error="Max retries exceeded"
        )


class TrackAsiaAddressLookup:
    """Class to handle TrackAsia address lookups with old and new administrative boundaries"""

    def __init__(
        self, api_key: str, delay_between_requests: float = 0.0, max_retries: int = 3
    ):
        self.api_key = api_key
        self.delay = (
            delay_between_requests  # No delay needed for TrackAsia with API key
        )
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "AddressClassification/1.0 (Educational Project)"}
        )

    def get_address_from_coordinates(self, lat: float, lon: float) -> AddressResult:
        """
        Get address from coordinates using TrackAsia API with old administrative boundaries.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            AddressResult object with old address information
        """
        url = "https://maps.track-asia.com/api/v2/geocode/json"

        params = {
            "latlng": f"{lat},{lon}",
            "key": self.api_key,
            "new_admin": False,  # Get old administrative boundaries
            "radius": 500,  # Search radius in meters
        }

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)

                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                if data and data.get("status") == "OK" and data.get("results"):
                    result = data["results"][0]

                    # Extract formatted address
                    formatted_address = result.get("formatted_address", "")

                    # Extract administrative components
                    province, district, ward = self._extract_administrative_components(
                        result
                    )

                    return AddressResult(
                        lat=lat,
                        lon=lon,
                        address=formatted_address,
                        success=True,
                        province=province,
                        district=district,
                        ward=ward,
                    )
                else:
                    return AddressResult(
                        lat=lat,
                        lon=lon,
                        address=None,
                        success=False,
                        error=f"No address found. Status: {data.get('status', 'Unknown')}",
                    )

            except requests.exceptions.RequestException as e:
                error_msg = f"TrackAsia API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                if attempt == self.max_retries - 1:
                    return AddressResult(
                        lat=lat, lon=lon, address=None, success=False, error=error_msg
                    )

        return AddressResult(
            lat=lat, lon=lon, address=None, success=False, error="Max retries exceeded"
        )

    def _extract_administrative_components(self, result: dict) -> tuple[str, str, str]:
        """
        Extract administrative components (province, district, ward) from TrackAsia API response.

        Args:
            result: Single result from TrackAsia API response

        Returns:
            Tuple of (province, district, ward)
        """
        province = None
        district = None
        ward = None

        try:
            address_components = result.get("address_components", [])

            for component in address_components:
                types = component.get("types", [])
                long_name = component.get("long_name", "")

                if "administrative_area_level_1" in types:
                    province = long_name
                elif "administrative_area_level_2" in types:
                    district = long_name
                elif "administrative_area_level_3" in types:
                    ward = long_name

        except Exception:
            # Log error but don't fail the entire request
            pass

        return province, district, ward


def extract_all_coordinates_from_geojson(
    geojson_path: str, logger: logging.Logger
) -> List[Tuple[float, float]]:
    """
    Extract ALL coordinates from GeoJSON features (process every record).

    Args:
        geojson_path: Path to the GeoJSON file
        logger: Logger instance for output

    Returns:
        List of (latitude, longitude) tuples
    """
    coordinates = []

    logger.info(f"Extracting ALL coordinates from GeoJSON file: {geojson_path}")
    logger.info("Processing every record in the file...")

    try:
        with open(geojson_path, "r", encoding="utf-8") as file:
            geojson_data = json.load(file)

            if geojson_data.get("type") != "FeatureCollection":
                logger.error("File is not a valid GeoJSON FeatureCollection")
                return coordinates

            features = geojson_data.get("features", [])
            total_features = len(features)
            logger.info(f"Total features to process: {total_features:,}")

            for i, feature in enumerate(features):
                if i % 10000 == 0:  # Progress indicator every 10k features
                    logger.info(
                        f"Processing feature {i:,}/{total_features:,} - Found {len(coordinates):,} coordinates so far"
                    )

                if feature.get("type") == "Feature" and "geometry" in feature:
                    geometry = feature["geometry"]

                    if geometry.get("type") == "Polygon" and "coordinates" in geometry:
                        # Get the centroid of the polygon (first coordinate of first ring)
                        coords = geometry["coordinates"][0][0]
                        if len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            coordinates.append((lat, lon))

                    elif geometry.get("type") == "Point" and "coordinates" in geometry:
                        coords = geometry["coordinates"]
                        if len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            coordinates.append((lat, lon))

    except Exception as e:
        logger.error(f"Error reading GeoJSON file: {e}")
        return coordinates

    logger.info(f"Extracted {len(coordinates):,} coordinates from all features")
    return coordinates


def extract_coordinates_from_geojson(
    geojson_path: str,
    max_features: int = 100,
    sample_strategy: str = "random",
    logger: logging.Logger = None,
) -> List[Tuple[float, float]]:
    """
    Extract coordinates from GeoJSON features with different sampling strategies.

    Args:
        geojson_path: Path to the GeoJSON file
        max_features: Maximum number of features to process
        sample_strategy: Strategy for sampling features ("random", "first", "last")
        logger: Logger instance for output

    Returns:
        List of (latitude, longitude) tuples
    """
    coordinates = []

    if logger:
        logger.info(f"Extracting coordinates from GeoJSON file: {geojson_path}")
        logger.info(f"Sampling strategy: {sample_strategy}")
        logger.info(f"Processing up to {max_features:,} features...")

    try:
        with open(geojson_path, "r", encoding="utf-8") as file:
            geojson_data = json.load(file)

            if geojson_data.get("type") != "FeatureCollection":
                if logger:
                    logger.error("File is not a valid GeoJSON FeatureCollection")
                return coordinates

            features = geojson_data.get("features", [])
            total_features = len(features)
            if logger:
                logger.info(f"Total features available: {total_features:,}")

            # Select features based on strategy
            if sample_strategy == "random":
                selected_features = random.sample(
                    features, min(max_features, total_features)
                )
            elif sample_strategy == "first":
                selected_features = features[:max_features]
            elif sample_strategy == "last":
                selected_features = features[-max_features:]
            else:
                selected_features = features[:max_features]

            if logger:
                logger.info(
                    f"Selected {len(selected_features):,} features for processing"
                )

            for i, feature in enumerate(selected_features):
                if i % 100 == 0 and logger:
                    logger.debug(
                        f"Extracting coordinates from feature {i:,}/{len(selected_features):,}"
                    )

                if feature.get("type") == "Feature" and "geometry" in feature:
                    geometry = feature["geometry"]

                    if geometry.get("type") == "Polygon" and "coordinates" in geometry:
                        # Get the centroid of the polygon (first coordinate of first ring)
                        coords = geometry["coordinates"][0][0]
                        if len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            coordinates.append((lat, lon))

                    elif geometry.get("type") == "Point" and "coordinates" in geometry:
                        coords = geometry["coordinates"]
                        if len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            coordinates.append((lat, lon))

    except Exception as e:
        if logger:
            logger.error(f"Error reading GeoJSON file: {e}")
        return coordinates

    if logger:
        logger.info(f"Extracted {len(coordinates):,} coordinates")
    return coordinates


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


def clean_address(
    address: str, remove_zipcode: bool = False, remove_nation: bool = False
) -> str:
    """
    Clean address by removing zipcode and/or nation if requested.

    Args:
        address: Original address string
        remove_zipcode: Whether to remove zipcode (e.g., "76913")
        remove_nation: Whether to remove nation (e.g., "Việt Nam")

    Returns:
        Cleaned address string
    """
    if not address:
        return address

    # Split by comma to get address components
    parts = [part.strip() for part in address.split(",")]

    if remove_zipcode:
        # Remove parts that are only digits (zipcodes)
        parts = [part for part in parts if not part.isdigit()]

    if remove_nation:
        # Remove common nation names
        nation_indicators = ["Việt Nam", "Vietnam", "VN"]
        parts = [part for part in parts if part not in nation_indicators]

    return ", ".join(parts).strip()


def get_addresses_from_coordinates(
    coordinates: List[Tuple[float, float]],
    output_path: str,
    delay_between_requests: float = 1.1,
    save_progress: bool = True,
    remove_zipcode: bool = False,
    remove_nation: bool = False,
    api_provider: str = "osm",
    trackasia_api_key: str = None,
    remove_prefixes: bool = False,
    no_txt_export: bool = False,
    logger: logging.Logger = None,
) -> None:
    """
    Get addresses for coordinates using specified API provider and save to file.

    Args:
        coordinates: List of (latitude, longitude) tuples
        output_path: Path to save the addresses
        delay_between_requests: Delay between API requests in seconds
        save_progress: Whether to save progress incrementally
        remove_zipcode: Whether to remove zipcode from addresses
        remove_nation: Whether to remove nation from addresses
        api_provider: API provider to use ("osm" or "trackasia")
        trackasia_api_key: API key for TrackAsia (required if api_provider is "trackasia")
        remove_prefixes: Whether to remove administrative prefixes from detailed results
        no_txt_export: Whether to skip exporting addresses to txt file
        logger: Logger instance for output
    """
    # Initialize the appropriate lookup service
    if api_provider == "trackasia":
        if not trackasia_api_key:
            raise ValueError(
                "TrackAsia API key is required when using trackasia provider"
            )
        # No delay needed for TrackAsia with API key
        lookup = TrackAsiaAddressLookup(trackasia_api_key, delay_between_requests=0.1)
    else:
        # OSM requires rate limiting
        lookup = OpenStreetMapAddressLookup(delay_between_requests)
    addresses = []
    results = []
    successful_requests = 0
    failed_requests = 0

    if logger:
        logger.info(f"Getting addresses for {len(coordinates):,} coordinates...")
        logger.info(f"Using {api_provider.upper()} API provider")

        if api_provider == "trackasia":
            logger.info("Mode: Old administrative addresses (pre-merge)")
            logger.info("No rate limiting (API key provided)")
        else:
            logger.info(
                f"Using {delay_between_requests}s delay between requests to respect rate limits"
            )

    for i, (lat, lon) in enumerate(coordinates):
        if logger:
            logger.info(
                f"Processing coordinate {i + 1:,}/{len(coordinates):,} - Lat: {lat:.6f}, Lon: {lon:.6f}"
            )

        result = lookup.get_address_from_coordinates(lat, lon)
        results.append(result)

        if result.success and result.address:
            # Clean the address if requested
            cleaned_address = clean_address(
                result.address, remove_zipcode, remove_nation
            )
            addresses.append(cleaned_address)
            successful_requests += 1

            # Log the results based on API provider
            if logger:
                if (
                    api_provider == "trackasia"
                    and result.province
                    and result.district
                    and result.ward
                ):
                    logger.info(f"  Address: {cleaned_address}")
                    logger.info(
                        f"  Province: {result.province}, District: {result.district}, Ward: {result.ward}"
                    )
                else:
                    logger.info(f"  Found: {cleaned_address}")
        else:
            failed_requests += 1
            if logger:
                logger.warning(f"  Failed: {result.error or 'No address found'}")

        # Save progress every 5 successful lookups
        if save_progress and successful_requests > 0 and successful_requests % 5 == 0:
            unique_addresses = list(
                dict.fromkeys(addresses)
            )  # Remove duplicates while preserving order
            with open(f"{output_path}.progress", "w", encoding="utf-8") as f:
                for addr in unique_addresses:
                    f.write(addr + "\n")
            if logger:
                logger.info(
                    f"  Progress saved: {len(unique_addresses)} addresses so far"
                )

    # Remove duplicates while preserving order
    unique_addresses = list(dict.fromkeys(addresses))

    # Filter results to only include unique addresses
    unique_results = []
    seen_addresses = set()

    for r in results:
        if r.success and r.address:
            cleaned_address = clean_address(r.address, remove_zipcode, remove_nation)
            if cleaned_address not in seen_addresses:
                seen_addresses.add(cleaned_address)
                unique_results.append(r)

    # Save txt file (optional)
    if not no_txt_export:
        if logger:
            logger.info(
                f"Saving {len(unique_addresses):,} unique addresses to {output_path}"
            )

        try:
            with open(output_path, "w", encoding="utf-8", newline="") as output_file:
                # For both OSM and TrackAsia, save addresses in OSM format
                for address in unique_addresses:
                    output_file.write(address + "\n")
        except Exception as e:
            if logger:
                logger.error(f"Error saving addresses to txt file: {e}")

    # Save detailed results (always save, with unique addresses only)
    results_file = output_path.replace(".txt", "_detailed_results.json")
    if logger:
        logger.info(
            f"Saving {len(unique_results):,} unique detailed results to {results_file}"
        )

    try:
        with open(results_file, "w", encoding="utf-8") as f:
            detailed_results = []
            for r in unique_results:
                result_data = {
                    "lat": r.lat,
                    "lon": r.lon,
                    "address": r.address,
                    "success": r.success,
                    "error": r.error,
                }

                # Add administrative components with optional prefix removal
                if remove_prefixes:
                    result_data["province"] = (
                        remove_prefix(r.province, "province")
                        if r.province
                        else r.province
                    )
                    result_data["district"] = (
                        remove_prefix(r.district, "district")
                        if r.district
                        else r.district
                    )
                    result_data["ward"] = (
                        remove_prefix(r.ward, "ward") if r.ward else r.ward
                    )
                else:
                    result_data["province"] = r.province
                    result_data["district"] = r.district
                    result_data["ward"] = r.ward

                detailed_results.append(result_data)

            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        if logger:
            if not no_txt_export:
                logger.info(f"Successfully saved addresses to {output_path}")
            logger.info(f"Detailed results saved to {results_file}")
            logger.info("Statistics:")
            logger.info(f"   - Total coordinates processed: {len(coordinates):,}")
            logger.info(f"   - Successful API requests: {successful_requests:,}")
            logger.info(f"   - Failed API requests: {failed_requests:,}")
            logger.info(f"   - Unique addresses found: {len(unique_results):,}")
            logger.info(
                f"   - Success rate: {successful_requests / len(coordinates) * 100:.1f}%"
            )

    except Exception as e:
        if logger:
            logger.error(f"Error saving addresses to file: {e}")


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description="Extract coordinates from GeoJSON and get addresses from OpenStreetMap"
    )
    parser.add_argument(
        "--geojson",
        default="hotosm_vnm_buildings_polygons_geojson.geojson",
        help="Path to GeoJSON file",
    )
    parser.add_argument(
        "--output", default="osm_addresses.txt", help="Output file for addresses"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50,
        help="Maximum number of features to process",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["random", "first", "last"],
        default="random",
        help="Strategy for sampling features",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.1,
        help="Delay between API requests in seconds (OSM only, TrackAsia has no delay)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Don't save progress incrementally"
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Process ALL records in the GeoJSON file (not just a sample)",
    )
    parser.add_argument(
        "--remove-zipcode",
        action="store_true",
        help="Remove zipcode from addresses (e.g., '76913')",
    )
    parser.add_argument(
        "--remove-nation",
        action="store_true",
        help="Remove nation from addresses (e.g., 'Vietnam')",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path for detailed logging",
    )
    parser.add_argument(
        "--api-provider",
        choices=["osm", "trackasia"],
        default="osm",
        help="API provider to use for address lookup (osm or trackasia)",
    )
    parser.add_argument(
        "--trackasia-api-key",
        help="TrackAsia API key (required when using trackasia provider)",
    )
    parser.add_argument(
        "--remove-prefixes",
        action="store_true",
        help="Remove administrative prefixes from province, district, ward in detailed results",
    )
    parser.add_argument(
        "--no-txt-export",
        action="store_true",
        help="Skip exporting addresses to txt file (only save detailed results JSON)",
    )

    args = parser.parse_args()

    # Validate API provider and key
    if args.api_provider == "trackasia" and not args.trackasia_api_key:
        print("Error: TrackAsia API key is required when using trackasia provider")
        print("Use --trackasia-api-key to provide your API key")
        return

    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)

    logger.info("Address Lookup Tool")
    logger.info("=" * 50)
    if args.api_provider == "trackasia":
        logger.info(
            "This tool extracts coordinates from GeoJSON and looks up addresses using TrackAsia API"
        )
        logger.info("Mode: Old administrative addresses (pre-merge)")
        logger.info("No rate limiting - API key provided for faster processing")
    else:
        logger.info(
            "This tool extracts coordinates from GeoJSON and looks up addresses using OpenStreetMap API"
        )
        logger.info(
            "Note: This may take a while due to rate limiting (1 request per second)"
        )
    logger.info("")

    # Extract coordinates
    if args.all_records:
        logger.warning("Processing ALL records - this may take a very long time!")
        logger.warning("The GeoJSON file contains over 1 million features.")
        logger.warning(
            "This could take several hours to complete due to API rate limits."
        )
        response = input("Do you want to continue? (y/N): ").strip().lower()
        if response != "y":
            logger.info("Operation cancelled.")
            return

        coordinates = extract_all_coordinates_from_geojson(args.geojson, logger)
    else:
        coordinates = extract_coordinates_from_geojson(
            args.geojson, args.max_features, args.sample_strategy, logger
        )

    if not coordinates:
        logger.error("No coordinates found. Exiting.")
        return

    logger.info(f"Found {len(coordinates):,} coordinates to process")

    # Show cleaning options
    if (
        args.remove_zipcode
        or args.remove_nation
        or args.remove_prefixes
        or args.no_txt_export
    ):
        logger.info("Processing options:")
        if args.remove_zipcode:
            logger.info("  - Will remove zipcodes (e.g., '76913')")
        if args.remove_nation:
            logger.info("  - Will remove nation (e.g., 'Vietnam')")
        if args.remove_prefixes:
            logger.info("  - Will remove administrative prefixes from detailed results")
        if args.no_txt_export:
            logger.info(
                "  - Will skip txt file export (only save detailed results JSON)"
            )

    # Get addresses
    get_addresses_from_coordinates(
        coordinates,
        args.output,
        args.delay,
        not args.no_progress,
        args.remove_zipcode,
        args.remove_nation,
        args.api_provider,
        args.trackasia_api_key,
        args.remove_prefixes,
        args.no_txt_export,
        logger,
    )


if __name__ == "__main__":
    main()
