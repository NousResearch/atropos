"""
Enhanced error introduction system for editing tasks.

This module provides error introduction capabilities for creating
challenging editing tasks with Pydantic models. It supports various error types,
constraint violations, and complex data structures.
"""

import copy
import random
import re
import string
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo


class ErrorType(Enum):
    """Types of errors that can be introduced."""

    TYPE_ERROR = "type_error"
    CONSTRAINT_ERROR = "constraint_error"
    FORMAT_ERROR = "format_error"
    ENUM_ERROR = "enum_error"
    REQUIRED_FIELD_MISSING = "required_field_missing"
    EXTRA_FIELD = "extra_field"
    NESTED_ERROR = "nested_error"
    LIST_ERROR = "list_error"
    VALIDATOR_ERROR = "validator_error"


class ErrorIntroductionConfig:
    """Configuration for error introduction."""

    def __init__(
        self,
        error_types: List[ErrorType] = None,
        max_errors: int = 1,
        probability: float = 1.0,
        seed: Optional[int] = None,
    ):
        if error_types is None:
            self.error_types = [
                ErrorType.TYPE_ERROR,
                ErrorType.CONSTRAINT_ERROR,
                ErrorType.FORMAT_ERROR,
                ErrorType.ENUM_ERROR,
                ErrorType.REQUIRED_FIELD_MISSING,
            ]
        else:
            self.error_types = error_types
        self.max_errors = max_errors
        self.probability = probability
        self.seed = seed

    @classmethod
    def from_env_config(
        cls,
        error_types_enabled: List[str],
        max_errors_per_item: int = 1,
        error_introduction_probability: float = 1.0,
        error_introduction_seed: Optional[int] = None,
    ):
        """Create config from environment config parameters."""
        error_type_mapping = {
            "type_error": ErrorType.TYPE_ERROR,
            "constraint_error": ErrorType.CONSTRAINT_ERROR,
            "format_error": ErrorType.FORMAT_ERROR,
            "enum_error": ErrorType.ENUM_ERROR,
            "required_field_missing": ErrorType.REQUIRED_FIELD_MISSING,
            "extra_field": ErrorType.EXTRA_FIELD,
            "nested_error": ErrorType.NESTED_ERROR,
            "list_error": ErrorType.LIST_ERROR,
            "validator_error": ErrorType.VALIDATOR_ERROR,
        }

        error_types = [
            error_type_mapping[name]
            for name in error_types_enabled
            if name in error_type_mapping
        ]

        return cls(
            error_types=error_types,
            max_errors=max_errors_per_item,
            probability=error_introduction_probability,
            seed=error_introduction_seed,
        )


def introduce_error_for_pydantic(
    data: Any,
    pydantic_model: Type[BaseModel],
    config: Optional[ErrorIntroductionConfig] = None,
    seed: Optional[int] = None,  # Backward compatibility
) -> Optional[Any]:
    """
    Introduce sophisticated errors into valid data for Pydantic model editing tasks.

    Args:
        data: Valid data structure (should be compatible with pydantic_model)
        pydantic_model: Pydantic model class to violate
        config: Error introduction configuration
        seed: Optional seed for backward compatibility

    Returns:
        Modified data with error(s), or None if no error could be introduced
    """
    # Handle backward compatibility
    if config is None:
        config = ErrorIntroductionConfig(seed=seed)
    elif seed is not None:
        config.seed = seed

    if config.seed is not None:
        random.seed(config.seed)

    # Check probability
    if random.random() > config.probability:
        return None

    # Validate input
    if not isinstance(data, dict):
        return None

    # Check if any error types are configured
    if not config.error_types:
        return None

    # Analyze the Pydantic model
    try:
        model_analysis = _analyze_pydantic_model(pydantic_model)
    except Exception:
        return None

    erroneous_data = copy.deepcopy(data)
    errors_introduced = 0
    max_attempts = 50  # Prevent infinite loops

    # Shuffle error types for variety
    available_error_types = config.error_types.copy()
    random.shuffle(available_error_types)

    for attempt in range(max_attempts):
        if errors_introduced >= config.max_errors:
            break

        if not available_error_types:
            break

        error_type = available_error_types[attempt % len(available_error_types)]

        try:
            original_data = copy.deepcopy(erroneous_data)
            success = False

            if error_type == ErrorType.TYPE_ERROR:
                success = _introduce_type_error(erroneous_data, model_analysis)
            elif error_type == ErrorType.CONSTRAINT_ERROR:
                success = _introduce_constraint_error(erroneous_data, model_analysis)
            elif error_type == ErrorType.FORMAT_ERROR:
                success = _introduce_format_error(erroneous_data, model_analysis)
            elif error_type == ErrorType.ENUM_ERROR:
                success = _introduce_enum_error(erroneous_data, model_analysis)
            elif error_type == ErrorType.REQUIRED_FIELD_MISSING:
                success = _introduce_required_field_error(
                    erroneous_data, model_analysis
                )
            elif error_type == ErrorType.EXTRA_FIELD:
                success = _introduce_extra_field_error(erroneous_data, model_analysis)
            elif error_type == ErrorType.NESTED_ERROR:
                success = _introduce_nested_error(
                    erroneous_data, model_analysis, pydantic_model
                )
            elif error_type == ErrorType.LIST_ERROR:
                success = _introduce_list_error(erroneous_data, model_analysis)
            elif error_type == ErrorType.VALIDATOR_ERROR:
                success = _introduce_validator_error(erroneous_data, model_analysis)

            if success:
                # Verify the error was actually introduced
                try:
                    pydantic_model(**erroneous_data)
                    # If validation passes, error wasn't introduced properly
                    erroneous_data = original_data
                except ValidationError:
                    # Error successfully introduced
                    errors_introduced += 1

        except Exception:
            # Restore data if error introduction failed
            erroneous_data = original_data
            continue

    # Return result if any errors were introduced
    if errors_introduced > 0 and erroneous_data != data:
        return erroneous_data

    return None


def _analyze_pydantic_model(model: Type[BaseModel]) -> Dict[str, Any]:
    """Analyze a Pydantic model to understand its structure and constraints."""
    analysis = {"fields": {}, "validators": {}, "nested_models": {}, "enums": {}}

    model_fields = model.model_fields

    for field_name, field_info in model_fields.items():
        field_analysis = _analyze_field(field_name, field_info)
        analysis["fields"][field_name] = field_analysis

        # Track nested models
        if field_analysis.get("nested_model"):
            analysis["nested_models"][field_name] = field_analysis["nested_model"]

        # Track enums
        if field_analysis.get("enum_values"):
            analysis["enums"][field_name] = field_analysis["enum_values"]

    # Analyze custom validators (Pydantic v2)
    if hasattr(model, "__pydantic_validators__"):
        analysis["validators"] = model.__pydantic_validators__
    elif hasattr(model, "__pydantic_decorators__"):
        analysis["validators"] = model.__pydantic_decorators__

    # Mark fields that have validators
    for field_name, field_info in model_fields.items():
        field_analysis = analysis["fields"].get(field_name, {})
        # Check if this field has any validators
        if hasattr(model, "__pydantic_decorators__"):
            decorators = model.__pydantic_decorators__
            if hasattr(decorators, "field_validators"):
                # Check if any field validator applies to this field
                for (
                    validator_name,
                    validator_info,
                ) in decorators.field_validators.items():
                    if hasattr(validator_info, "info") and hasattr(
                        validator_info.info, "fields"
                    ):
                        if field_name in validator_info.info.fields:
                            field_analysis["has_validators"] = True
                            break
        analysis["fields"][field_name] = field_analysis

    return analysis


def _analyze_field(field_name: str, field_info: FieldInfo) -> Dict[str, Any]:
    """Analyze a single Pydantic field."""
    field_analysis = {
        "name": field_name,
        "type": "unknown",
        "required": field_info.is_required(),
        "default": field_info.default if field_info.default is not None else None,
        "constraints": {},
        "has_validators": False,
        "nested_model": None,
        "enum_values": None,
        "is_list": False,
        "is_dict": False,
        "is_optional": False,
    }

    # Analyze the annotation
    annotation = field_info.annotation

    # Handle Union types (including Optional)
    if get_origin(annotation) is Union:
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            # This is Optional[T]
            field_analysis["is_optional"] = True
            # Get the non-None type
            annotation = next(arg for arg in args if arg is not type(None))

    # Handle List types
    if get_origin(annotation) is list:
        field_analysis["is_list"] = True
        field_analysis["type"] = "list"
        list_args = get_args(annotation)
        if list_args:
            field_analysis["list_item_type"] = list_args[0]

    # Handle Dict types
    elif get_origin(annotation) is dict:
        field_analysis["is_dict"] = True
        field_analysis["type"] = "dict"

    # Handle basic types
    elif annotation in (str, int, float, bool):
        field_analysis["type"] = annotation.__name__

    # Handle special Pydantic types
    elif hasattr(annotation, "__name__"):
        if annotation.__name__ in ["EmailStr", "HttpUrl", "UUID"]:
            field_analysis["type"] = annotation.__name__.lower()
        elif annotation.__name__ == "datetime":
            field_analysis["type"] = "datetime"
        elif annotation.__name__ == "date":
            field_analysis["type"] = "date"
        elif annotation.__name__ == "Decimal":
            field_analysis["type"] = "decimal"

    # Handle Enums
    if hasattr(annotation, "__bases__") and Enum in annotation.__bases__:
        field_analysis["type"] = "enum"
        field_analysis["enum_values"] = [e.value for e in annotation]

    # Handle nested Pydantic models
    try:
        if hasattr(annotation, "__bases__") and any(
            issubclass(base, BaseModel)
            for base in annotation.__bases__
            if base != BaseModel
        ):
            field_analysis["type"] = "nested_model"
            field_analysis["nested_model"] = annotation
        elif hasattr(annotation, "__mro__") and BaseModel in annotation.__mro__:
            field_analysis["type"] = "nested_model"
            field_analysis["nested_model"] = annotation
    except (TypeError, AttributeError):
        # If we can't determine if it's a BaseModel, skip
        pass

    # Extract constraints from Field() - Pydantic v2 approach
    if hasattr(field_info, "constraints") and field_info.constraints:
        for constraint_name, constraint_value in field_info.constraints.items():
            if constraint_value is not None:
                field_analysis["constraints"][constraint_name] = constraint_value

    # Extract common Field constraints directly from field_info
    constraint_attrs = ["min_length", "max_length", "ge", "le", "gt", "lt", "pattern"]
    for attr in constraint_attrs:
        if hasattr(field_info, attr):
            value = getattr(field_info, attr)
            if value is not None:
                field_analysis["constraints"][attr] = value
        # Also check if it's in the metadata
        elif hasattr(field_info, "metadata"):
            for metadata_item in field_info.metadata:
                if hasattr(metadata_item, attr):
                    value = getattr(metadata_item, attr)
                    if value is not None:
                        field_analysis["constraints"][attr] = value
                        break

    return field_analysis


def _introduce_type_error(data: Dict[str, Any], model_analysis: Dict[str, Any]) -> bool:
    """Introduce type errors by changing value types."""
    fields = model_analysis["fields"]

    for field_name, field_info in fields.items():
        if field_name not in data:
            continue

        current_value = data[field_name]
        field_type = field_info["type"]

        # Skip if optional and current value is None
        if field_info["is_optional"] and current_value is None:
            continue

        # Introduce type-specific errors
        if field_type == "str" and not isinstance(current_value, str):
            continue
        elif field_type == "int" and not isinstance(current_value, int):
            continue
        elif field_type == "bool" and not isinstance(current_value, bool):
            continue

        # Create type mismatches
        if field_type == "str" and isinstance(current_value, str):
            data[field_name] = random.choice([42, True, [], {}])
            return True
        elif field_type == "int" and isinstance(current_value, int):
            data[field_name] = random.choice(["not_a_number", True, []])
            return True
        elif field_type == "bool" and isinstance(current_value, bool):
            data[field_name] = random.choice(["not_a_boolean", 42])
            return True
        elif field_type == "list" and isinstance(current_value, list):
            data[field_name] = "not_a_list"
            return True
        elif field_type == "dict" and isinstance(current_value, dict):
            data[field_name] = "not_a_dict"
            return True

    return False


def _introduce_constraint_error(
    data: Dict[str, Any], model_analysis: Dict[str, Any]
) -> bool:
    """Introduce constraint violation errors."""
    fields = model_analysis["fields"]

    for field_name, field_info in fields.items():
        if field_name not in data:
            continue

        current_value = data[field_name]
        constraints = field_info["constraints"]

        if not constraints:
            continue

        # String length constraints
        if isinstance(current_value, str):
            if (
                "min_length" in constraints
                and len(current_value) >= constraints["min_length"]
            ):
                # Make it too short
                data[field_name] = "x" * (constraints["min_length"] - 1)
                return True
            elif (
                "max_length" in constraints
                and len(current_value) <= constraints["max_length"]
            ):
                # Make it too long
                data[field_name] = "x" * (constraints["max_length"] + 10)
                return True

        # Numeric constraints
        elif isinstance(current_value, (int, float)):
            if "ge" in constraints and current_value >= constraints["ge"]:
                # Make it too small
                data[field_name] = constraints["ge"] - 1
                return True
            elif "gt" in constraints and current_value > constraints["gt"]:
                # Make it too small
                data[field_name] = constraints["gt"]
                return True
            elif "le" in constraints and current_value <= constraints["le"]:
                # Make it too large
                data[field_name] = constraints["le"] + 1
                return True
            elif "lt" in constraints and current_value < constraints["lt"]:
                # Make it too large
                data[field_name] = constraints["lt"]
                return True

        # List constraints
        elif isinstance(current_value, list):
            if (
                "max_items" in constraints
                and len(current_value) <= constraints["max_items"]
            ):
                # Add too many items
                extra_items = ["extra"] * (
                    constraints["max_items"] + 5 - len(current_value)
                )
                data[field_name] = current_value + extra_items
                return True
            elif (
                "min_items" in constraints
                and len(current_value) >= constraints["min_items"]
            ):
                # Remove too many items
                items_to_keep = max(0, constraints["min_items"] - 1)
                data[field_name] = current_value[:items_to_keep]
                return True

    return False


def _mutate_string(s: str, mutation_rate: float = 0.1) -> str:
    """Apply character-level mutations to a string."""
    if not s:
        return s

    mutations = []
    for i, char in enumerate(s):
        if random.random() < mutation_rate:
            mutation_type = random.choice(
                ["swap", "delete", "insert", "duplicate", "case"]
            )

            if mutation_type == "swap" and i < len(s) - 1:
                # Swap with next character
                mutations.append(s[:i] + s[i + 1] + s[i] + s[i + 2 :])
            elif mutation_type == "delete":
                # Delete character
                mutations.append(s[:i] + s[i + 1 :])
            elif mutation_type == "insert":
                # Insert random character
                random_char = random.choice(
                    string.ascii_letters + string.digits + ".-_@"
                )
                mutations.append(s[:i] + random_char + s[i:])
            elif mutation_type == "duplicate":
                # Duplicate character
                mutations.append(s[:i] + char + char + s[i + 1 :])
            elif mutation_type == "case" and char.isalpha():
                # Change case
                mutations.append(s[:i] + char.swapcase() + s[i + 1 :])

    return mutations[0] if mutations else s


def _generate_invalid_email(valid_email: str = "") -> str:
    """Generate invalid email addresses dynamically."""
    strategies = [
        # Character mutations
        lambda: _mutate_string(
            valid_email if valid_email and "@" in valid_email else "user@example.com"
        ),
        # Missing @ symbol
        lambda: (
            valid_email.replace("@", "") if "@" in valid_email else "userexample.com"
        ),
        # Multiple @ symbols
        lambda: (
            valid_email.replace("@", "@@")
            if "@" in valid_email
            else "user@@example.com"
        ),
        # Missing domain
        lambda: valid_email.split("@")[0] + "@" if "@" in valid_email else "user@",
        # Missing local part
        lambda: "@"
        + (valid_email.split("@")[1] if "@" in valid_email else "example.com"),
        # Invalid characters
        lambda: f"{random.choice(['user name', 'user..name', '.user', 'user.'])}@example.com",
        # Invalid domain
        lambda: f"user@{random.choice(['', '.com', 'example.', 'example..com', '123.456.789.0'])}",
        # Spaces in various positions
        lambda: f"{random.choice(['user ', ' user', 'us er'])}@{random.choice(['exam ple.com', 'example.com '])}",
        # Invalid TLD
        lambda: f"user@example.{random.choice(['', 'c', 'toolong', '123', 'com.', '.com'])}",
        # Special cases
        lambda: random.choice(
            [
                "user@[not.an.ip]",
                "user@localhost",
                "user name@example.com",
                "user@example@com",
                "user#example.com",
                "user@example,com",
                "(comment)user@example.com",
                "user@exam ple.com",
                "user@-example.com",
                "user@example-.com",
            ]
        ),
    ]

    return random.choice(strategies)()


def _generate_invalid_url(valid_url: str = "") -> str:
    """Generate invalid URLs dynamically."""
    strategies = [
        # Character mutations
        lambda: _mutate_string(
            valid_url
            if valid_url.startswith(("http://", "https://"))
            else "https://example.com"
        ),
        # Missing protocol
        lambda: (
            valid_url.replace("http://", "").replace("https://", "")
            if valid_url
            else "www.example.com"
        ),
        # Invalid protocol
        lambda: (
            f"{random.choice(['htp://', 'htttp://', 'http:/', 'http:/'])}"
            f"{valid_url.split('://', 1)[1] if '://' in valid_url else 'example.com'}"
        ),
        # Missing domain
        lambda: random.choice(["http://", "https://", "http:///path"]),
        # Invalid port
        lambda: f"http://example.com:{random.choice(['', '99999', '-80', 'abc', '65536'])}",
        # Invalid characters in domain
        lambda: f"http://{random.choice(['exam ple', 'exam_ple', 'exam@ple', 'exam#ple', '-example', 'example-'])}.com",
        # Invalid TLD
        lambda: f"http://example.{random.choice(['', 'c', '123', 'verylongtld', '.com', 'com.'])}",
        # Spaces in URL
        lambda: f"http://{random.choice(['example .com', 'exam ple.com', 'example.com '])}/path",
        # Double slashes in wrong places
        lambda: "http:example.com" if not valid_url else valid_url.replace("://", ":"),
        # Special cases
        lambda: random.choice(
            [
                "javascript:alert('xss')",
                "file:///etc/passwd",
                "ftp://unsupported.com",
                "http://[not:valid:ipv6]",
                "http://300.300.300.300",
                "http://example..com",
                "http://.example.com",
                "http://example.com..",
                "http://exam ple.com",
                "http://example.com/path with spaces",
                "http://user:pass:word@example.com",
                "ht!tp://example.com",
            ]
        ),
    ]

    return random.choice(strategies)()


def _generate_invalid_uuid(valid_uuid: str = "") -> str:
    """Generate invalid UUIDs dynamically."""
    # Standard UUID format: 8-4-4-4-12 hexadecimal digits
    strategies = [
        # Character mutations
        lambda: _mutate_string(
            valid_uuid
            if len(valid_uuid) > 30
            else "550e8400-e29b-41d4-a716-446655440000"
        ),
        # Wrong length - too short
        lambda: "550e8400-e29b-41d4-a716",
        # Wrong length - too long
        lambda: "550e8400-e29b-41d4-a716-446655440000-extra",
        # Invalid characters (not hexadecimal)
        lambda: (
            f"{''.join(random.choices('ghijklmnopqrstuvwxyz!@#$%', k=8))}-"
            f"{''.join(random.choices('0123456789abcdef', k=4))}-"
            f"{''.join(random.choices('0123456789abcdef', k=4))}-"
            f"{''.join(random.choices('0123456789abcdef', k=4))}-"
            f"{''.join(random.choices('0123456789abcdef', k=12))}"
        ),
        # Wrong format - missing hyphens
        lambda: "550e8400e29b41d4a716446655440000",
        # Wrong format - extra hyphens
        lambda: "550e-8400-e29b-41d4-a716-4466-5544-0000",
        # Wrong format - wrong positions for hyphens
        lambda: "550e84-00e29b-41d4-a716446655440000",
        # Mixed case (some UUID validators are case-sensitive)
        lambda: "550E8400-E29B-41D4-A716-446655440000",
        # Partially valid
        lambda: f"550e8400-e29b-41d4-{random.choice(['xxxx', '12345', 'a71g'])}-446655440000",
        # Special cases
        lambda: random.choice(
            [
                "not-a-uuid",
                "00000000-0000-0000-0000-000000000000",  # Might be rejected as nil UUID
                "550e8400_e29b_41d4_a716_446655440000",  # Underscores instead of hyphens
                "{550e8400-e29b-41d4-a716-446655440000}",  # With braces
                "550e8400e29b41d4a716446655440000Z",  # Extra character
                "550e8400-e29b-11d4-a716-446655440000",  # Wrong version
                "g50e8400-e29b-41d4-a716-446655440000",  # Invalid hex at start
            ]
        ),
    ]

    return random.choice(strategies)()


def _introduce_format_error(
    data: Dict[str, Any], model_analysis: Dict[str, Any]
) -> bool:
    """Introduce format-specific errors for special types using dynamic generation."""
    fields = model_analysis["fields"]

    for field_name, field_info in fields.items():
        if field_name not in data:
            continue

        current_value = data[field_name]
        field_type = field_info["type"]

        if not isinstance(current_value, str):
            continue

        if field_type == "emailstr":
            # Dynamically generate invalid emails
            data[field_name] = _generate_invalid_email(current_value)
            return True

        elif field_type == "httpurl":
            # Dynamically generate invalid URLs
            data[field_name] = _generate_invalid_url(current_value)
            return True

        elif field_type == "uuid":
            # Dynamically generate invalid UUIDs
            data[field_name] = _generate_invalid_uuid(current_value)
            return True

        # Pattern constraint violations
        elif "pattern" in field_info["constraints"]:
            pattern = field_info["constraints"]["pattern"]
            if re.match(pattern, current_value):
                # Generate a string that doesn't match
                data[field_name] = "INVALID_FORMAT_123!@#"
                return True

        # Heuristic format errors for common field names
        elif field_name.lower() in [
            "email",
            "e_mail",
            "email_address",
            "contact_email",
            "user_email",
        ]:
            # Treat as email even if not EmailStr type
            if "@" in current_value and "." in current_value:
                data[field_name] = _generate_invalid_email(current_value)
                return True

        elif field_name.lower() in [
            "url",
            "website",
            "link",
            "uri",
            "homepage",
            "site_url",
        ]:
            # Treat as URL even if not HttpUrl type
            if current_value.startswith(("http://", "https://")):
                data[field_name] = _generate_invalid_url(current_value)
                return True

        elif field_name.lower() in [
            "uuid",
            "id",
            "guid",
            "uuid_field",
            "identifier",
            "unique_id",
        ]:
            # Treat as UUID-like field
            if len(current_value) > 30:  # Likely a UUID
                data[field_name] = _generate_invalid_uuid(current_value)
                return True

    return False


def _introduce_enum_error(data: Dict[str, Any], model_analysis: Dict[str, Any]) -> bool:
    """Introduce enum value errors with dynamic generation."""
    enums = model_analysis["enums"]

    for field_name, enum_values in enums.items():
        if field_name not in data:
            continue

        current_value = data[field_name]

        if current_value in enum_values:
            # Dynamically generate invalid enum values
            strategies = [
                # Case variations of valid values
                lambda: (
                    current_value.upper()
                    if current_value.islower()
                    else current_value.lower()
                ),
                lambda: current_value.swapcase(),
                lambda: (
                    current_value.capitalize()
                    if current_value.islower()
                    else current_value
                ),
                # Character mutations of valid values
                lambda: _mutate_string(current_value, mutation_rate=0.2),
                # Numeric variations
                lambda: str(random.randint(0, 999)),
                # Prefixed/suffixed variations
                lambda: random.choice(["INVALID_", "WRONG_", "BAD_"]) + current_value,
                lambda: current_value + random.choice(["_INVALID", "_WRONG", "2"]),
                # Common typos
                lambda: (
                    current_value[:-1]
                    if len(current_value) > 1
                    else current_value + "x"
                ),
                lambda: (
                    current_value[1:] if len(current_value) > 1 else "x" + current_value
                ),
                # Related but wrong values
                lambda: random.choice(
                    [
                        "undefined",
                        "null",
                        "none",
                        "unknown",
                        "default",
                        "true",
                        "false",
                        "0",
                        "1",
                        "-1",
                        "yes",
                        "no",
                        "N/A",
                        "TBD",
                    ]
                ),
                # Empty or None
                lambda: random.choice(["", None]),
            ]

            # Try to generate a value that's definitely not in the enum
            for _ in range(10):
                invalid_value = random.choice(strategies)()
                if invalid_value not in enum_values:
                    data[field_name] = invalid_value
                    return True

            # Fallback to a definitely invalid value
            data[field_name] = (
                f"DEFINITELY_NOT_A_VALID_ENUM_{random.randint(1000, 9999)}"
            )
            return True

    return False


def _introduce_required_field_error(
    data: Dict[str, Any], model_analysis: Dict[str, Any]
) -> bool:
    """Remove required fields."""
    fields = model_analysis["fields"]

    required_fields = [
        name
        for name, field_info in fields.items()
        if field_info["required"] and name in data
    ]

    if required_fields:
        field_to_remove = random.choice(required_fields)
        del data[field_to_remove]
        return True

    return False


def _convert_case(field_name: str) -> str:
    """Convert between camelCase and snake_case."""
    if "_" in field_name:
        # snake_case to camelCase
        parts = field_name.split("_")
        return parts[0] + "".join(word.capitalize() for word in parts[1:])
    else:
        # camelCase to snake_case (simple version)
        result = []
        for i, char in enumerate(field_name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.lower())
        return "".join(result)


def _introduce_extra_field_error(
    data: Dict[str, Any], model_analysis: Dict[str, Any]
) -> bool:
    """Add unexpected extra fields with context-aware generation."""
    # Analyze existing fields to generate plausible but wrong field names
    existing_fields = list(data.keys())

    strategies = [
        # Variations of existing fields
        lambda: (
            random.choice(existing_fields)
            + random.choice(["2", "_new", "_old", "_temp", "_backup"])
            if existing_fields
            else "extra_field"
        ),
        lambda: (
            random.choice(existing_fields)
            + random.choice(["_id", "_name", "_value", "_type"])
            if existing_fields
            else "extra_field"
        ),
        lambda: (
            random.choice(["temp_", "old_", "new_", "backup_"])
            + random.choice(existing_fields)
            if existing_fields
            else "extra_field"
        ),
        # Common typos of existing fields
        lambda: (
            _mutate_string(random.choice(existing_fields), mutation_rate=0.15)
            if existing_fields
            else "extra_field"
        ),
        # Commonly mistaken fields
        lambda: random.choice(
            [
                "id",
                "ID",
                "_id",
                "uid",
                "uuid",
                "created_at",
                "updated_at",
                "timestamp",
                "is_active",
                "active",
                "enabled",
                "status",
                "name",
                "title",
                "description",
                "value",
                "type",
                "kind",
                "category",
                "class",
                "data",
                "metadata",
                "extra",
                "additional_info",
                "user_id",
                "user",
                "owner",
                "author",
                "count",
                "total",
                "amount",
                "quantity",
                "_internal",
                "__private",
                "debug_info",
            ]
        ),
        # Underscored versions
        lambda: (
            "_" + random.choice(existing_fields)
            if existing_fields and not existing_fields[0].startswith("_")
            else "_extra_field"
        ),
        # Camelcase/snake_case conversions
        lambda: (
            _convert_case(random.choice(existing_fields))
            if existing_fields
            else "extraField"
        ),
    ]

    # Generate field name
    field_name = random.choice(strategies)()

    # Ensure uniqueness
    counter = 1
    original_name = field_name
    while field_name in data:
        field_name = f"{original_name}_{counter}"
        counter += 1

    # Generate contextual value based on field name
    value_strategies = [
        lambda: "default_value",
        lambda: random.randint(0, 100),
        lambda: random.choice([True, False]),
        lambda: [],
        lambda: {},
        lambda: None,
        lambda: random.choice([0.0, 1.0, -1.0]),
        lambda: f"value_{random.randint(1, 1000)}",
        lambda: {"nested": "data"},
        lambda: [1, 2, 3],
    ]

    # If field name suggests a type, use appropriate value
    if any(suffix in field_name.lower() for suffix in ["_id", "id", "_uid"]):
        data[field_name] = random.choice(
            [random.randint(1, 10000), f"id_{random.randint(1000, 9999)}"]
        )
    elif any(suffix in field_name.lower() for suffix in ["_at", "timestamp", "date"]):
        data[field_name] = random.choice(
            ["2024-01-01", "2024-01-01T00:00:00Z", 1704067200]
        )
    elif any(
        suffix in field_name.lower() for suffix in ["is_", "has_", "enabled", "active"]
    ):
        data[field_name] = random.choice([True, False, 1, 0, "true", "false"])
    else:
        data[field_name] = random.choice(value_strategies)()

    return True


def _introduce_nested_error(
    data: Dict[str, Any],
    model_analysis: Dict[str, Any],
    pydantic_model: Type[BaseModel],
) -> bool:
    """Introduce errors in nested objects."""
    nested_models = model_analysis["nested_models"]

    for field_name, nested_model_class in nested_models.items():
        if field_name not in data or not isinstance(data[field_name], dict):
            continue

        # Recursively introduce errors in nested objects
        nested_config = ErrorIntroductionConfig(
            error_types=[
                ErrorType.TYPE_ERROR,
                ErrorType.CONSTRAINT_ERROR,
                ErrorType.FORMAT_ERROR,
            ],
            max_errors=1,
        )

        erroneous_nested = introduce_error_for_pydantic(
            data[field_name], nested_model_class, nested_config
        )

        if erroneous_nested is not None:
            data[field_name] = erroneous_nested
            return True

    return False


def _introduce_list_error(data: Dict[str, Any], model_analysis: Dict[str, Any]) -> bool:
    """Introduce errors in list fields."""
    fields = model_analysis["fields"]

    for field_name, field_info in fields.items():
        if not field_info["is_list"] or field_name not in data:
            continue

        current_list = data[field_name]
        if not isinstance(current_list, list) or not current_list:
            continue

        # Different list error strategies
        error_strategies = [
            lambda lst: lst + [{"invalid": "item"}],  # Add invalid item
            lambda lst: lst + ["wrong_type"],  # Add wrong type item
            lambda lst: [None] + lst,  # Add None item
            lambda lst: lst[:1] if len(lst) > 1 else lst + ["extra"],  # Wrong length
        ]

        strategy = random.choice(error_strategies)
        try:
            data[field_name] = strategy(current_list.copy())
            return True
        except Exception:
            continue

    return False


def _introduce_validator_error(
    data: Dict[str, Any], model_analysis: Dict[str, Any]
) -> bool:
    """Introduce errors that would trigger custom validators."""
    fields = model_analysis["fields"]

    for field_name, field_info in fields.items():
        if not field_info["has_validators"] or field_name not in data:
            continue

        current_value = data[field_name]

        # Common validator violations
        if isinstance(current_value, str):
            validator_violating_values = [
                "",  # Empty string
                "   ",  # Whitespace only
                "\n\t",  # Whitespace characters
                "  " + current_value,  # Leading whitespace
                current_value + "  ",  # Trailing whitespace
            ]
            data[field_name] = random.choice(validator_violating_values)
            return True

    return False


# Backward compatibility: ensure the old function signature still works
def introduce_error_for_pydantic_old(
    data: Any, pydantic_model: Type[BaseModel], seed: Optional[int] = None
) -> Optional[Any]:
    """Legacy function signature for backward compatibility."""
    return introduce_error_for_pydantic(data, pydantic_model, seed=seed)
