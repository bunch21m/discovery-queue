"""Utility helpers for configurable auto-wishlist behaviour."""

from __future__ import annotations

import operator
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .game_functions import get_game_from_database


# Configuration keys managed by the auto wishlister.
# Keeps track of which keys expect string vs numeric rules.
_STRING_RULE_KEYS = {"genres", "categories", "description_keywords"}
_NUMERIC_RULE_KEYS = {"required_age", "price", "positive", "negative", "positive_ratio"}

# Default comparison operators for numeric keys.
# These are used if no comparison is specified in a rule 
# (shouldn't be possible after sanitisation which happens on the
# frontend, but included here just in case, and since this backend
# may be used independently of the frontend to generate test data).
_DEFAULT_COMPARISONS: Dict[str, str] = {
	"required_age": "lte",
	"price": "lte",
	"positive": "gte",
	"negative": "lte",
	"positive_ratio": "gte",
}

# Maps between some string in a JSON rule and the corresponding operator function
# in python's operator module.
_OPERATORS: Dict[str, Callable[[float, float], bool]] = {
	"lt": operator.lt,
	"lte": operator.le,
	"gt": operator.gt,
	"gte": operator.ge,
	"eq": operator.eq,
}

# In-memory storage of user configurations.
# A more robust implementation would persist these to a database or file TODO.
_USER_CONFIGS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


def _empty_config() -> Dict[str, List[Dict[str, Any]]]:
	"""
	This function creates and returns an empty auto-wishlist configuration
	with all expected keys initialized to empty lists. It's used to ensure
	that a configuration always contains all necessary keys, even if no
    rules have been defined for them.
	
	Returns:
        A dictionary representing an empty auto-wishlist configuration.
    """
	
	config: Dict[str, List[Dict[str, Any]]] = {}
	for key in _STRING_RULE_KEYS.union(_NUMERIC_RULE_KEYS):
		config[key] = []
	return config


def _normalise_key(key: str) -> Optional[str]:
	"""
	Normalises various possible key names to canonical configuration keys.
	This helps in mapping user-provided keys to the expected internal keys.
	For example, "genre" and "genres" both map to "genres". This helps if
	a user provides slightly different key names in their configuration,
	though much like with the _DEFAULT_COMPARISONS, this shouldn't be
    necessary after sanitisation on the frontend.
	
	Args:
        key: The input key string to normalise.
		
    Returns:
        The canonical configuration key, or None if the key is unrecognised.
    """
	
	mapping = {
		"genres": "genres",
		"genre": "genres",
		"categories": "categories",
		"category": "categories",
		"descriptionkeywords": "description_keywords",
		"descriptionkeyword": "description_keywords",
		"description_keywords": "description_keywords",
		"requiredage": "required_age",
		"required_age": "required_age",
		"price": "price",
		"positive": "positive",
		"negative": "negative",
		"positiveratio": "positive_ratio",
		"positive_ratio": "positive_ratio",
	}
	return mapping.get(key.replace(" ", "").lower())


def _ensure_float(value: Any) -> float:
	"""
	Ensures that the provided value can be converted to a float.
	Useful for sanitising numeric rule values.
	
	Args:
        value: The value to convert.
	
    Returns:
        The float representation of the value.
    """
	
	if value is None or value == "":
		raise ValueError("Numeric value cannot be empty")
	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Invalid numeric value '{value}'") from exc


def _ensure_string(value: Any) -> str:
	"""
	Ensures that the provided value is a non-empty string.
	Useful for making sure a string rule is actually valid.
	
	Args:
        value: The value to check.
		
    Returns:
        The stripped string representation of the value.
    """
	
	if value is None:
		raise ValueError("String value cannot be empty")
	string_value = str(value).strip()
	if not string_value:
		raise ValueError("String value cannot be empty")
	return string_value


def _sanitise_string_rules(rules: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
	"""
	Sanitises a list of string-based rules by ensuring each rule has valid
    string values and numeric scores.
	
	Args:
        rules: An iterable of rule mappings to sanitise.
		
    Returns:
        A list of sanitised string rules.
    """
    
	sanitised: List[Dict[str, Any]] = []
	for rule in rules:
		try:
			value = _ensure_string(rule.get("value"))
			score = _ensure_float(rule.get("score"))
		except ValueError:
			continue
		sanitised.append({"value": value.lower(), "score": score})
	return sanitised


def _sanitise_numeric_rules(
	key: str, rules: Iterable[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
	"""
    Sanitises a list of numeric-based rules by ensuring each rule has valid
    numeric values, scores, and comparison operators.
	
	Args:
        key: The configuration key these rules correspond to.
        rules: An iterable of rule mappings to sanitise.
        
    Returns:
        A list of sanitised numeric rules.
	"""
	
	sanitised: List[Dict[str, Any]] = []
	default_comparison = _DEFAULT_COMPARISONS.get(key, "gte")
	for rule in rules:
		try:
			value = _ensure_float(rule.get("value"))
			score = _ensure_float(rule.get("score"))
		except ValueError:
			continue
		comparison = str(rule.get("comparison") or default_comparison).lower()
		if comparison not in _OPERATORS:
			comparison = default_comparison
		sanitised.append({
			"comparison": comparison,
			"value": value,
			"score": score,
		})
	return sanitised


def _sanitise_config(raw_config: Optional[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
	"""
    Sanitises a raw auto-wishlist configuration provided by a user.

    Args:
        raw_config: The raw configuration mapping to sanitise.
		
    Returns:
        A sanitised configuration dictionary.
	"""
	
	config = _empty_config()
	if not raw_config:
		return config

	for raw_key, rules in raw_config.items():
		canonical = _normalise_key(raw_key)
		if canonical is None:
			continue
		if canonical in _STRING_RULE_KEYS:
			config[canonical] = _sanitise_string_rules(rules if isinstance(rules, Iterable) else [])
		elif canonical in _NUMERIC_RULE_KEYS:
			config[canonical] = _sanitise_numeric_rules(canonical, rules if isinstance(rules, Iterable) else [])
	return config


def set_user_config(username: str, raw_config: Optional[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
	"""
	Sets and stores the auto-wishlist configuration for a user.
	
	Args:
        username: The username to associate the configuration with.
        raw_config: The raw configuration mapping to sanitise and store.
		
    Returns:
        The stored sanitised configuration dictionary.
    """

	clean_username = (username or "").strip() or "__default__"
	config = _sanitise_config(raw_config)
	_USER_CONFIGS[clean_username] = config
	return config


def get_user_config(username: str) -> Dict[str, List[Dict[str, Any]]]:
	"""
    Retrieves the auto-wishlist configuration for a user.
	
    Args:
        username: The username whose configuration to retrieve.

    Returns:
        The stored sanitised configuration dictionary.
	"""

	clean_username = (username or "").strip() or "__default__"
	return _USER_CONFIGS.get(clean_username, _empty_config())


def _coerce_iterable(items: Any) -> List[str]:
	"""
	Coerces the input into a list of strings. If the input is None, returns an empty list.
    If the input is a single string, returns a list containing that string. If the input is an iterable,
    attempts to convert each item to a string and returns the resulting list. If conversion fails,
    returns an empty list.
	
	Args:
        items: The input to coerce into a list of strings.
		
    Returns:
        A list of strings.
    """
	
	if not items:
		return []
	if isinstance(items, str):
		return [items]
	try:
		return [str(item) for item in items]
	except TypeError:
		return []


def _to_lower_set(items: Iterable[str]) -> set[str]:
	"""
	Converts an iterable of strings to a set of lowercase strings.
	Used for case-insensitive matching of features like genres and categories.
	
	Args:
        items: An iterable of strings to convert.
		
    Returns:
        A set of lowercase strings.
    """
	return {item.lower() for item in items if isinstance(item, str)}


def _extract_numeric(game: Mapping[str, Any], key: str) -> float:
	"""
	Extracts a numeric value from a game mapping for the given key.
	Uses float conversion and defaults to 0.0 if the value is missing or invalid.
	Used to retrieve numeric features for scoring.
	
	Args:
        game: The game mapping to extract the value from.
        key: The key whose value to extract.
		
    Returns:
        The extracted numeric value as a float. Defaults to 0.0 if missing or invalid.
    """
	
	value = game.get(key)
	if value in (None, ""):
		return 0.0
	try:
		return float(value)
	except (TypeError, ValueError):
		return 0.0


def _positive_ratio(game: Mapping[str, Any]) -> float:
	"""
	Calculates the ratio of positive reviews to total reviews for a game.
	Used as a helper when scoring based on review positivity to figure out
	if a game satisfies the ratio-based rule (since a game does not directly
	store the ratio of positive to total reviews).
    Returns 0.0 if there are no reviews to avoid division by zero.
	
	Args:
        game: The game mapping to calculate the ratio for.
		
    Returns:
        The positive review ratio as a float between 0.0 and 1.0.
    """
	
	positive = _extract_numeric(game, "positive")
	negative = _extract_numeric(game, "negative")
	total = positive + negative
	if total <= 0:
		return 0.0
	return positive / total


def score_game(game: Optional[Mapping[str, Any]], config: Optional[Mapping[str, Any]] = None) -> float:
	"""
	Scores a game based on the provided auto-wishlist configuration.
	Uses all the configured rules to check over all features of the game
    and sums up the scores for each matching rule.
	
	Args:
        game: The game mapping to score.
        config: The auto-wishlist configuration to use for scoring.
		
    Returns:
        The total score for the game as a float.
    """

	if not game:
		return 0.0

	active_config = _sanitise_config(config) if config is not None else None
	if active_config is None:
		# None indicates we should use pre-sanitised config (already stored)
		active_config = _empty_config()

	total_score = 0.0

	features_genres = _to_lower_set(_coerce_iterable(game.get("genres")))
	features_categories = _to_lower_set(_coerce_iterable(game.get("categories")))
	short_description = str(game.get("short_description") or "")
	detailed_description = str(game.get("detailed_description") or "")
	combined_description = f"{short_description}\n{detailed_description}".lower()

	for rule in active_config.get("genres", []):
		if rule["value"] in features_genres:
			total_score += rule["score"]

	for rule in active_config.get("categories", []):
		if rule["value"] in features_categories:
			total_score += rule["score"]

	for rule in active_config.get("description_keywords", []):
		if rule["value"] in combined_description:
			total_score += rule["score"]

	numeric_values: Dict[str, float] = {
		"required_age": _extract_numeric(game, "required_age"),
		"price": _extract_numeric(game, "price"),
		"positive": _extract_numeric(game, "positive"),
		"negative": _extract_numeric(game, "negative"),
		"positive_ratio": _positive_ratio(game),
	}

	for key, rules in active_config.items():
		if key not in _NUMERIC_RULE_KEYS:
			continue
		value = numeric_values.get(key, 0.0)
		for rule in rules:
			comparator = _OPERATORS[rule["comparison"]]
			if comparator(value, rule["value"]):
				total_score += rule["score"]

	return total_score


def score_game_for_user(game: Optional[Mapping[str, Any]], username: str) -> float:
	"""
	Fetches the auto-wishlist configuration for a user and scores the game accordingly.
	Wrapper for conveniently calling score_game with the appropriate config
	when scoring based on a username since the config is stored per-user.
	
	Args:
        game: The game mapping to score.
        username: The username whose configuration to use for scoring.
		
    Returns:
        The total score for the game as a float.
	"""

	config = get_user_config(username)
	return score_game(game, config)


def score_app_id_for_user(app_id: str, username: str) -> float:
	"""
	Scores a game identified by its app ID for a specific user.
    Fetches the game from the database and uses the user's configuration
	to score it. Yet another convenience wrapper for calling score_game,
	this time with an app ID and username.
	
	Args:
        app_id: The app ID of the game to score.
        username: The username whose configuration to use for scoring.
		
    Returns:
        The total score for the game as a float.
	"""

	if not app_id:
		return 0.0
	game = get_game_from_database(app_id)
	if game is None:
		raise LookupError(f"Game {app_id} not found")
	return score_game_for_user(game, username)


def action_for_score(score: float) -> str:
	"""
	Determines the action ("wishlist" or "skip") to take for a game
	based on the score. Positive scores indicate "wishlist",
	while zero or negative scores indicate "skip".
	
	Args:
        score: The score of the game.
		
    Returns:
        The action string: "wishlist" or "skip".
    """

	return "wishlist" if score > 0 else "skip"
