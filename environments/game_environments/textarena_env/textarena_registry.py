#!/usr/bin/env python3
"""
TextArena Game Registry

Provides a registry system for dynamically discovering and managing TextArena games.
Supports random selection, filtering by player count, and game metadata caching.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from dotenv import load_dotenv
from textarena.envs.registration import ENV_REGISTRY

# Ensure all built-in game modules are imported so the registry is fully populated
try:
    # Some installs lazily register games; importing ensures full registry
    import textarena.envs.multi_player  # noqa: F401
    import textarena.envs.single_player  # noqa: F401
    import textarena.envs.two_player  # noqa: F401
except Exception:
    # Don't fail discovery if imports change upstream; we log during discovery
    pass

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class TextArenaGameRegistry:
    """Registry for TextArena games with dynamic discovery and categorization.

    Adds a JSON-backed persistent cache to avoid expensive environment initialization
    during discovery. The cache is written incrementally to allow resume on interruption.
    """

    # Class-level caches shared across all instances
    _discovered = False
    _single_player_games: List[str] = []
    _two_player_games: List[str] = []
    _multi_player_games: List[str] = []
    _unknown_games: List[str] = []  # Failed detection - never selected
    _invalid_games: List[str] = []  # Failed validation during selection
    _error_games: List[str] = []  # Failed to import/instantiate
    _player_count_cache: Dict[str, Tuple[int, int]] = {}
    _game_metadata_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        seed: Optional[int] = None,
        max_player_detection: int = 4,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
        reset_cache: bool = False,
    ):
        self.rng = random.Random(seed)
        self.max_player_detection = max_player_detection
        # Cache configuration
        self.use_cache = use_cache
        self.reset_cache = reset_cache
        if cache_path is None:
            # Default to a cache next to this file inside the repo so it can be committed
            cache_path = os.path.join(
                os.path.dirname(__file__), "textarena_registry_cache.json"
            )
        self.cache_path = cache_path

    # ---------------------------
    # Persistent cache management
    # ---------------------------
    def _load_cache_from_file(self) -> Dict[str, Any]:
        """Load cache file if present; return dict or empty if missing/disabled.

        Populates class-level caches for quick access.
        """
        if not self.use_cache:
            return {}
        try:
            if self.reset_cache and os.path.exists(self.cache_path):
                os.remove(self.cache_path)
            if not os.path.exists(self.cache_path):
                return {}
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            games = data.get("games", {})

            # Reset current in-memory caches before loading
            TextArenaGameRegistry._player_count_cache.clear()
            TextArenaGameRegistry._game_metadata_cache.clear()
            TextArenaGameRegistry._single_player_games.clear()
            TextArenaGameRegistry._two_player_games.clear()
            TextArenaGameRegistry._multi_player_games.clear()
            TextArenaGameRegistry._unknown_games.clear()
            TextArenaGameRegistry._invalid_games.clear()
            TextArenaGameRegistry._error_games.clear()

            for env_id, meta in games.items():
                min_p = int(meta.get("min_players", 0))
                max_p = int(meta.get("max_players", 0))
                TextArenaGameRegistry._player_count_cache[env_id] = (min_p, max_p)
                # Rebuild metadata
                game_type = meta.get("game_type") or (
                    "unknown"
                    if max_p == 0
                    else (
                        "single" if max_p == 1 else ("two" if max_p == 2 else "multi")
                    )
                )
                TextArenaGameRegistry._game_metadata_cache[env_id] = {
                    "env_id": env_id,
                    "min_players": min_p,
                    "max_players": max_p,
                    "game_type": game_type,
                    "name": env_id.split("-")[0] if "-" in env_id else env_id,
                }
            return data
        except Exception as e:
            logger.warning(
                f"Failed to load TextArena registry cache from {self.cache_path}: {e}"
            )
            return {}

    def _persist_cache_to_file(self) -> None:
        """Persist current known metadata to cache file (best-effort)."""
        if not self.use_cache:
            return
        try:
            # Build games map from metadata cache
            games: Dict[str, Any] = {}
            for env_id, meta in TextArenaGameRegistry._game_metadata_cache.items():
                games[env_id] = {
                    "min_players": meta.get("min_players", 0),
                    "max_players": meta.get("max_players", 0),
                    "game_type": meta.get("game_type", "unknown"),
                }
            payload = {
                "version": 1,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "max_player_detection": self.max_player_detection,
                "games": games,
            }
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.warning(
                f"Failed to write TextArena registry cache to {self.cache_path}: {e}"
            )

    def _extend_player_count_detection(
        self, env_id: str, start_from: int
    ) -> Tuple[int, int]:
        """Extend detection for an environment for player counts >= start_from.

        Uses existing cached min/max if available and attempts additional counts up to
        self.max_player_detection. Returns the updated (min_players, max_players).
        """
        cached = TextArenaGameRegistry._player_count_cache.get(env_id, (0, 0))
        valid_counts: List[int] = []
        # Seed with previously known successful counts if any
        if cached != (0, 0):
            valid_counts.extend(range(cached[0], cached[1] + 1))
        # Try higher counts
        for num_players in range(max(1, start_from), self.max_player_detection + 1):
            try:
                env = ta.make(env_id)
                try:
                    env.reset(num_players=num_players)
                except TypeError:
                    if num_players == 1:
                        env.reset()
                        valid_counts.append(1)
                    # If signature doesn't accept num_players, we cannot test >1 reliably
                    break
                valid_counts.append(num_players)
                try:
                    env.close()
                except Exception:
                    pass
            except Exception:
                # Silently skip unsupported counts here (we don't want to spam logs)
                continue

        if valid_counts:
            min_players = min(valid_counts)
            max_players = max(valid_counts)
            TextArenaGameRegistry._player_count_cache[env_id] = (
                min_players,
                max_players,
            )
            # Update metadata
            if max_players == 0:
                game_type = "unknown"
            elif max_players == 1:
                game_type = "single"
            elif max_players == 2:
                game_type = "two"
            else:
                game_type = "multi"
            TextArenaGameRegistry._game_metadata_cache[env_id] = {
                "env_id": env_id,
                "min_players": min_players,
                "max_players": max_players,
                "game_type": game_type,
                "name": env_id.split("-")[0] if "-" in env_id else env_id,
            }
            return (min_players, max_players)
        else:
            # If still unknown, set to (0,0)
            TextArenaGameRegistry._player_count_cache[env_id] = (0, 0)
            TextArenaGameRegistry._game_metadata_cache[env_id] = {
                "env_id": env_id,
                "min_players": 0,
                "max_players": 0,
                "game_type": "unknown",
                "name": env_id.split("-")[0] if "-" in env_id else env_id,
            }
            return (0, 0)

    def _fallback_player_range_from_registration(self) -> Dict[str, Tuple[int, int]]:
        """Parse textarena.envs.__init__ for bracketed player hints and ids.

        Returns a mapping of env_id -> (min_players, max_players) inferred from comments
        like "# Snake [2-15 Players]" and the subsequent register/register_with_versions
        lines. This avoids importing broken env modules just to discover player counts.
        """
        try:
            import textarena.envs as ta_envs  # type: ignore

            init_path = ta_envs.__file__
            if not init_path or not os.path.exists(init_path):
                return {}
            with open(init_path, "r", encoding="utf-8") as f:
                src = f.read()
            lines = src.splitlines()
            current_range: Optional[Tuple[int, int]] = None
            hint_re = re.compile(
                r"\[(\d+)(?:\s*[-–]\s*(\d+))?\s*Players?\]", re.IGNORECASE
            )
            id_re = re.compile(r"register(?:_with_versions)?\(id=\"([^\"]+)\"")
            result: Dict[str, Tuple[int, int]] = {}
            for line in lines:
                cmt = line.strip()
                if cmt.startswith("#"):
                    m = hint_re.search(cmt)
                    if m:
                        lo = int(m.group(1))
                        hi = int(m.group(2)) if m.group(2) else lo
                        current_range = (lo, hi)
                        continue
                m = id_re.search(line)
                if m and current_range is not None:
                    env_id = m.group(1)
                    result[env_id] = current_range
            return result
        except Exception:
            return {}

    def discover_games(self) -> None:
        """Discover all available TextArena games from the registry.

        Uses a persistent JSON cache when available to avoid expensive initialization.
        Missing entries will be detected and appended, saving incrementally to allow resume.
        """
        if TextArenaGameRegistry._discovered:
            return

        try:
            # Attempt importing categories again in case discovery is first touch
            try:
                import textarena.envs.multi_player  # noqa: F401
                import textarena.envs.single_player  # noqa: F401
                import textarena.envs.two_player  # noqa: F401
            except Exception as e:
                logger.debug(f"Optional category imports failed (safe to ignore): {e}")

            # Get all registered environment IDs
            all_games = list(ENV_REGISTRY.keys())
            logger.info(
                f"Starting discovery of {len(all_games)} TextArena environments..."
            )

            # Try loading cache first
            cache_data = self._load_cache_from_file()
            # Parse static player-count hints from registration file once
            static_ranges = self._fallback_player_range_from_registration()

            # Clear classifications; will rebuild from metadata for only current all_games
            TextArenaGameRegistry._single_player_games.clear()
            TextArenaGameRegistry._two_player_games.clear()
            TextArenaGameRegistry._multi_player_games.clear()
            TextArenaGameRegistry._unknown_games.clear()
            TextArenaGameRegistry._invalid_games.clear()
            TextArenaGameRegistry._error_games.clear()

            # If cache was built with a lower detection ceiling, extend detection
            cache_max_detect = None
            if isinstance(cache_data, dict):
                cache_max_detect = cache_data.get("max_player_detection")
            if (
                cache_max_detect is not None
                and cache_max_detect < self.max_player_detection
            ):
                logger.info(
                    "Extending player-count detection from cached max=%s to new max=%s",
                    cache_max_detect,
                    self.max_player_detection,
                )
                updated = 0
                for idx, env_id in enumerate(all_games, start=1):
                    self._extend_player_count_detection(
                        env_id, start_from=cache_max_detect + 1
                    )
                    # Persist periodically to allow resume and avoid data loss
                    if idx % 25 == 0:
                        self._persist_cache_to_file()
                    updated += 1
                # Persist after extension
                self._persist_cache_to_file()

            # Reconcile metadata using static hints and extend where appropriate
            # 1) For any envs with known static ranges, update metadata if it's missing or weaker
            if static_ranges:
                # Build mapping from base id -> all registered ids that share this base
                base_to_ids: Dict[str, List[str]] = {}
                for base_id, rng in static_ranges.items():
                    matches = [
                        gid
                        for gid in all_games
                        if gid == base_id or gid.startswith(base_id + "-")
                    ]
                    if matches:
                        base_to_ids[base_id] = matches
                # Apply updates
                for base_id, ids in base_to_ids.items():
                    s_min, s_max = static_ranges[base_id]
                    for env_id in ids:
                        meta = TextArenaGameRegistry._game_metadata_cache.get(env_id)
                        current_min, current_max = (
                            (meta.get("min_players", 0), meta.get("max_players", 0))
                            if meta
                            else (0, 0)
                        )
                        # If unknown or clearly underspecified, try to extend via live detection first
                        if current_max <= 2 and s_max > 2:
                            try:
                                self._extend_player_count_detection(
                                    env_id, start_from=max(3, current_max + 1)
                                )
                                # refresh after attempt
                                meta = TextArenaGameRegistry._game_metadata_cache.get(
                                    env_id
                                )
                                current_min, current_max = (
                                    (
                                        meta.get("min_players", 0),
                                        meta.get("max_players", 0),
                                    )
                                    if meta
                                    else (0, 0)
                                )
                            except Exception:
                                pass
                        # If still missing or weaker than static, adopt static values
                        if current_max < s_max:
                            game_type = (
                                "unknown"
                                if s_max == 0
                                else (
                                    "single"
                                    if s_max == 1
                                    else ("two" if s_max == 2 else "multi")
                                )
                            )
                            TextArenaGameRegistry._player_count_cache[env_id] = (
                                s_min,
                                s_max,
                            )
                            TextArenaGameRegistry._game_metadata_cache[env_id] = {
                                "env_id": env_id,
                                "min_players": s_min,
                                "max_players": s_max,
                                "game_type": game_type,
                                "name": (
                                    env_id.split("-")[0] if "-" in env_id else env_id
                                ),
                            }
                # Persist reconciliation
                self._persist_cache_to_file()

            # Rebuild classifications from metadata cache for current set of games
            TextArenaGameRegistry._single_player_games.clear()
            TextArenaGameRegistry._two_player_games.clear()
            TextArenaGameRegistry._multi_player_games.clear()
            TextArenaGameRegistry._unknown_games.clear()
            TextArenaGameRegistry._invalid_games.clear()
            for env_id in all_games:
                meta = TextArenaGameRegistry._game_metadata_cache.get(env_id)
                if meta is None:
                    continue
                if meta.get("game_type") == "error":
                    TextArenaGameRegistry._error_games.append(env_id)
                    continue
                max_players = int(meta.get("max_players", 0))
                if max_players == 0:
                    TextArenaGameRegistry._unknown_games.append(env_id)
                elif max_players == 1:
                    TextArenaGameRegistry._single_player_games.append(env_id)
                elif max_players == 2:
                    TextArenaGameRegistry._two_player_games.append(env_id)
                else:
                    TextArenaGameRegistry._multi_player_games.append(env_id)

            # Detect only missing games and persist incrementally so runs can resume
            missing_env_ids = [
                env_id
                for env_id in all_games
                if env_id not in TextArenaGameRegistry._game_metadata_cache
            ]
            total_missing = len(missing_env_ids)
            for idx, env_id in enumerate(missing_env_ids, start=1):
                # Prefer static inference from registration if available (no instantiation)
                if env_id in static_ranges:
                    min_players, max_players = static_ranges[env_id]
                    # Prepopulate metadata from static if needed
                    if env_id not in TextArenaGameRegistry._game_metadata_cache:
                        game_type = (
                            "unknown"
                            if max_players == 0
                            else (
                                "single"
                                if max_players == 1
                                else ("two" if max_players == 2 else "multi")
                            )
                        )
                        TextArenaGameRegistry._player_count_cache[env_id] = (
                            min_players,
                            max_players,
                        )
                        TextArenaGameRegistry._game_metadata_cache[env_id] = {
                            "env_id": env_id,
                            "min_players": min_players,
                            "max_players": max_players,
                            "game_type": game_type,
                            "name": env_id.split("-")[0] if "-" in env_id else env_id,
                        }
                else:
                    min_players, max_players = self._detect_player_count(env_id)

                # Classify using metadata (allows "error" override set by detector)
                meta = TextArenaGameRegistry._game_metadata_cache.get(env_id, {})
                if meta.get("game_type") == "error":
                    TextArenaGameRegistry._error_games.append(env_id)
                else:
                    if max_players == 0:  # Failed detection
                        TextArenaGameRegistry._unknown_games.append(env_id)
                    elif max_players == 1:
                        TextArenaGameRegistry._single_player_games.append(env_id)
                    elif max_players == 2:
                        TextArenaGameRegistry._two_player_games.append(env_id)
                    else:
                        TextArenaGameRegistry._multi_player_games.append(env_id)

                # Ensure metadata cache updated for this env and persist to file for resume
                # Determine game type
                if max_players == 0:
                    game_type = "unknown"
                elif max_players == 1:
                    game_type = "single"
                elif max_players == 2:
                    game_type = "two"
                else:
                    game_type = "multi"
                TextArenaGameRegistry._player_count_cache[env_id] = (
                    min_players,
                    max_players,
                )
                TextArenaGameRegistry._game_metadata_cache[env_id] = {
                    "env_id": env_id,
                    "min_players": min_players,
                    "max_players": max_players,
                    "game_type": game_type,
                    "name": env_id.split("-")[0] if "-" in env_id else env_id,
                }

                # Persist after each classification to enable resume
                self._persist_cache_to_file()

                logger.info(
                    f"  Classified {idx}/{total_missing} missing games (env: {env_id}, type: {game_type})"
                )

            # Log summary (not sample)
            logger.info(f"Discovery complete! Found {len(all_games)} games:")
            logger.info(
                f"  Single-player: {len(TextArenaGameRegistry._single_player_games)}"
            )
            logger.info(f"  Two-player: {len(TextArenaGameRegistry._two_player_games)}")
            logger.info(
                f"  Multi-player: {len(TextArenaGameRegistry._multi_player_games)}"
            )
            logger.info(
                f"  Unknown (excluded): {len(TextArenaGameRegistry._unknown_games)}"
            )
            logger.info(
                f"  Error (excluded): {len(TextArenaGameRegistry._error_games)}"
            )
            logger.info(
                f"  Invalid (excluded): {len(TextArenaGameRegistry._invalid_games)}"
            )

            # Persist final state just in case anything changed via classification rebuild
            self._persist_cache_to_file()

            TextArenaGameRegistry._discovered = True

        except Exception as e:
            logger.error(f"Failed to discover TextArena games: {e}")
            raise

    def _detect_player_count(self, env_id: str) -> Tuple[int, int]:
        """
        Detect the valid player count range for a game by trying to instantiate it.

        Returns:
            Tuple of (min_players, max_players)
        """
        # Check class-level cache first
        if env_id in TextArenaGameRegistry._player_count_cache:
            return TextArenaGameRegistry._player_count_cache[env_id]

        # Try to detect by instantiation
        logger.warning(f"Detecting player count for {env_id}")

        # Try player counts up to max_player_detection
        valid_counts = []
        had_make_error = False
        for num_players in range(1, self.max_player_detection + 1):
            try:
                env = ta.make(env_id)
                # Try to reset with this player count
                try:
                    env.reset(num_players=num_players)
                except TypeError:
                    # Some games don't accept num_players argument
                    if num_players == 1:
                        env.reset()  # Try without num_players
                        valid_counts.append(1)
                    break  # Can't test other player counts

                valid_counts.append(num_players)

                # Try to close, but don't fail if it errors
                try:
                    env.close()
                except Exception:
                    pass  # Ignore close errors

            except Exception as e:
                # Distinguish import/instantiation errors from unsupported player counts
                had_make_error = True
                logger.warning(
                    f"Game {env_id} failed to initialize at {num_players} players: {e}"
                )
                continue

        if valid_counts:
            min_players = min(valid_counts)
            max_players = max(valid_counts)
            # Cache the result in class-level cache
            TextArenaGameRegistry._player_count_cache[env_id] = (
                min_players,
                max_players,
            )
            return (min_players, max_players)
        else:
            # Mark as error if initialization consistently failed; otherwise unknown
            TextArenaGameRegistry._player_count_cache[env_id] = (0, 0)
            if had_make_error:
                TextArenaGameRegistry._game_metadata_cache[env_id] = {
                    "env_id": env_id,
                    "min_players": 0,
                    "max_players": 0,
                    "game_type": "error",
                    "name": env_id.split("-")[0] if "-" in env_id else env_id,
                }
                logger.debug(f"Environment initialization error for {env_id}")
            else:
                logger.debug(f"Could not detect player count for {env_id}")
            return (0, 0)

    def get_game_metadata(self, env_id: str) -> Dict[str, Any]:
        """Get or create metadata for a game."""
        # Check class-level cache first
        if env_id not in TextArenaGameRegistry._game_metadata_cache:
            # Get player count from cache (should already be detected during discovery)
            if env_id in TextArenaGameRegistry._player_count_cache:
                min_players, max_players = TextArenaGameRegistry._player_count_cache[
                    env_id
                ]
            else:
                # Fallback to detection if not in cache
                min_players, max_players = self._detect_player_count(env_id)

            # Determine game type
            if max_players == 0:
                game_type = "unknown"
            elif max_players == 1:
                game_type = "single"
            elif max_players == 2:
                game_type = "two"
            else:
                game_type = "multi"

            TextArenaGameRegistry._game_metadata_cache[env_id] = {
                "env_id": env_id,
                "min_players": min_players,
                "max_players": max_players,
                "game_type": game_type,
                "name": env_id.split("-")[0] if "-" in env_id else env_id,
            }

        return TextArenaGameRegistry._game_metadata_cache[env_id]

    def get_random_game(
        self,
        game_filter: str = "all",
        exclude_games: Optional[List[str]] = None,
        max_players: Optional[int] = None,
        min_players: Optional[int] = None,
        validate_on_select: bool = True,
        max_selection_attempts: int = 25,
        specific_game: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get a random game with optional filtering.

        Args:
            game_filter: "single", "two", "multi", or "all"
            exclude_games: List of game IDs to exclude (substring match)
            max_players: Maximum number of players supported
            min_players: Minimum number of players supported
            validate_on_select: Whether to quickly validate env create/reset
            max_selection_attempts: Attempts to find a valid env
            specific_game: Exact env_id to force selection (for debugging)

        Returns:
            Tuple of (env_id, metadata)
        """
        if not TextArenaGameRegistry._discovered:
            self.discover_games()

        # Start with pre-classified games based on filter (NEVER include unknown games)
        if game_filter == "single":
            candidates = TextArenaGameRegistry._single_player_games.copy()
        elif game_filter == "two":
            candidates = TextArenaGameRegistry._two_player_games.copy()
        elif game_filter == "multi":
            candidates = TextArenaGameRegistry._multi_player_games.copy()
        else:  # "all"
            candidates = (
                TextArenaGameRegistry._single_player_games
                + TextArenaGameRegistry._two_player_games
                + TextArenaGameRegistry._multi_player_games
            )

        # If a specific game is requested, try that first
        if specific_game:
            specific_matches = [
                g
                for g in candidates
                if g == specific_game or g.lower() == specific_game.lower()
            ]
            if not specific_matches:
                raise ValueError(
                    f"Specific game '{specific_game}' not found among candidates"
                )
            candidates = specific_matches

        # Apply exclude filter
        if exclude_games:
            candidates = [
                g
                for g in candidates
                if not any(exc.lower() in g.lower() for exc in exclude_games)
            ]

        # Remove any games previously marked invalid
        if TextArenaGameRegistry._invalid_games:
            invalid_set = set(TextArenaGameRegistry._invalid_games)
            candidates = [g for g in candidates if g not in invalid_set]

        # Apply additional player count filters if specified
        if min_players is not None or max_players is not None:
            filtered = []
            for env_id in candidates:
                metadata = self.get_game_metadata(env_id)
                if min_players and metadata["max_players"] < min_players:
                    continue
                if max_players and metadata["min_players"] > max_players:
                    continue
                filtered.append(env_id)
            candidates = filtered

        if not candidates:
            raise ValueError(
                f"No games match the specified filters (filter={game_filter}, candidates after filtering: 0)"
            )

        # Select and optionally validate a random game
        attempts = 0
        last_err: Optional[Exception] = None
        while attempts < max_selection_attempts and candidates:
            attempts += 1
            selected = self.rng.choice(candidates)
            metadata = self.get_game_metadata(selected)

            if not validate_on_select:
                return selected, metadata

            try:
                if self._quick_validate(selected, metadata):
                    return selected, metadata
                else:
                    # Mark invalid and remove from candidates
                    self._mark_invalid(selected)
                    candidates = [g for g in candidates if g != selected]
            except Exception as e:
                last_err = e
                self._mark_invalid(selected)
                candidates = [g for g in candidates if g != selected]

        # If we get here, validation failed repeatedly
        if last_err:
            logger.warning(
                f"Failed to validate a game after {attempts} attempts: {last_err}"
            )
        raise ValueError("No valid games available after validation attempts")

    def list_available_games(self, game_filter: str = "all") -> List[str]:
        """List all available games, optionally filtered by type."""
        if not TextArenaGameRegistry._discovered:
            self.discover_games()

        # Return pre-classified lists (never includes unknown games)
        if game_filter == "single":
            base = TextArenaGameRegistry._single_player_games
        elif game_filter == "two":
            base = TextArenaGameRegistry._two_player_games
        elif game_filter == "multi":
            base = TextArenaGameRegistry._multi_player_games
        elif game_filter == "all":
            base = (
                TextArenaGameRegistry._single_player_games
                + TextArenaGameRegistry._two_player_games
                + TextArenaGameRegistry._multi_player_games
            )
        else:
            base = []

        # Filter out invalid games consistently
        if TextArenaGameRegistry._invalid_games:
            invalid_set = set(TextArenaGameRegistry._invalid_games)
            return [g for g in base if g not in invalid_set]
        return list(base)

    def _mark_invalid(self, env_id: str) -> None:
        """Mark a game as invalid and remove it from classified lists."""
        if env_id not in TextArenaGameRegistry._invalid_games:
            TextArenaGameRegistry._invalid_games.append(env_id)
        for lst in (
            TextArenaGameRegistry._single_player_games,
            TextArenaGameRegistry._two_player_games,
            TextArenaGameRegistry._multi_player_games,
        ):
            if env_id in lst:
                try:
                    lst.remove(env_id)
                except ValueError:
                    pass

    def _quick_validate(self, env_id: str, metadata: Dict[str, Any]) -> bool:
        """Quickly validate that a game can be made, reset, and observed.

        Keeps this very lightweight to avoid slowing down selection too much.
        """
        try:
            env = ta.make(env_id)
            num_players = max(1, metadata.get("min_players", 1))
            try:
                obs = env.reset(num_players=num_players)
            except TypeError:
                obs = env.reset()

            # Try to obtain observations either from reset or state
            has_obs = False
            if obs is not None:
                has_obs = True
            elif hasattr(env, "state"):
                st = env.state
                if hasattr(st, "get_current_player_observation"):
                    o = st.get_current_player_observation()
                    has_obs = o is not None
                elif hasattr(st, "observations"):
                    has_obs = bool(st.observations)

            # Try a very simple step if we have any observation
            if has_obs:
                try:
                    result = env.step("1")
                    # Accept various return formats; any non-crashing response is OK
                    _ = result
                except Exception:
                    # Step might fail for some games without proper action format; still accept
                    pass
            try:
                env.close()
            except Exception:
                pass
            return has_obs
        except Exception as e:
            logger.debug(f"Quick validation failed for {env_id}: {e}")
            return False

    def _probe_error_for_env(self, env_id: str) -> bool:
        """Attempt a basic make/reset to detect import/initialization errors.

        Returns True if an error was detected and recorded.
        """
        try:
            meta = self.get_game_metadata(env_id)
            n = max(1, int(meta.get("min_players", 1)) or 1)
            env = ta.make(env_id)
            try:
                env.reset(num_players=n)
            except TypeError:
                env.reset()
            try:
                env.close()
            except Exception:
                pass
            return False
        except Exception as e:
            TextArenaGameRegistry._error_games.append(env_id)
            TextArenaGameRegistry._game_metadata_cache[env_id] = {
                "env_id": env_id,
                "min_players": 0,
                "max_players": 0,
                "game_type": "error",
                "name": env_id.split("-")[0] if "-" in env_id else env_id,
                "error": str(e),
            }
            self._persist_cache_to_file()
            logger.debug(f"Error probe recorded for {env_id}: {e}")
            return True

    def probe_errors(
        self, only_unknown: bool = True, only_ids: Optional[List[str]] = None
    ) -> int:
        """Probe environments to mark those that raise on initialization as 'error'.

        Args:
            only_unknown: if True, restrict to envs currently classified as unknown
            only_ids: optional explicit subset of env_ids to probe

        Returns: number of envs newly marked as error
        """
        if not TextArenaGameRegistry._discovered:
            self.discover_games()
        targets: List[str]
        if only_ids:
            targets = [eid for eid in only_ids if eid in ENV_REGISTRY]
        elif only_unknown:
            targets = list(TextArenaGameRegistry._unknown_games)
        else:
            # All non-error envs
            known = set(TextArenaGameRegistry._error_games)
            targets = [eid for eid in ENV_REGISTRY.keys() if eid not in known]
        marked = 0
        for env_id in targets:
            if self._probe_error_for_env(env_id):
                marked += 1
        # Rebuild classifications after marking
        TextArenaGameRegistry._single_player_games.clear()
        TextArenaGameRegistry._two_player_games.clear()
        TextArenaGameRegistry._multi_player_games.clear()
        TextArenaGameRegistry._unknown_games.clear()
        for env_id, meta in TextArenaGameRegistry._game_metadata_cache.items():
            if meta.get("game_type") == "error":
                continue
            max_players = int(meta.get("max_players", 0))
            if max_players == 0:
                TextArenaGameRegistry._unknown_games.append(env_id)
            elif max_players == 1:
                TextArenaGameRegistry._single_player_games.append(env_id)
            elif max_players == 2:
                TextArenaGameRegistry._two_player_games.append(env_id)
            else:
                TextArenaGameRegistry._multi_player_games.append(env_id)
        self._persist_cache_to_file()
        return marked


registry = None


def create_textarena_registry(
    seed: Optional[int] = None,
    max_player_detection: int = 4,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
    reset_cache: bool = False,
) -> TextArenaGameRegistry:
    """Create a TextArena game registry.

    Args:
        seed: Random seed for reproducibility
        max_player_detection: Maximum number of players to test during detection (default 4)
        cache_path: Optional path to JSON cache file (default is near this module)
        use_cache: When True, load and write to cache file to avoid expensive detection
        reset_cache: When True, delete existing cache before discovery and rebuild

    Returns:
        TextArenaGameRegistry instance
    """
    global registry
    if registry is None:
        registry = TextArenaGameRegistry(
            seed=seed,
            max_player_detection=max_player_detection,
            cache_path=cache_path,
            use_cache=use_cache,
            reset_cache=reset_cache,
        )
    return registry


def _cli_main(argv: Optional[List[str]] = None) -> int:
    """Simple CLI to build/update the TextArena registry cache.

    Usage:
      python textarena_registry.py [--reset] [--no-cache] [--cache-path PATH] [--max-detect N]
    """
    parser = argparse.ArgumentParser(
        description="Build/refresh TextArena registry cache"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete and rebuild cache from scratch"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing cache (forces in-memory detection only)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cache JSON (defaults next to this module)",
    )
    parser.add_argument(
        "--max-detect",
        type=int,
        default=4,
        help="Maximum players to try during detection for each env",
    )
    parser.add_argument(
        "--probe-errors",
        action="store_true",
        help="After discovery, probe envs to mark import/init failures as error",
    )
    parser.add_argument(
        "--probe-only-unknown",
        action="store_true",
        help="When probing errors, restrict to currently unknown envs (default)",
    )
    parser.add_argument(
        "--probe-ids",
        type=str,
        default=None,
        help="Comma-separated list of env ids to probe for errors",
    )
    args = parser.parse_args(argv)

    reg = create_textarena_registry(
        seed=None,
        max_player_detection=args.max_detect,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
        reset_cache=args.reset,
    )
    reg.discover_games()
    if args.probe_errors:
        ids = (
            [s.strip() for s in args.probe_ids.split(",") if s.strip()]
            if args.probe_ids
            else None
        )
        marked = reg.probe_errors(only_unknown=args.probe_only_unknown, only_ids=ids)
        logger.info(f"Probed errors; newly marked: {marked}")
    # Ensure final cache persisted
    # _persist_cache_to_file is called at the end of discover, but call again defensively
    try:
        reg._persist_cache_to_file()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(_cli_main())
