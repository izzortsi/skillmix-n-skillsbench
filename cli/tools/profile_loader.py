"""
profile_loader.py

Load and save PipelineProfile instances as YAML files.
Profiles are stored in cli/profiles/.
"""

from pathlib import Path

from config.pipeline_profile import PipelineProfile


PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


def _ensure_profiles_dir() -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR


def load_profile(name: str) -> PipelineProfile:
    """Load a named profile from YAML. Raises FileNotFoundError if missing."""
    import yaml

    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    profile = PipelineProfile()
    for key, value in data.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
    profile.profile_name = name
    return profile


def save_profile(profile: PipelineProfile) -> Path:
    """Save a profile to YAML. Returns the file path.

    If run_dir still has the default value, auto-set it to use the profile name.
    """
    import yaml

    _ensure_profiles_dir()
    path = PROFILES_DIR / f"{profile.profile_name}.yaml"

    # auto-set run_dir based on profile name when using default
    default_run_dir = PipelineProfile().run_dir
    if profile.run_dir == default_run_dir and profile.profile_name != "default":
        profile.run_dir = f"data/pipeline-runs/{profile.profile_name}"

    data = {}
    for key, value in profile.__dict__.items():
        data[key] = value

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return path


def list_profiles() -> list:
    """List all saved profile names."""
    _ensure_profiles_dir()
    profiles = []
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        profiles.append(path.stem)
    return profiles


def delete_profile(name: str) -> bool:
    """Delete a named profile. Returns True if deleted, False if not found."""
    path = PROFILES_DIR / f"{name}.yaml"
    if path.exists():
        path.unlink()
        return True
    return False
