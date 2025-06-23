from typing import Any, Dict, List

# Assuming VisualCue structure is known or passed in a structured way
# from .synthetic_users import VisualCue # For type hinting if using VisualCue objects directly


def format_cue_params_for_csharp(cue_data: Dict[str, Any], class_name_prefix="") -> str:
    """Helper to format cue parameters into C# static fields."""
    lines = []
    for key, value in cue_data.items():
        if isinstance(value, float):
            lines.append(
                f"        public static readonly float {class_name_prefix}{key.capitalize()} = {value:.3f}f;"
            )
        elif isinstance(value, int):
            lines.append(
                f"        public static readonly int {class_name_prefix}{key.capitalize()} = {value};"
            )
        elif isinstance(
            value, str
        ):  # For parameters like animation_name if you add them
            lines.append(
                f'        public static readonly string {class_name_prefix}{key.capitalize()} = "{value}";'
            )
        # Add other type handlers as needed (e.g., for color structs if not using separate H,S,V)
    return "\n".join(lines)


def generate_unity_affordance_library(discoveries: Dict[str, Dict[str, Any]]) -> str:
    """
    Generates Unity C# code for a library of discovered affordance cues.

    Args:
        discoveries: A dictionary where keys are descriptive names for cues/patterns
                     (e.g., "UniversalOptimalBreathing", "YouthFastPulseHighGlow")
                     and values are dictionaries of the cue parameters for that discovery.
                     Example: {
                         "UniversalBreathingCue": {"pulse_hz": 2.3, "glow": 0.35, ...},
                         "YouthHighEnergyCue": {"pulse_hz": 4.1, "particle_density": 0.8, ...}
                     }
    Returns:
        A string containing the generated C# code for a Unity script.
    """

    cue_classes_code = ""
    apply_methods_code = ""
    enum_cases = ""
    switch_cases_apply = ""

    cue_enum_entries = []

    for i, (cue_name, cue_data) in enumerate(discoveries.items()):
        # Sanitize cue_name for C# class/enum name
        class_name_suffix = "".join(
            filter(str.isalnum, cue_name.title().replace("_", "").replace(" ", ""))
        )
        enum_entry_name = class_name_suffix
        cue_enum_entries.append(enum_entry_name)

        params_code = format_cue_params_for_csharp(
            cue_data, class_name_prefix=f"{class_name_suffix}_"
        )

        cue_classes_code += """
    public static class {class_name_suffix}Params {{
{params_code}
    }}
"""
        # Basic application logic - this would need to be much more sophisticated
        # and actually use the parameters to drive shader properties or particle systems.
        apply_methods_code += """
    private static IEnumerator Apply{class_name_suffix}Internal(GameObject target) {{
        var renderer = target.GetComponent<Renderer>();
        if (renderer == null) yield break;
        var material = renderer.material; // Creates instance

        // Example: Use discovered parameters
        // float currentGlow = {class_name_suffix}Params.{class_name_suffix}_Glow;
        // float currentPulseHz = {class_name_suffix}Params.{class_name_suffix}_PulseHz;
        // This is a simplified placeholder for actual cue application logic

        float time = 0;
        float duration = 5f; // Example duration
        float baseIntensity = material.HasProperty("_GlowIntensity") ? material.GetFloat("_GlowIntensity") : 0f;

        while (time < duration) {{
            // Example: Pulsing glow using the specific cue's parameters
            // This logic would need to be standardized or made more flexible
            if (material.HasProperty("_GlowIntensity") && {class_name_suffix}Params.{class_name_suffix}_AnimationType == 1) {{ // Pulse animation
                 float phase = Mathf.Sin({class_name_suffix}Params.{class_name_suffix}_PulseHz * time * Mathf.PI * 2);
                 material.SetFloat("_GlowIntensity", {class_name_suffix}Params.{class_name_suffix}_Glow * (0.5f + 0.5f * phase));
            }} else if (material.HasProperty("_GlowIntensity")) {{
                 material.SetFloat("_GlowIntensity", {class_name_suffix}Params.{class_name_suffix}_Glow);
            }}
            // Add logic for edge, particles, color based on {class_name_suffix}Params
            time += Time.deltaTime;
            yield return null;
        }}
        // Optionally reset to baseIntensity or original material state
        // if (material.HasProperty("_GlowIntensity")) material.SetFloat("_GlowIntensity", baseIntensity);
        yield return null;
    }}
"""
        switch_cases_apply += f"                case CueType.{enum_entry_name}: StartCoroutine(Apply{class_name_suffix}Internal(targetObject)); break;\n"

    if cue_enum_entries:
        enum_cases = "\n        ".join([f"{name}," for name in cue_enum_entries])
        cue_enum_definition = """
public enum CueType {{
        {enum_cases}
    }}
"""
    else:
        cue_enum_definition = "// No cues defined for CueType enum\n"

    csharp_code = """
// Auto-generated C# script for Unity VR Affordances
// Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

using UnityEngine;
using System.Collections; // Required for IEnumerator

namespace VRAffordanceEvolutionPack {{

    // Enum to select cue type in Inspector or code
    {cue_enum_definition}

    // Static classes holding the discovered parameters for each cue type
    namespace DiscoveredCueParams {{
{cue_classes_code}
    }}

    public class ApplyDiscoveredAffordance : MonoBehaviour {{
        public CueType cueToApply = CueType.{cue_enum_entries[0] if cue_enum_entries else "/* DefaultCue */"};
        public GameObject targetObject;

        void Start() {{
            if (targetObject == null) {{
                targetObject = this.gameObject;
            }}
            ApplySelectedCue(targetObject, cueToApply);
        }}

        public void ApplySelectedCue(GameObject obj, CueType cueType) {{
            // Stop any previous cues if necessary (complex to manage generally)
            StopAllCoroutines();

            switch (cueType) {{
{switch_cases_apply}
                default:
                    Debug.LogWarning($"Cue type {{cueType}} not handled.");
                    break;
            }}
        }}

        // Internal methods to apply each specific cue type
{apply_methods_code}
    }}
}}
"""
    return csharp_code


if __name__ == "__main__":
    # Example: Data structure you might get from your analysis
    mock_discoveries_for_unity = {
        "UniversalBreathingCue": {
            "glow": 0.35,
            "pulse_hz": 2.3,
            "edge": 0.42,
            "color_hue": 210.0,
            "color_saturation": 0.8,
            "color_value": 0.9,
            "particle_density": 0.1,
            "particle_speed": 0.5,
            "animation_type": 2,  # 2 for breathe
            "size_change_amplitude": 0.15,
            "blur_amount": 0.1,
        },
        "YouthHighEnergyCue": {
            "glow": 0.75,
            "pulse_hz": 4.1,
            "edge": 0.3,
            "color_hue": 60.0,
            "color_saturation": 0.9,
            "color_value": 1.0,
            "particle_density": 0.8,
            "particle_speed": 1.5,
            "animation_type": 1,  # 1 for pulse
            "size_change_amplitude": 0.05,
            "blur_amount": 0.05,
        },
        "SeniorClearContrastCue": {
            "glow": 0.2,
            "pulse_hz": 1.2,
            "edge": 0.9,
            "color_hue": 240.0,
            "color_saturation": 0.7,
            "color_value": 0.8,
            "particle_density": 0.0,
            "particle_speed": 0.2,
            "animation_type": 0,  # 0 for static
            "size_change_amplitude": 0.0,
            "blur_amount": 0.0,
        },
    }

    print("Testing Unity C# Code Generation...")
    unity_code = generate_unity_affordance_library(mock_discoveries_for_unity)

    print("\n--- Generated Unity C# Code ---")
    print(unity_code)

    # Save to file
    output_filename = "DiscoveredAffordances.cs"
    with open(output_filename, "w") as f:
        f.write(unity_code)
    print(f"\nUnity C# code saved to {output_filename}")
