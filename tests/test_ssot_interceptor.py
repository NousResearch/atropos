import pytest
import re
from atroposlib.sampling.ssot_wrapper import SSoTExplorationWrapper, ActionParserInterceptor

def test_interceptor_adversarial_noise():
    # wrapper = SSoTExplorationWrapper() # Not needed for this specific test
    adversarial_output = """
    <random_string>7fG2#kL9!pY4@zR6</random_string>
    <thinking>
    H = (ord('7') * 31^0 + ord('f') * 31^1 ...) mod 10^9+7
    "Unescaped quotes" and \\backslashes\\ that might break JSON
    Multiple lines of CoT math...
    Calculation error simulation: H = 123456789
    </thinking>
    <action>
    {"tool": "move_piece", "params": {"from": "e2", "to": "e4"}}
    </action>
    """
    
    parsed = ActionParserInterceptor.intercept_response(adversarial_output)
    
    # Assert isolation: Noise must be stripped, action must be preserved
    assert "<random_string>" not in parsed
    assert "<thinking>" not in parsed
    assert "Unescaped quotes" not in parsed
    assert "<action>" in parsed
    assert '"tool": "move_piece"' in parsed
    
    # Verify exact structure
    assert parsed.strip().startswith("<action>")
    assert parsed.strip().endswith("</action>")
    
    print("\n✓ Interceptor isolated the action from adversarial noise.")

if __name__ == "__main__":
    test_interceptor_adversarial_noise()
