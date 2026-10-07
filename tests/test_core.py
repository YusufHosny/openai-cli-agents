# ABOUTME: Unit tests for the shared primitives: transcript flattening and JSON extraction from
# ABOUTME: free-form model output.
from openai_cli_agents.core import Turn, extract_json, flatten_turns


def test_flatten_prefixes_roles_only_for_multi_turn():
  assert flatten_turns([Turn("system", "s"), Turn("user", "u")]) == ("s", "u")
  assert flatten_turns([Turn("user", "u")]) == ("", "u")
  system, conv = flatten_turns([Turn("system", "s"), Turn("user", "u"), Turn("assistant", "a")])
  assert system == "s" and conv == "USER: u\n\nASSISTANT: a"


def test_flatten_joins_multiple_system_turns():
  system, _ = flatten_turns([Turn("system", "a"), Turn("system", "b"), Turn("user", "u")])
  assert system == "a\n\nb"


def test_extract_json_handles_fences_and_prose():
  assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
  assert extract_json('sure! {"a": "}{", "b": [1]} done') == {"a": "}{", "b": [1]}
  assert extract_json("no json here") is None
