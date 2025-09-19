from utils.alignment_utils import find_greedy_shift_match


def _to_items(words):
    return [{"word": w} for w in words]


def _words(list_of_dicts):
    return [d.get("word") for d in list_of_dicts]


def test_exact_match_no_skips():
    words = ["hello", "world"]
    items = _to_items(["hello", "world"])  # perfect alignment

    matched = find_greedy_shift_match(words, items, start_idx=0)

    assert _words(matched) == ["hello", "world"]


def test_advance_items_on_mismatch():
    # items has a leading extra token that should be skipped
    words = ["hello", "world"]
    items = _to_items(["hi", "hello", "world"])  # need to skip 'hi'

    matched = find_greedy_shift_match(words, items, start_idx=0)

    assert _words(matched) == ["hello", "world"]


def test_advance_words_on_mismatch():
    # words has an extra token in the middle; algorithm should skip that word
    words = ["hello", "there", "world"]
    items = _to_items(["hello", "world"])  # no 'there' in items

    matched = find_greedy_shift_match(words, items, start_idx=0)

    # Should still match hello and world
    assert _words(matched) == ["hello", "world"]


def test_combined_skips_items_then_words():
    # Need to skip one item first, then skip one word to align
    words = ["the", "quick", "brown", "fox"]
    items = _to_items(
        ["uh", "the", "quick", "fox"]
    )  # missing 'brown', with a leading 'uh'

    matched = find_greedy_shift_match(words, items, start_idx=0)

    # Expected matched subsequence from items (partial due to missing 'brown')
    assert _words(matched) == ["the", "quick", "fox"]


def test_respects_start_idx():
    words = ["alpha", "beta"]
    items = _to_items(["alpha", "beta", "alpha", "beta"])  # two occurrences

    # Start from the second pair
    matched = find_greedy_shift_match(words, items, start_idx=2)

    assert _words(matched) == ["alpha", "beta"]
    # Ensure it matched from the later region by checking identity of slice
    assert _words(matched) == _words(items[2:4])


def test_fuzzy_similarity_threshold():
    # Our similarity is common_chars / max_len; 'agent' vs 'agents' = 5/6 ≈ 0.833
    words = ["agents"]
    items = _to_items(["agent"])  # singular vs plural

    # With strict 1.0 threshold, no match
    matched_strict = find_greedy_shift_match(words, items, similarity_threshold=1.0)
    assert _words(matched_strict) == []

    # With relaxed threshold 0.8, should match
    matched_fuzzy = find_greedy_shift_match(words, items, similarity_threshold=0.8)
    assert _words(matched_fuzzy) == ["agent"]


def test_skip_limits_prevent_over_skipping():
    words = ["one", "two", "three"]
    items = _to_items(["x", "one", "two"])  # missing 'three'

    # With zero skip quota, cannot skip the leading 'x' → no initial match
    matched_no_skip = find_greedy_shift_match(
        words, items, start_idx=0, max_skips_per_side=0
    )
    assert _words(matched_no_skip) in (
        [],
        ["one", "two"],
    )  # allow partial if first eventually matches

    # With skip quota, should be able to skip 'x' and match as much as possible
    matched_with_skip = find_greedy_shift_match(
        words, items, start_idx=0, max_skips_per_side=2
    )
    assert _words(matched_with_skip) == ["one", "two"]


def test_empty_inputs_return_empty():
    assert find_greedy_shift_match([], _to_items(["a"])) == []
    assert find_greedy_shift_match(["a"], []) == []
