"""Shared test fixtures, owned here rather than in whichever test file was first.

Tests already imported each other for fixtures — ``import
tests.test_b300_arena_provider as provider_fixtures`` — which makes an arbitrary
test file the authority for a shape five others need, and means deleting or
renaming a test breaks unrelated suites. The builders live here instead, so a
fixture has an owner that is not also a test.
"""
