#!/usr/bin/env bash
# .claude-hooks-config.sh - Project-specific configuration for Claude hooks
# Place this file in your project root to customize hook behavior

# Testing configuration
export CLAUDE_HOOKS_TEST_MODES="focused"
export CLAUDE_HOOKS_REQUIRE_TESTS="false"

# Uncomment to enable debug output
# export CLAUDE_HOOKS_DEBUG="1"

# Example: Only run focused tests (faster for large projects)
# export CLAUDE_HOOKS_TEST_MODES="focused"

# Example: Require all modules to have tests
# export CLAUDE_HOOKS_REQUIRE_TESTS="true"

# Example: Skip package tests, only run focused tests
# export CLAUDE_HOOKS_TEST_MODES="focused"

# Example: Only run package tests (no focused tests)
# export CLAUDE_HOOKS_TEST_MODES="package"
