"""Deprecated - unused.

This used to make a SEPARATE LLM call after the main answer to generate follow-up question
suggestions. That's been replaced with agents/final_answer.py: the responding agent's own final-
answer system prompt now asks for 2 follow-up questions as part of the SAME reply (see
FOLLOW_UP_INSTRUCTION / split_follow_up_questions there), so no second round trip is needed.
Nothing imports this module anymore - kept only because this environment's tools can't delete a
file outright. Safe to delete this file entirely next time this directory is touched.
"""
