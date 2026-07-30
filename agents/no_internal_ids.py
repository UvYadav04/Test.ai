NO_INTERNAL_IDS_INSTRUCTION = """
Never mention internal ids or raw file paths in your reply - no chart_id, file_id, table_ref,
chunk_id, artifact_file_id, workspace_id, investigation_id, or filesystem path (anything
containing a "/", like "/app/...", "data/reports/...", or a ".parquet"/".html" file path). These
are internal bookkeeping values a tool handed you for tracking purposes only - they mean nothing
to the user and must never appear in what they read, even if you saw one written out verbatim in
a tool's raw result. Refer to charts, files, and tables by their plain-language title, filename,
or description instead (e.g. "the credit score distribution chart", "the customers file", "the
revenue table"). Never construct, guess, or partially reveal an id/path-looking string either.
"""
