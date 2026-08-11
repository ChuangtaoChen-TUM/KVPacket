---
name: repository-navigation
description: Answer exact questions about a repository using bounded read-only inspection tools.
---

<kvpacket-skill id="skill.repository-navigation">
# Repository navigation

Answer questions using only evidence obtained from the checked-out repository. Do not
rely on remembered versions of a project: the task uses a specific immutable commit.

Start by identifying a distinctive symbol from the question. Use `search_text` when
the defining file is unknown. Search for syntactic forms such as `def name`,
`class Name`, `NAME =`, `raise ErrorName`, or a call like `name(`. A search result is
only a location hint; inspect the surrounding implementation with `read_file` before
answering semantic questions.

Use `list_files` when a search returns no result or when package layout is unclear.
Restrict later operations to the most plausible source directory. Do not scan generated
artifacts or infer that a same-named symbol in a different file is the requested one.

For caller questions, inspect the enclosing function or method around the call site.
For defaults and base classes, read the complete declaration, including declarations
continued across lines. For exception questions, distinguish an explicit `raise` from
an exception merely caught, imported, documented, or passed as a value.

The requested final format is part of the task. Once the evidence is sufficient, stop
calling tools and return only the requested path, symbol, exception name, base list, or
Python literal. Do not add an explanation, Markdown fence, citation, or sentence.
</kvpacket-skill>
