#!/usr/bin/env python3

from typing import Any, Generator

from enum import Enum


import argparse
import logging
import sys
from pathlib import Path

from networkx.algorithms.coloring.greedy_coloring import strategy_connected_sequential
from clang.cindex import Cursor, CursorKind, Diagnostic, Index, Type, TypeKind
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Script directory for resolving default paths
SCRIPT_DIR = Path(__file__).resolve().parent

API_LIST = [
    "ncclCommQueryProperties",
    "ncclDevCommCreate",
    "ncclDevCommDestroy",
]

EXCLUDE_TYPES = [
    "size_t",
    "uint8_t",
    "uint32_t",
    "uint64_t",
]

class API_tree:
    def __init__(self, cursor: Cursor):
        self.api_cursor = cursor
        self.children_to_generate: list[Cursor] = []

    def add_child_to_generate(self, child: Cursor):
        self.children_to_generate.append(child)


def get_api_cursor(cursor: Cursor, api: str) -> Cursor | None:
    for descendant in cursor.walk_preorder():
        if descendant.spelling == api and descendant.kind == CursorKind.FUNCTION_DECL:
            return descendant

    return None


def get_parameters(cursor: Cursor) -> Generator[Any, Any, None]:
    if cursor.kind != CursorKind.FUNCTION_DECL:
        return

    for child in cursor.get_children():
        if child.kind == CursorKind.PARM_DECL:
            yield child


def peel_type(t: Type) -> Type:
    # remove ptr
    depth = 0
    while t.kind == TypeKind.POINTER:
        depth += 1
        t = t.get_pointee()
    return t, depth


class EdgeKind(Enum):
    BY_VALUE = "by_value"
    BY_POINTER = "by_pointer"
    BY_TYPEDEF = "by_typedef"


def add_edge(g: nx.DiGraph, a: Cursor, b: Cursor, edge_kind: EdgeKind) -> None:
    a_usr = a.get_usr()
    b_usr = b.get_usr()

    def _cursor_file(c: Cursor) -> str:
        if c.location.file:
            try:
                # Prefer shorter, stable paths in legend.
                repo_root = Path(__file__).resolve().parents[1]
                return str(Path(c.location.file.name).resolve().relative_to(repo_root))
            except Exception:
                return str(c.location.file.name)
        return "<unknown>"

    if a_usr not in g:
        g.add_node(a_usr, cursor=a, label=(a.spelling or a.displayname or a_usr), file=_cursor_file(a))

    if b_usr not in g:
        g.add_node(b_usr, cursor=b, label=(b.spelling or b.displayname or b_usr), file=_cursor_file(b))

    if g.has_edge(a_usr, b_usr):
        old_kind = g[a_usr][b_usr]["kind"]
        if old_kind == EdgeKind.BY_POINTER and edge_kind == EdgeKind.BY_VALUE:
            g[a_usr][b_usr]["kind"] = EdgeKind.BY_VALUE
        return

    g.add_edge(a_usr, b_usr, kind=edge_kind)


def walk_type_deps(
    g: nx.DiGraph,
    cursor: Cursor,
    nccl_device_include: Path,
    visited: set[str],
) -> None:
    # if the declaration is outside the NCCL device include directory, skip it
    if cursor.location.file and not Path(cursor.location.file.name).is_relative_to(nccl_device_include):
        return

    # Prefer the definition (forward decls have no fields/enumerators).
    if cursor.kind in (CursorKind.STRUCT_DECL, CursorKind.ENUM_DECL):
        cursor = cursor.get_definition() or cursor

    # De-duplicate traversal and break cycles by USR.
    usr = cursor.get_usr()
    if usr:
        if usr in visited:
            return
        visited.add(usr)

    if cursor.kind == CursorKind.TYPEDEF_DECL:
        underlying_type = peel_type(cursor.underlying_typedef_type)[0]
        underlying = underlying_type.get_declaration()
        logger.info(
            "typedef %s -> %s (%s) %s %s",
            cursor.spelling,
            underlying.spelling,
            underlying.kind,
            underlying_type.spelling,
            underlying_type.kind,
        )
        if underlying.kind != CursorKind.NO_DECL_FOUND and underlying_type.spelling not in EXCLUDE_TYPES:
            add_edge(g, cursor, underlying, EdgeKind.BY_TYPEDEF)
            walk_type_deps(g, underlying, nccl_device_include, visited)
    elif cursor.kind == CursorKind.ENUM_DECL:
        dependency = cursor.type.get_declaration()
        # Cursor object identity is not stable; use USR to avoid enum->self edges.
        if dependency.get_usr() and dependency.get_usr() == cursor.get_usr():
            return
        if dependency != cursor:
            logger.info(
                "enum %s -> %s (%s) %s %s",
                cursor.spelling,
                dependency.spelling,
                dependency.kind,
                dependency.type.spelling,
                dependency.type.kind,
            )
            if dependency.type.kind != TypeKind.INVALID:
                add_edge(g, cursor, dependency, EdgeKind.BY_VALUE)
    elif cursor.kind == CursorKind.STRUCT_DECL:
        for child in cursor.get_children():
            logger.info(
                "field %s (%s) %s %s",
                child.spelling,
                child.kind,
                child.type.spelling,
                child.type.kind,
            )
            if child.type == cursor.type:
                continue
            if child.kind == CursorKind.FIELD_DECL:
                if child.type.kind == TypeKind.POINTER:
                    dependency = peel_type(child.type)[0].get_declaration()
                    if dependency.type.kind != TypeKind.INVALID and dependency.type.spelling not in EXCLUDE_TYPES and dependency.type.get_canonical().get_declaration().get_usr() != cursor.get_usr():
                        add_edge(g, cursor, dependency, EdgeKind.BY_POINTER)
                        walk_type_deps(g, dependency, nccl_device_include, visited)
                elif child.type.kind == TypeKind.RECORD:
                    if child.type.kind != TypeKind.INVALID:
                        add_edge(g, cursor, child.type.get_declaration(), EdgeKind.BY_VALUE)
                        walk_type_deps(g, child.type.get_declaration(), nccl_device_include, visited)
                elif child.type.kind == TypeKind.ELABORATED:
                    dependency = child.type.get_named_type().get_declaration()
                    if dependency.type.kind != TypeKind.INVALID and dependency.type.spelling not in EXCLUDE_TYPES:
                        logger.info(
                            "elaborated %s -> %s (%s) %s %s",
                            child.spelling,
                            dependency.spelling,
                            dependency.kind,
                            dependency.type.spelling,
                            dependency.type.kind,
                        )
                        add_edge(g, cursor, dependency, EdgeKind.BY_VALUE)
                        walk_type_deps(g, dependency, nccl_device_include, visited)
                elif child.type.kind in (TypeKind.CONSTANTARRAY, TypeKind.INCOMPLETEARRAY, TypeKind.VARIABLEARRAY):
                    dependency = child.type.get_array_element_type().get_declaration()
                    if dependency.type.kind != TypeKind.INVALID and dependency.type.spelling not in EXCLUDE_TYPES:
                        add_edge(g, cursor, dependency, EdgeKind.BY_VALUE)
                        walk_type_deps(g, dependency, nccl_device_include, visited)
                else:
                    logger.warning("Ignoring field type: %s %s", child.spelling, child.type.kind)
    elif cursor.kind == CursorKind.NO_DECL_FOUND:
        logger.warning(
            "No declaration found for %s %s %s",
            cursor.spelling,
            cursor.type.spelling,
            cursor.type.kind,
        )
    else:
        logger.warning("Unsupported kind: %s:%s", cursor.spelling, cursor.kind)


def draw_graph(g: nx.DiGraph, output_file: Path):
    # Graphviz/pydot is strict about node IDs. clang USRs can contain characters
    # like ';' which break DOT syntax, so we relabel nodes to safe IDs for layout
    # and drawing, while preserving original labels.
    mapping = {n: f"n{i}" for i, n in enumerate(g.nodes())}
    g_safe = nx.relabel_nodes(g, mapping, copy=True)

    orig_labels = nx.get_node_attributes(g, "label")
    labels_safe = {mapping[orig]: orig_labels.get(orig, orig) for orig in g.nodes()}

    # Color nodes by the file they originate from.
    orig_files = nx.get_node_attributes(g, "file")
    file_values = [orig_files.get(orig, "<unknown>") for orig in g.nodes()]
    unique_files = sorted(set(file_values))

    # Pick a palette. tab20 is nice up to 20; beyond that, cycle HSV.
    tab20 = plt.get_cmap("tab20")
    hsv = plt.get_cmap("hsv")
    file2color: dict[str, Any] = {}
    for i, f in enumerate(unique_files):
        if len(unique_files) <= 20:
            file2color[f] = tab20(i % 20)
        else:
            file2color[f] = hsv(i / max(1, len(unique_files)))

    colors_safe = {mapping[orig]: file2color.get(orig_files.get(orig, "<unknown>"), (0.7, 0.7, 0.7, 1.0))
                   for orig in g.nodes()}

    pos = nx.nx_pydot.graphviz_layout(g_safe, prog="dot")
    # graphviz_layout may omit isolated nodes; fill missing positions.
    if len(pos) != g_safe.number_of_nodes():
        fallback = nx.spring_layout(g_safe, seed=0)
        for n in g_safe.nodes():
            pos.setdefault(n, fallback[n])

    plt.figure(figsize=(14, 10))
    nx.draw(g_safe, pos,
            labels=labels_safe,
            with_labels=True,
            node_color=[colors_safe[n] for n in g_safe.nodes()],
            node_size=1500,
            font_size=8,
            edge_color='gray',
            arrowsize=20,
            alpha=0.8
        )

    # Legend: show which file maps to which color.
    handles = [mpatches.Patch(color=file2color[f], label=f) for f in unique_files]
    plt.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=7,
        title="cursor.location.file",
        title_fontsize=8,
    )

    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Wrote graph to {output_file.resolve()}")


def normalize_type_for_pycparser(type_spelling: str, type_obj: Type = None) -> str:
    """Normalize type names for pycparser compatibility.

    pycparser doesn't handle bool and enum types well, so convert them to fixed-width integer types.
    This also preserves correct struct layout (bool is 1 byte, enum is typically 4 bytes or explicitly sized).

    Args:
        type_spelling: The string representation of the type
        type_obj: Optional Type object to check TypeKind
    """
    # Check TypeKind.BOOL first if we have the type object
    if type_obj and type_obj.kind == TypeKind.BOOL:
        return "uint8_t"

    # Strip leading/trailing whitespace
    type_spelling = type_spelling.strip()

    # Convert bool types to uint8_t
    if type_spelling in ("bool", "_Bool"):
        return "uint8_t"

    # Convert ncclGinType_t to uint8_t (it's typedef'd as uint8_t enum in C mode)
    if type_spelling == "ncclGinType_t":
        return "uint8_t"

    # Handle const bool, volatile bool, etc.
    parts = type_spelling.split()
    normalized_parts = []
    for part in parts:
        if part in ("bool", "_Bool"):
            normalized_parts.append("uint8_t")
        elif part == "ncclGinType_t":
            normalized_parts.append("uint8_t")
        else:
            normalized_parts.append(part)

    return " ".join(normalized_parts)


def emit_api(api_cursor: Cursor) -> str:
    params = ", ".join(f"{normalize_type_for_pycparser(param.type.spelling, param.type)} {param.spelling}" for param in get_parameters(api_cursor))
    return f"{normalize_type_for_pycparser(api_cursor.result_type.spelling, api_cursor.result_type)} {api_cursor.spelling}({params})"


def emit_struct(struct_cursor: Cursor) -> str:
    struct_def = struct_cursor.get_definition() or struct_cursor
    self_usr = struct_def.get_usr()
    tag = f"{struct_def.spelling}" if (struct_def.spelling and not struct_def.is_anonymous()) else ""

    def _is_ptr_to_self(t: Type) -> bool:
        """True if `t` is (or typedef-resolves to) pointer-to-this-struct.

        Handles:
        1) typedef struct a a_t; struct a { a_t *next; }
        2) typedef struct a* a_t; struct a { a_t next; }
        3) struct a { struct a *next; }
        """
        if not self_usr:
            return False
        try:
            ct = t.get_canonical()
        except Exception:
            ct = t
        if ct.kind != TypeKind.POINTER:
            return False
        try:
            pointee = ct.get_pointee()
            pointee_ct = pointee.get_canonical()
            decl = pointee_ct.get_declaration()
            return bool(decl) and decl.get_usr() == self_usr
        except Exception:
            return False

    children = list(struct_def.get_children())
    if len(children) == 0:
        return f"struct {tag}"

    lines = [f"struct {tag} {{"] if tag else [f"struct {{"]
    for child in children:
        if child.kind != CursorKind.FIELD_DECL:
            continue

        t = child.type

        # Linked-list/self-referential pointers (including via typedefs) -> void*
        if _is_ptr_to_self(t):
            lines.append(f"    void* {child.spelling};")
            continue

        # Arrays: emit as `elem name[N];` (not `elem[N] name;`)
        if t.kind in (TypeKind.CONSTANTARRAY, TypeKind.INCOMPLETEARRAY, TypeKind.VARIABLEARRAY):
            try:
                elem = t.get_array_element_type()
                n = t.get_array_size()  # -1 for incomplete/variable in some cases
            except Exception:
                elem = getattr(t, "element_type", t)
                n = getattr(t, "element_count", -1)

            elem_sp = "void*" if _is_ptr_to_self(elem) else normalize_type_for_pycparser(elem.spelling, elem)
            if isinstance(n, int) and n >= 0:
                lines.append(f"    {elem_sp} {child.spelling}[{n}];")
            else:
                lines.append(f"    {elem_sp} {child.spelling}[];")
            continue

        # Default
        lines.append(f"    {normalize_type_for_pycparser(t.spelling, t)} {child.spelling};")

    lines.append("}")
    return "\n".join(lines)


def emit_enum(enum_cursor: Cursor, has_tag: bool = True) -> str:
    tag = f"{enum_cursor.spelling} " if has_tag else ""
    lines = [f"enum {tag}{{"]
    for child in enum_cursor.get_children():
        if child.kind != CursorKind.ENUM_CONSTANT_DECL:
            continue
        lines.append(f"    {child.spelling} = {child.enum_value},")
    lines.append("}")
    return "\n".join(lines)


def emit_typedef(typedef_cursors: list[Cursor], out: list[str]) -> None:
    typedef_names = []
    for typedef_cursor in typedef_cursors:
        # ignore typedefs from nccl.h
        if Path(typedef_cursor.location.file.name).name == "nccl.h":
            continue
        _, depth = peel_type(typedef_cursor.underlying_typedef_type)
        typedef_names.append("*" * depth + f"{typedef_cursor.spelling}")

    if len(typedef_names) == 0:
        return

    underlying_type = peel_type(typedef_cursors[0].underlying_typedef_type)[0]
    underlying = underlying_type.get_declaration()

    logger.info(f"{','.join(typedef_names)} -> {underlying.spelling}, {underlying_type.spelling}, {underlying_type.kind.spelling}")

    if underlying.kind == CursorKind.STRUCT_DECL:
        str = emit_struct(underlying)
        str = f"typedef {str} {','.join(typedef_names)};\n"
        logger.info(str)
        out.append(str)
    elif underlying.kind == CursorKind.ENUM_DECL:
        str = emit_enum(underlying, typedef_cursors[0].spelling != underlying.spelling)
        str = f"typedef {str} {','.join(typedef_names)};\n"
        logger.info(str)
        out.append(str)
    elif underlying.kind == CursorKind.TYPEDEF_DECL:
        if underlying.type.spelling not in EXCLUDE_TYPES:
            emit_typedef([underlying], out)
        str = f"typedef {underlying.spelling} {','.join(typedef_names)};\n"
        logger.info(str)
        out.append(str)
    elif underlying.kind == CursorKind.NO_DECL_FOUND:
        str = f"typedef {normalize_type_for_pycparser(underlying_type.spelling, underlying_type)} {','.join(typedef_names)};\n"
        logger.info(str)
        out.append(str)
    else:
        logger.error("canonical type of %s is not a struct or enum: %s", typedef_cursors[0].spelling, underlying.kind)


def walk_graph(g: nx.DiGraph, node_id: str, out: list[str]) -> None:
    if g.nodes[node_id].get('used'):
        return

    for child_id in g.successors(node_id):
        walk_graph(g, child_id, out)

    if not g.nodes[node_id].get('used'):
        self_cursor = g.nodes[node_id].get('cursor')
        if not self_cursor:
            logger.error("Node %s has no cursor", node_id)
            return

        typedef_nodeids = [ancestor for ancestor, _, attr in g.in_edges(node_id, data=True) if attr['kind'] == EdgeKind.BY_TYPEDEF]
        if len(typedef_nodeids) > 0:
            emit_typedef([g.nodes[node_id]['cursor'] for node_id in typedef_nodeids], out)
            for n in typedef_nodeids:
                g.nodes[n]['used'] = True
        elif self_cursor.kind == CursorKind.TYPEDEF_DECL:
            emit_typedef([self_cursor], out)
        elif self_cursor.kind == CursorKind.STRUCT_DECL:
            out.append(emit_struct(self_cursor) + ";\n")
        elif self_cursor.kind == CursorKind.ENUM_DECL:
            out.append(emit_enum(self_cursor) + ";\n")
        else:
            logger.error("Node %s has unsupported kind: %s", node_id, self_cursor.kind)

        g.nodes[node_id]['used'] = True


def generate_header(g: nx.DiGraph, api_cursors: list[Cursor], output_file: Path) -> None:
    out = [
        "/*************************************************************************",
        " * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.",
        " *",
        " * See LICENSE.txt for license information",
        " ************************************************************************/",
        "",
        "/*",
        " * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY",
        " *",
        " * This header is automatically generated from NCCL Device API headers",
        " * for pycparser/Cybind compatibility. It provides C-compatible declarations",
        " * of Device API host-side functions and types.",
        " */",
        "",
        "#ifndef _NCCL_DEVICE_HOSTONLY_H_",
        "#define _NCCL_DEVICE_HOSTONLY_H_",
        "",
        "#include <nccl.h>",
        "#include <stdint.h>",
        "",
        "/* Force C mode - pycparser doesn't support C++ */",
        "#ifdef __cplusplus",
        "#undef __cplusplus",
        "#endif",
        "",
        "/* Note: C++ bool types are converted to uint8_t for pycparser compatibility",
        " * and to preserve correct struct layout (bool is 1 byte, not 4 bytes like",
        " * Cython's bint). */",
        "",
    ]
    for node_id in (n for n in g if g.in_degree(n) == 0):
        walk_graph(g, node_id, out)

    out.append("/* Function declarations */")
    out.append("")
    out.append("#ifdef __cplusplus")
    out.append("extern \"C\" {")
    out.append("#endif")
    out.append("")

    for api_cursor in api_cursors:
        out.append(emit_api(api_cursor) + ";")

    out.append("")
    out.append("#ifdef __cplusplus")
    out.append("}")
    out.append("#endif")
    out.append("")

    out.append("#endif /* _NCCL_DEVICE_HOSTONLY_H_ */")

    generated_header = "\n".join(out)

    # HACK: Fix broken typedef that references undefined type as Cybind doesn't support typedef chain.
    generated_header = generated_header.replace(
        "typedef ncclDevResourceHandle ncclDevResourceHandle_t;",
        "typedef uint32_t ncclDevResourceHandle_t;"
    )

    with open(output_file, "w") as f:
        f.write(generated_header)


def configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=lvl, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a header file from NCCL device API headers"
    )
    parser.add_argument(
        "--include", "-I",
        action="append",
        dest="includes",
        type=Path,
        default=[],
        help="Add directory to include search path (can be specified multiple times, e.g. -I dir1 -I dir2)"
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=SCRIPT_DIR.parent.parent.parent / "build/include/nccl_device.h",
        help="Path to the header file to parse (default: ../../build/include/nccl_device.h relative to script)",
    )
    parser.add_argument(
        "--nccl-device-include",
        type=Path,
        default=SCRIPT_DIR.parent.parent.parent / "build/include/nccl_device",
        help="Path to the NCCL device include directory, definitions in this directory will be included in the output (default: ../../build/include/nccl_device/ relative to script)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=SCRIPT_DIR.parent.parent.parent / "build/include/nccl_device_expanded.h",
        help="Output file path (default: ../../build/include/nccl_device_expanded.h relative to script)",
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )
    parser.add_argument(
        "--output-graph",
        type=Path,
        default=None,
        help="Generate and save a visualization of the type dependency graph to the specified path (e.g., graph.png)",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)

    if not args.header.exists():
        logger.error("Header file not found: %s", args.header)
        return 1

    if not args.output.parent.exists():
        logger.error("Output parentdirectory not found: %s", args.output.parent)
        return 1

    clang_args = [
        '-x', 'c-header', '-std=c99',
        '-D_NCCL_GIN_DEVICE_COMMON_H_',
        '-D_NCCL_DEVICE_GIN_SESSION__FUNCS_H_',
        '-D_NCCL_DEVICE_VECTOR__TYPES_H_',
        '-D_NCCL_DEVICE_REDUCE_COPY__TYPES_H_',
        '-D_NCCL_DEVICE_MULTIMEM__FUNCS_H_',
        '-D_NCCL_DEVICE_REDUCE_COPY__IMPL_H_',
        '-D_NCCL_DEVICE_BARRIER__FUNCS_H_',
        ] + [f'-I{inc}' for inc in args.includes]

    index = Index.create()

    tu = index.parse(str(args.header), args=clang_args)
    if not tu:
        logger.error("Failed to parse header: %s", args.header)
        return 1

    # Check for diagnostics
    had_error = False
    for diag in tu.diagnostics:
        if diag.severity >= Diagnostic.Error:
            had_error = True
            logger.error("Parse error: %s, %s:%s", diag.spelling, diag.location.file.name, diag.location.line)
        elif diag.severity >= Diagnostic.Warning:
            logger.warning("Warning: %s", diag.spelling)

    g = nx.DiGraph()
    visited: set[str] = set()
    api_cursors: list[Cursor] = []

    for api in API_LIST:
        api_cursor = get_api_cursor(tu.cursor, api)
        if api_cursor is None:
            logger.error("API not found: %s", api)
            continue
        api_cursors.append(api_cursor)

        for param in get_parameters(api_cursor):
            param_t = peel_type(param.type)[0]
            decl_c = param_t.get_declaration()

            walk_type_deps(g, decl_c, args.nccl_device_include, visited)

    # Optionally generate and save the type dependency graph
    if args.output_graph:
        draw_graph(g, args.output_graph)

    generate_header(g, api_cursors, args.output)

    return 0

if __name__ == "__main__":
    sys.exit(main())
