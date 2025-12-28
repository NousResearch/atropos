"""
Expression Edit Distance (EED) Score Module for PHYBench Evaluation.

This module implements the EED Score metric for evaluating mathematical expressions,
as described in the PHYBench paper (https://arxiv.org/abs/2504.16074).

The EED Score measures similarity between model-generated and reference expressions
by computing tree edit distance over their SymPy expression trees. It provides:
- Continuous scoring (0-100) that captures partial correctness
- 204% improved sample efficiency over binary scoring
- Robust handling of equivalent mathematical forms

Key components:
- LaTeX preprocessing and normalization
- SymPy expression tree construction
- Extended Zhang-Shasha tree edit distance algorithm
- Configurable scoring with subtree discount

Dependencies:
- sympy: Symbolic mathematics
- latex2sympy2_extended: LaTeX to SymPy conversion
- numpy: Numerical operations

Based on the official PHYBench implementation:
https://github.com/phybench-official/phybench/tree/main/EED
"""

import re
from typing import List, Optional, Tuple

from numpy import ones, zeros

# Try to import required dependencies
try:
    from latex2sympy2_extended import latex2sympy
    from sympy import (
        Add,
        Float,
        Function,
        Integer,
        Mul,
        Pow,
        Rational,
        Symbol,
        expand,
        posify,
        simplify,
    )
    from sympy.core.numbers import Exp1, Infinity, NegativeInfinity, Pi

    EED_AVAILABLE = True
except ImportError:
    EED_AVAILABLE = False
    latex2sympy = None


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Cost configuration for tree edit operations
# These can be modified if different node types should have different weights
INSERT_COST = {"number": 1, "symbol": 1, "operator": 1, "function": 1}
DELETE_COST = {"number": 1, "symbol": 1, "operator": 1, "function": 1}
UPDATE_COST = {"number": 1, "symbol": 1, "operator": 1, "function": 1}

# Cost of updating between different types (e.g., number -> symbol)
CHANGE_TYPE_COST = 1

# Subtree discount configuration
# Minimum size to trigger cluster discount for subtree operations
BAR_SIZE = 5
# Discount slope for subtree operations (0.6 means 40% discount)
DISCOUNT_SLOPE = 0.6

# Timeout limits (in seconds)
SIMPLIFY_TIME_LIMIT = 30
EQUALS_TIME_LIMIT = 10


# =============================================================================
# TREE NODE CLASS
# =============================================================================


class TreeNode:
    """
    A node in the expression tree representation.

    Attributes:
        label: Node label (e.g., "number_2", "symbol_x", "operator_Add")
        children: List of child TreeNode objects
        subtree_size: Cached size of the subtree rooted at this node
    """

    def __init__(self, label: str, children: Optional[List["TreeNode"]] = None):
        self.label = label
        self.children = children if children is not None else []
        self.subtree_size = 0

    def get_children(self) -> List["TreeNode"]:
        """Return the list of child nodes."""
        return self.children

    def __str__(self) -> str:
        return self.label


# =============================================================================
# TREE EDIT DISTANCE FUNCTIONS
# =============================================================================


def calc_tree_size(node: TreeNode) -> int:
    """
    Calculate the size of a subtree based on total insertion cost.

    The size equals the sum of insertion costs of all nodes in the subtree.
    Results are cached in node.subtree_size for efficiency.

    Args:
        node: Root node of the subtree

    Returns:
        Total size of the subtree
    """
    # Get insertion cost for this node type
    node_type = node.label.split("_")[0]
    total = INSERT_COST.get(node_type, 1)

    # Return cached value if available (for non-leaf nodes)
    if node.children and node.subtree_size != 0:
        return node.subtree_size

    # Recursively calculate size of children
    for child in node.children:
        total += calc_tree_size(child)

    # Cache the result
    node.subtree_size = total
    return total


def update_func(x: TreeNode, y: TreeNode) -> float:
    """
    Calculate the cost of updating node x to node y.

    Args:
        x: Source node
        y: Target node

    Returns:
        Update cost (0 if identical, type-specific cost if same type, else CHANGE_TYPE_COST)
    """
    if x.label == y.label:
        return 0

    x_type = x.label.split("_")[0]
    y_type = y.label.split("_")[0]

    if x_type == y_type:
        return UPDATE_COST.get(x_type, 1)

    return CHANGE_TYPE_COST


def remove_func(x: TreeNode) -> float:
    """Calculate the cost of removing a single node."""
    node_type = x.label.split("_")[0]
    return DELETE_COST.get(node_type, 1)


def remove_tree_func(x: TreeNode) -> float:
    """
    Calculate the cost of removing an entire subtree.

    Applies discount for large subtrees (cluster discount).

    Args:
        x: Root of subtree to remove

    Returns:
        Removal cost with potential discount
    """
    if not x.children:
        return remove_func(x)

    size = calc_tree_size(x)
    # Apply discount for large subtrees
    return min(size, DISCOUNT_SLOPE * (size - BAR_SIZE) + BAR_SIZE)


def insert_func(x: TreeNode) -> float:
    """Calculate the cost of inserting a single node."""
    node_type = x.label.split("_")[0]
    return INSERT_COST.get(node_type, 1)


def insert_tree_func(x: TreeNode) -> float:
    """Calculate the cost of inserting an entire subtree (same as removal)."""
    return remove_tree_func(x)


# =============================================================================
# ANNOTATED TREE FOR ZHANG-SHASHA ALGORITHM
# =============================================================================


class AnnotatedTree:
    """
    Annotated tree structure for the Zhang-Shasha algorithm.

    Computes post-order enumeration, left-most descendants, and keyroots
    needed for efficient tree edit distance calculation.
    """

    def __init__(self, root: TreeNode, get_children):
        self.get_children = get_children
        self.root = root
        self.nodes = []  # Post-order enumeration of nodes
        self.ids = []  # Matching list of IDs
        self.lmds = []  # Left-most descendants
        self.keyroots = None

        # Build the annotated structure
        import collections

        stack = [(root, collections.deque())]
        pstack = []
        j = 0

        while stack:
            n, anc = stack.pop()
            nid = j
            for c in self.get_children(n):
                a = collections.deque(anc)
                a.appendleft(nid)
                stack.append((c, a))
            pstack.append(((n, nid), anc))
            j += 1

        lmds = {}
        keyroots = {}
        i = 0

        while pstack:
            (n, nid), anc = pstack.pop()
            self.nodes.append(n)
            self.ids.append(nid)

            if not self.get_children(n):
                lmd = i
                for a in anc:
                    if a not in lmds:
                        lmds[a] = i
                    else:
                        break
            else:
                lmd = lmds.get(nid, i)

            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1

        self.keyroots = sorted(keyroots.values())


def ext_distance(
    a_root: TreeNode,
    b_root: TreeNode,
    get_children,
    single_insert_cost,
    insert_cost,
    single_remove_cost,
    remove_cost,
    update_cost_func,
) -> float:
    """
    Compute extended tree edit distance using modified Zhang-Shasha algorithm.

    This implementation extends the standard algorithm with subtree insertion
    and deletion operations for handling clustered changes.

    Args:
        a_root: Root of first tree
        b_root: Root of second tree
        get_children: Function to get children of a node
        single_insert_cost: Cost function for single node insertion
        insert_cost: Cost function for subtree insertion
        single_remove_cost: Cost function for single node removal
        remove_cost: Cost function for subtree removal
        update_cost_func: Cost function for updating a node

    Returns:
        Tree edit distance between the two trees
    """
    a_tree = AnnotatedTree(a_root, get_children)
    b_tree = AnnotatedTree(b_root, get_children)

    size_a = len(a_tree.nodes)
    size_b = len(b_tree.nodes)

    treedists = zeros((size_a, size_b), float)
    fd = 1000 * ones((size_a + 1, size_b + 1), float)

    def treedist(x: int, y: int):
        al = a_tree.lmds
        bl = b_tree.lmds
        an = a_tree.nodes
        bn = b_tree.nodes

        fd[al[x]][bl[y]] = 0

        for i in range(al[x], x + 1):
            node = an[i]
            fd[i + 1][bl[y]] = fd[al[i]][bl[y]] + remove_cost(node)

        for j in range(bl[y], y + 1):
            node = bn[j]
            fd[al[x]][j + 1] = fd[al[x]][bl[j]] + insert_cost(node)

        for i in range(al[x], x + 1):
            for j in range(bl[y], y + 1):
                node1 = an[i]
                node2 = bn[j]

                costs = [
                    fd[i][j + 1] + single_remove_cost(node1),
                    fd[i + 1][j] + single_insert_cost(node2),
                    fd[al[i]][j + 1] + remove_cost(node1),
                    fd[i + 1][bl[j]] + insert_cost(node2),
                ]
                m = min(costs)

                if al[x] == al[i] and bl[y] == bl[j]:
                    treedists[i][j] = min(m, fd[i][j] + update_cost_func(node1, node2))
                    fd[i + 1][j + 1] = treedists[i][j]
                else:
                    fd[i + 1][j + 1] = min(m, fd[al[i]][bl[j]] + treedists[i][j])

    for x in a_tree.keyroots:
        for y in b_tree.keyroots:
            treedist(x, y)

    return treedists[-1][-1]


# =============================================================================
# LATEX PREPROCESSING
# =============================================================================


def brackets_balanced(s: str) -> bool:
    """Check if brackets in a string are balanced."""
    stack = []
    bracket_pairs = {")": "(", "]": "[", "}": "{"}

    for char in s:
        if char in bracket_pairs.values():
            stack.append(char)
        elif char in bracket_pairs:
            if not stack or stack[-1] != bracket_pairs[char]:
                return False
            stack.pop()

    return len(stack) == 0


def find_first_unescaped_brace(s: str) -> int:
    """Find the position of the first unescaped opening brace."""
    escaped = False
    for i, c in enumerate(s):
        if c == "\\" and not escaped:
            escaped = True
            continue
        if c == "{" and not escaped:
            return i
        escaped = False
    return -1


def extract_bracket_content(s: str, bracket_position: int) -> Tuple[Optional[str], int]:
    """Extract content inside braces starting at given position."""
    brace_start = bracket_position + 1
    brace_depth = 0
    content = []
    escaped = False

    for i in range(brace_start, len(s)):
        char = s[i]
        if escaped:
            content.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            content.append(char)
            continue
        if char == "{":
            brace_depth += 1
            content.append(char)
        elif char == "}":
            if brace_depth == 0:
                return "".join(content), i
            brace_depth -= 1
            content.append(char)
        else:
            content.append(char)

    return None, -1


def remove_command(s: str, command: str, keep_inside: bool = False) -> str:
    """
    Remove all occurrences of a LaTeX command from a string.

    Args:
        s: Input string
        command: Command to remove (e.g., "\\textbf")
        keep_inside: If True, preserve content inside braces

    Returns:
        String with command removed
    """
    pos = s.find(command)
    if pos < 0:
        return s

    end_index = pos + len(command)
    level = 0

    if end_index < len(s) and s[end_index] == "{":
        while end_index < len(s):
            if s[end_index] == "{":
                level += 1
            elif s[end_index] == "}":
                level -= 1
                if level == 0:
                    break
            end_index += 1
        if keep_inside:
            s1 = s[:pos] + s[pos + len(command) + 1 : end_index] + s[end_index + 1 :]
        else:
            s1 = s[:pos] + s[end_index + 1 :]
    else:
        s1 = s[:pos] + s[end_index:]

    if command not in s1:
        return s1
    return remove_command(s1, command, keep_inside)


def extract_last_equal_content(s: str) -> str:
    """Extract content after the last equality/comparison operator."""
    comparison_operators = ("=", "\\approx", "\\ge", "\\le", "\\geq", "\\leq", "<", ">")
    content = s

    for sign in comparison_operators:
        if sign in s:
            rfind_index = s.rfind(sign)
            if rfind_index != -1:
                content = s[rfind_index + len(sign) :]

    return content.strip()


def convert_latex_fractions(latex_str: str) -> str:
    """Convert non-standard fractions like \\frac\\alpha2 to \\frac{\\alpha}{2}."""
    pattern = (
        r"\\frac((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))"
        r"((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))"
    )

    def replacer(match):
        numerator, denominator = match.group(1), match.group(2)
        wrap_num = (
            f"{{{numerator}}}"
            if not (numerator.startswith("{") and numerator.endswith("}"))
            else numerator
        )
        wrap_den = (
            f"{{{denominator}}}"
            if not (denominator.startswith("{") and denominator.endswith("}"))
            else denominator
        )
        return rf"\frac{wrap_num}{wrap_den}"

    return re.sub(pattern, replacer, latex_str)


def convert_vec_syntax(text: str) -> str:
    """Convert \\vec x to \\vec{x}."""
    pattern = r"\\vec(\s*)(\\?[a-zA-Zα-ωΑ-Ω]+)"
    replacement = r"\\vec{\2}"
    return re.sub(pattern, replacement, text)


def first_preprocess(s: str, extract_box: bool = True) -> str:
    """
    First stage of LaTeX preprocessing.

    Extracts boxed content, removes outer braces, and extracts content after equals.

    Args:
        s: Input LaTeX string
        extract_box: Whether to extract content from \\boxed{}

    Returns:
        Preprocessed string
    """
    s = s.replace("\\{", "(")
    s = s.replace("\\}", ")")

    if not brackets_balanced(s):
        return s

    if extract_box:
        boxed_content = remove_command(s, "\\boxed", keep_inside=True)
    else:
        boxed_content = s

    # Remove overall braces
    def remove_overall_brace(text: str) -> Tuple[str, bool]:
        pos = find_first_unescaped_brace(text)
        if pos == -1:
            return text, False

        content, final = extract_bracket_content(text, pos)
        if content and (final == len(text) - 1 or "}" not in text[final + 1 :]):
            # Check if there's a command before the brace
            if pos > 0 and text[pos - 1] not in (" ", "\t", "\n"):
                return text, False
            return content, True
        return text, False

    # Remove outer braces iteratively
    for _ in range(10):
        boxed_content, changed = remove_overall_brace(boxed_content)
        if not changed:
            break

    # Handle \\quad separator
    if "\\quad" in boxed_content:
        boxed_content = boxed_content.split("\\quad")[0]

    # Extract content after last equals sign
    last_equal_content = extract_last_equal_content(boxed_content)

    # Remove outer braces again
    for _ in range(10):
        last_equal_content, changed = remove_overall_brace(last_equal_content)
        if not changed:
            break

    return last_equal_content


def second_preprocess(s: str) -> str:
    """
    Second stage of LaTeX preprocessing.

    Removes/modifies LaTeX commands and normalizes expressions.

    Args:
        s: Input string from first preprocessing stage

    Returns:
        Normalized LaTeX string ready for conversion
    """
    # Commands to completely remove (including their content)
    kill_commands = ["\\begin", "\\end"]

    # Commands to remove but keep their content
    remove_commands = [
        "\\text",
        "\\mathbf",
        "\\mathrm",
        "\\pmb",
        "\\hat",
        "\\overline",
        "\\boldsymbol",
    ]

    # Content to remove entirely
    remove_content = [
        "\\,",
        "$",
        ",",
        "`",
        "latex",
        "\\left",
        "\\right",
        "\\Bigr",
        "\\Bigl",
        "\n",
        "\\]",
        "\\[",
        "\\Big",
        "\\bigl",
        "\\bigr",
        "\\biggl",
        "\\biggr",
        "\\displaystyle",
        "\\infty",
    ]

    # Content replacements
    replace_content = [
        ("\\operatorname{asin}", "\\asin"),
        ("\\operatorname{sech}", "\\sech"),
        ("\\operatorname{acos}", "\\acos"),
        ("\\operatorname{sinh}", "\\sinh"),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\Exp", "\\exp"),
        ("\\times", "\\bar{times}"),
        ("\\partial", "\\bar{partial}"),
        ("\\perp", "\\bar{perp}"),
        ("\\epsilon", "\\varepsilon"),
        ("\\varOmega", "\\Omega"),
        ("I", "\\bar{I}"),
        ("_e", "_{e}"),
        ("e_", "\\bar{e}_"),
        ("E_", "\\bar{E}_"),
        ("\\pm", "+"),
        ("\\mp", "-"),
        ("{+}", "{p}"),
        ("{-}", "{m}"),
        ("_+", "_p"),
        ("_-", "_m"),
    ]

    # Apply transformations
    for command in kill_commands:
        s = remove_command(s, command, keep_inside=False)

    for command in remove_commands:
        s = remove_command(s, command, keep_inside=True)

    for content in remove_content:
        s = s.replace(content, "")

    for old, new in replace_content:
        s = s.replace(old, new)

    # Additional transformations
    s = convert_latex_fractions(s)
    s = convert_vec_syntax(s)

    # Remove trailing period
    if s and s[-1] == ".":
        s = s[:-1]

    return s


class LaTeXNormalizationConfig:
    """Configuration for latex2sympy normalization."""

    basic_latex: bool = True
    units: bool = False
    malformed_operators: bool = True
    nits: bool = True
    boxed = "all"
    equations: bool = False


class LaTeXConversionConfig:
    """Configuration for latex2sympy conversion."""

    interpret_as_mixed_fractions: bool = False
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False


def master_convert(s: str):
    """
    Convert a LaTeX string to a SymPy expression.

    This is the main conversion function that applies preprocessing
    and uses latex2sympy for the actual conversion.

    Args:
        s: LaTeX string to convert

    Returns:
        SymPy expression

    Raises:
        Various exceptions if conversion fails
    """
    if not EED_AVAILABLE:
        raise ImportError(
            "latex2sympy2_extended and sympy are required for EED scoring"
        )

    preprocessed_stage1 = first_preprocess(s)
    preprocessed_stage2 = second_preprocess(preprocessed_stage1)

    sym = latex2sympy(
        preprocessed_stage2,
        normalization_config=LaTeXNormalizationConfig(),
        conversion_config=LaTeXConversionConfig(),
    )
    return sym


# =============================================================================
# SYMPY TO TREE CONVERSION
# =============================================================================


def sympy_to_tree(expr) -> TreeNode:
    """
    Convert a SymPy expression to a tree structure.

    Args:
        expr: SymPy expression

    Returns:
        TreeNode representing the expression

    Raises:
        ValueError: If expression contains unsupported types
    """
    # Numbers and constants
    if isinstance(
        expr, (Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity)
    ):
        return TreeNode(label=f"number_{expr}", children=[])

    # Symbols
    if isinstance(expr, Symbol):
        return TreeNode(label=f"symbol_{expr}", children=[])

    # Binary operators (Add, Mul, Pow)
    if isinstance(expr, (Add, Mul, Pow)):
        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label=f"operator_{op_name}", children=children)

    # Functions
    if isinstance(expr, Function):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label=f"function_{func_name}", children=children)

    raise ValueError(f"Unsupported SymPy type: {type(expr).__name__}")


# =============================================================================
# SCORING FUNCTION
# =============================================================================


def score_calc(tree_dist: float, tree_size: int) -> float:
    """
    Calculate EED score from tree distance and size.

    The scoring function:
    - 100 if distance is 0 (exact match)
    - 60 - 100*r if 0 < r < 0.6 (partial credit)
    - 0 if r >= 0.6 (too different)

    where r = distance / tree_size

    Args:
        tree_dist: Tree edit distance
        tree_size: Size of the ground truth tree

    Returns:
        Score between 0 and 100
    """
    if tree_dist == 0:
        return 100.0

    return max(0, 100 * DISCOUNT_SLOPE - 100 * tree_dist / tree_size)


def time_simplify(expr, timeout: int = SIMPLIFY_TIME_LIMIT):
    """
    Simplify expression with timeout protection.

    Args:
        expr: SymPy expression to simplify
        timeout: Timeout in seconds

    Returns:
        Simplified expression, or original if timeout/error
    """
    try:
        # Note: For production use, consider using multiprocessing for true timeout
        return simplify(expr)
    except Exception:
        return expr


def time_equal(expr1, expr2, timeout: int = EQUALS_TIME_LIMIT) -> bool:
    """
    Check expression equality with timeout protection.

    Args:
        expr1: First expression
        expr2: Second expression
        timeout: Timeout in seconds

    Returns:
        True if expressions are equal, False otherwise
    """
    try:
        return expr1.equals(expr2)
    except Exception:
        return False


# =============================================================================
# MAIN EED FUNCTION
# =============================================================================


def compute_eed_score(
    answer_latex: str,
    test_latex: str,
    debug_mode: bool = False,
) -> Tuple[float, float, int, float]:
    """
    Compute the EED (Expression Edit Distance) Score between two LaTeX expressions.

    This function evaluates the similarity between a ground truth answer and
    a model-generated answer by:
    1. Converting LaTeX to SymPy expressions
    2. Simplifying and checking for equivalence
    3. Building expression trees
    4. Computing tree edit distance
    5. Converting distance to a 0-100 score

    Args:
        answer_latex: Ground truth answer in LaTeX format
        test_latex: Model-generated answer in LaTeX format
        debug_mode: If True, raise exceptions instead of returning defaults

    Returns:
        Tuple of (score, relative_distance, tree_size, distance):
        - score: EED score from 0-100 (100 = perfect match)
        - relative_distance: distance / tree_size (-1 if error)
        - tree_size: Size of ground truth tree (-1 if error)
        - distance: Raw tree edit distance (-1 if error)
    """
    if not EED_AVAILABLE:
        if debug_mode:
            raise ImportError("EED scoring requires latex2sympy2_extended and sympy")
        return 0, -1, -1, -1

    # Handle empty or invalid input
    if not test_latex:
        return 0, -1, -1, -1

    # Skip unsupported expressions (integrals, sums)
    if "\\int" in test_latex or "\\int" in answer_latex:
        return 0, -1, -1, -1
    if "\\sum" in test_latex or "\\sum" in answer_latex:
        return 0, -1, -1, -1

    # Quick check for exact string match
    if answer_latex == test_latex:
        return 100, 0.0, -1, 0

    # Skip if test is much longer than answer (likely wrong)
    if len(test_latex) > 3 * len(answer_latex):
        return 0, -1, -1, -1

    # Convert LaTeX to SymPy
    try:
        answer_exp = master_convert(answer_latex)
        test_exp = master_convert(test_latex)
    except Exception as e:
        if debug_mode:
            raise ValueError(f"Failed to convert LaTeX: {e}")
        return 0, -1, -1, -1

    # Simplify and check equivalence
    try:
        # Assume all symbols are positive for simplification
        answer_exp, rep1 = posify(answer_exp)
        answer_exp = time_simplify(answer_exp)

        test_exp, rep2 = posify(test_exp)
        test_exp = time_simplify(test_exp)

        # Restore original symbols
        answer_exp = answer_exp.subs(rep1)
        test_exp = test_exp.subs(rep2)

        # Check for equivalence
        zero_exp = time_simplify(expand(answer_exp - test_exp))

        if answer_exp == test_exp or zero_exp == 0:
            return 100, 0.0, 0, 0

        if time_equal(answer_exp, test_exp):
            return 100, 0.0, 0, 0

    except Exception as e:
        if debug_mode:
            raise ValueError(f"Failed during simplification: {e}")
        return 0, -1, -1, -1

    # Build expression trees
    try:
        tree_answer = sympy_to_tree(answer_exp)
        tree_test = sympy_to_tree(test_exp)
    except Exception as e:
        if debug_mode:
            raise ValueError(f"Failed to build expression tree: {e}")
        return 0, -1, -1, -1

    # Compute tree edit distance
    try:
        distance = ext_distance(
            tree_test,
            tree_answer,
            get_children=lambda x: x.get_children(),
            single_insert_cost=insert_func,
            insert_cost=insert_tree_func,
            single_remove_cost=remove_func,
            remove_cost=remove_tree_func,
            update_cost_func=update_func,
        )
    except Exception as e:
        if debug_mode:
            raise ValueError(f"Failed to calculate distance: {e}")
        tree_size = calc_tree_size(tree_answer)
        return 0, -1, tree_size, -1

    # Calculate final score
    tree_size = calc_tree_size(tree_answer)
    rel_distance = distance / tree_size
    score = score_calc(distance, tree_size)

    return score, rel_distance, tree_size, distance


def extract_boxed_content(latex_str: str) -> Optional[str]:
    """
    Extract content from \\boxed{} in a LaTeX string.

    Args:
        latex_str: LaTeX string potentially containing \\boxed{}

    Returns:
        Content inside \\boxed{}, or None if not found
    """
    # Pattern to match \boxed{...} with nested braces
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    match = re.search(pattern, latex_str)
    if match:
        return match.group(1)
    return None


def extract_all_boxed(latex_str: str) -> List[str]:
    """
    Extract all \\boxed{} contents from a LaTeX string.

    Handles arbitrarily nested braces by counting brace depth.

    Args:
        latex_str: LaTeX string

    Returns:
        List of contents from all \\boxed{} occurrences
    """
    results = []
    i = 0
    boxed_pattern = "\\boxed{"

    while i < len(latex_str):
        # Find next \boxed{
        pos = latex_str.find(boxed_pattern, i)
        if pos == -1:
            break

        # Start after \boxed{
        start = pos + len(boxed_pattern)
        depth = 1
        j = start

        # Count braces to find matching closing brace
        while j < len(latex_str) and depth > 0:
            if latex_str[j] == '{':
                depth += 1
            elif latex_str[j] == '}':
                depth -= 1
            j += 1

        if depth == 0:
            # Extract content between braces
            content = latex_str[start:j-1].strip()
            results.append(content)

        i = j

    return results

