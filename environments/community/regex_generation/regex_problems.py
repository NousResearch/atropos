"""
Hand-crafted regex problems with natural language descriptions,
positive/negative test cases, and difficulty ratings.
"""

PROBLEMS = [
    # --- Easy ---
    {
        "description": (
            "Match a string that contains only digits (0-9),"
            " one or more characters long."
        ),
        "positive": ["123", "0", "999999", "42", "007"],
        "negative": ["abc", "12a3", "", " 123", "12.3", "12 34"],
        "difficulty": "easy",
    },
    {
        "description": ("Match a string that starts with 'hello' (case-sensitive)."),
        "positive": ["hello", "hello world", "hellooo", "hello123"],
        "negative": ["Hello", "hi hello", "HELLO", "hell"],
        "difficulty": "easy",
    },
    {
        "description": "Match a string that ends with '.txt'.",
        "positive": ["file.txt", "my_doc.txt", ".txt", "a.txt"],
        "negative": ["file.csv", "txt", "file.txt.bak", "file.txts"],
        "difficulty": "easy",
    },
    {
        "description": (
            "Match a string consisting of exactly three lowercase letters."
        ),
        "positive": ["abc", "xyz", "foo", "bar"],
        "negative": ["ab", "abcd", "ABC", "a1c", "ab ", " ab"],
        "difficulty": "easy",
    },
    {
        "description": (
            "Match a string that is either 'yes' or 'no'"
            " (exact match, case-sensitive)."
        ),
        "positive": ["yes", "no"],
        "negative": ["Yes", "NO", "maybe", "yes ", " no", "yesno"],
        "difficulty": "easy",
    },
    {
        "description": ("Match a string that contains at least one uppercase letter."),
        "positive": ["Hello", "ABC", "aB", "123A456"],
        "negative": ["hello", "123", "abc!", ""],
        "difficulty": "easy",
    },
    {
        "description": (
            "Match a non-empty string consisting only of"
            " whitespace characters (spaces, tabs)."
        ),
        "positive": ["   ", " ", "\t", "  \t  "],
        "negative": ["", "a", " a ", "hello"],
        "difficulty": "easy",
    },
    {
        "description": "Match a string that starts with a digit.",
        "positive": ["1abc", "0", "9test", "3 things"],
        "negative": ["abc", " 1", "a1", ""],
        "difficulty": "easy",
    },
    # --- Medium ---
    {
        "description": (
            "Match a valid email address: one or more"
            " alphanumeric/dot/underscore/hyphen characters,"
            " then '@', then one or more alphanumeric/dot/hyphen"
            " characters, then '.', then two to four letters."
        ),
        "positive": [
            "user@example.com",
            "first.last@domain.org",
            "name_123@test.co",
            "a@b.io",
        ],
        "negative": [
            "@example.com",
            "user@.com",
            "user@com",
            "user@domain.toolongext",
            "user@@domain.com",
        ],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a time string in 24-hour format HH:MM"
            " where HH is 00-23 and MM is 00-59."
        ),
        "positive": ["00:00", "12:30", "23:59", "09:05"],
        "negative": [
            "24:00",
            "12:60",
            "1:30",
            "12:5",
            "12-30",
            "ab:cd",
        ],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a US zip code: exactly 5 digits, optionally"
            " followed by a dash and exactly 4 more digits."
        ),
        "positive": ["12345", "00000", "12345-6789", "99999-0000"],
        "negative": [
            "1234",
            "123456",
            "12345-678",
            "12345-67890",
            "abcde",
        ],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a hex color code: a '#' followed by exactly"
            " 6 hexadecimal characters (0-9, a-f, A-F)."
        ),
        "positive": ["#aabbcc", "#123456", "#ABCDEF", "#a1B2c3"],
        "negative": [
            "#abc",
            "#1234567",
            "aabbcc",
            "#GHIJKL",
            "# aabbcc",
        ],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a valid IPv4 address. Each octet is 0-255,"
            " separated by dots. No leading zeros allowed"
            " except for the number 0 itself."
        ),
        "positive": [
            "192.168.1.1",
            "0.0.0.0",
            "255.255.255.255",
            "10.0.0.1",
        ],
        "negative": [
            "256.1.1.1",
            "1.2.3.256",
            "01.02.03.04",
            "1.2.3",
            "1.2.3.4.5",
            "abc.def.ghi.jkl",
        ],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a date in the format YYYY-MM-DD where YYYY"
            " is four digits, MM is 01-12, and DD is 01-31."
        ),
        "positive": ["2024-01-15", "1999-12-31", "2000-06-01"],
        "negative": [
            "2024-13-01",
            "2024-00-15",
            "2024-01-32",
            "2024-01-00",
            "24-01-15",
            "2024/01/15",
        ],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a string of only alphanumeric characters and"
            " underscores. Must start with a letter or underscore"
            " and be 1-30 characters long. Like a variable name."
        ),
        "positive": ["my_var", "_private", "x", "CamelCase", "var_123"],
        "negative": ["123abc", "my-var", "my var", "", "a" * 31],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a string enclosed in double quotes. Inside the"
            " quotes, any characters are allowed except unescaped"
            ' double quotes. Escaped quotes (\\") are allowed.'
        ),
        "positive": [
            '"hello"',
            '"hello world"',
            '""',
            '"she said \\"hi\\""',
        ],
        "negative": [
            "hello",
            '"missing end',
            'no "quotes" here',
            "'single'",
        ],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a phone number in the format (XXX) XXX-XXXX" " where X is a digit."
        ),
        "positive": [
            "(123) 456-7890",
            "(000) 000-0000",
            "(999) 999-9999",
        ],
        "negative": [
            "123-456-7890",
            "(123)456-7890",
            "(123) 456 7890",
            "(12) 456-7890",
            "(1234) 456-7890",
        ],
        "difficulty": "medium",
    },
    # --- Hard ---
    {
        "description": (
            "Match a valid CSS class selector: starts with a dot,"
            " followed by a letter, hyphen, or underscore, then"
            " zero or more letters, digits, hyphens, or underscores."
        ),
        "positive": [".my-class", ".a", "._private", ".btn-primary-2"],
        "negative": [
            "my-class",
            ".123",
            ". space",
            ".my class",
            ".",
        ],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a valid semantic version: MAJOR.MINOR.PATCH"
            " where each is a non-negative integer without leading"
            " zeros (except 0 itself). Optionally followed by a"
            " hyphen and a pre-release label (alphanumeric, dots)."
        ),
        "positive": [
            "1.0.0",
            "0.1.0",
            "12.34.56",
            "1.0.0-alpha",
            "1.0.0-beta.1",
        ],
        "negative": [
            "1.0",
            "1.0.0.0",
            "01.0.0",
            "1.02.0",
            "v1.0.0",
            "1.0.0-",
        ],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a positive or negative integer or decimal"
            " number. May optionally start with + or -, must"
            " have digits before or after the decimal point."
        ),
        "positive": ["42", "-3.14", "+0.5", "100", "0.001", "-7"],
        "negative": [".", "+-3", "12.34.5", "abc", "1.2.3", "3e10", ""],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a URL starting with http:// or https://,"
            " followed by a domain (letters, digits, dots,"
            " hyphens), then optionally a path of slashes"
            " and URL-safe characters."
        ),
        "positive": [
            "http://example.com",
            "https://www.google.com/search",
            "https://a.b.c/path/to/page",
            "http://test.io/",
        ],
        "negative": [
            "ftp://example.com",
            "example.com",
            "http://",
            "https:///path",
        ],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a valid MAC address: six groups of two"
            " hexadecimal digits separated by colons."
        ),
        "positive": [
            "00:1A:2B:3C:4D:5E",
            "ff:ff:ff:ff:ff:ff",
            "AA:BB:CC:DD:EE:FF",
            "01:23:45:67:89:ab",
        ],
        "negative": [
            "00:1A:2B:3C:4D",
            "00:1A:2B:3C:4D:5E:6F",
            "001A2B3C4D5E",
            "GG:HH:II:JJ:KK:LL",
            "00-1A-2B-3C-4D-5E",
        ],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a valid Markdown heading: one to six '#'"
            " characters at the start, followed by a space,"
            " then at least one non-whitespace character."
        ),
        "positive": [
            "# Title",
            "## Section",
            "###### Deep",
            "### My Heading 3",
        ],
        "negative": [
            "####### Too deep",
            "#NoSpace",
            "# ",
            "Not a heading",
            "",
        ],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a Python f-string placeholder: starts with"
            " '{', ends with '}', contains at least one"
            " character inside that is not a brace."
        ),
        "positive": ["{x}", "{name!r}", "{value:.2f}", "{obj.attr}"],
        "negative": ["{}", "{ }", "no braces", "{", "}", "{{escaped}}"],
        "difficulty": "medium",
    },
    {
        "description": (
            "Match a valid HTML opening tag (not self-closing)."
            " Starts with '<', then a tag name (letters),"
            " optionally attributes, then '>'. No '/' before '>'."
        ),
        "positive": ["<div>", "<span>", '<a href="link">', "<p>"],
        "negative": ["<div/>", "</div>", "div", "< div>", "<>", "<123>"],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a string containing a repeated word (same word"
            " appearing consecutively, separated by a space)."
            " For example 'the the' or 'is is'."
        ),
        "positive": [
            "the the cat",
            "I said said it",
            "go go go",
            "yes yes",
        ],
        "negative": [
            "no repeats here",
            "the cat the dog",
            "hello world",
        ],
        "difficulty": "hard",
    },
    {
        "description": (
            "Match a credit card-like number: exactly 16 digits,"
            " optionally separated into groups of 4 by dashes"
            " or spaces (but not mixed)."
        ),
        "positive": [
            "1234567890123456",
            "1234-5678-9012-3456",
            "1234 5678 9012 3456",
        ],
        "negative": [
            "1234-5678-90123456",
            "123456789012345",
            "12345678901234567",
            "1234 5678 9012-3456",
            "abcd-efgh-ijkl-mnop",
        ],
        "difficulty": "hard",
    },
]
