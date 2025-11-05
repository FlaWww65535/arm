# referenced from baguetter-main https://github.com/mixedbread-ai/baguetter
import re
import string

def lowercasing(text: str):
  return text.lower()

def normalize_ampersand(text: str) -> str:
  return text.replace("&", " and ")

_SPECIAL_CHARS_TRANS = str.maketrans("‘’´“”–-", "'''\"\"--")  # noqa: RUF001
def normalize_special_chars(text: str) -> str:
  return text.translate(_SPECIAL_CHARS_TRANS)

def normalize_acronyms(text: str) -> str:
  return re.sub(r"\.(?!(\S[^. ])|\d)", "", text)

_PUNCTUATION_TRANSLATION = str.maketrans(
    string.punctuation,
    " " * len(string.punctuation),
)
def remove_punctuation(text: str) -> str:
  return text.translate(_PUNCTUATION_TRANSLATION)

def strip_whitespaces(text: str) -> str:
    """Remove extra whitespaces.

    Args:
        text (str): The input text.

    Returns:
        The normalized text.

    """
    return " ".join(text.split())

# def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
#     """Remove stopwords.

#     Args:
#         tokens (List[str]): The input tokens.
#         stopwords (Set[str]): The stopwords.

#     Returns:
#         The normalized tokens.

#     """
#     return [t for t in tokens if t not in stopwords]

# def apply_stemmer(tokens: list[str], stemmer) -> list[str]:
  

#     """Apply stemmer.

#     Args:
#         tokens (List[str]): The input tokens.
#         stemmer (Callable[[str], str]): The stemmer.

#     Returns:
#         The normalized tokens.

#     """
#     return list(map(stemmer, tokens))

# def remove_empty(tokens: list[str]) -> list[str]:
#     """Remove empty tokens.

#     Args:
#         tokens (List[str]): The input tokens.

#     Returns:
#         The normalized tokens.

#     """
#     return [t for t in tokens if t]

def normalize(text: str):
  # , remove_stopwords, apply_stemmer, remove_empty
  for f in [lowercasing, normalize_ampersand, normalize_special_chars, normalize_acronyms, remove_punctuation, strip_whitespaces]:
    text = f(text)

  return text

def tokenize(text: str):
  text = normalize(text).split(' ')
  return set(text)