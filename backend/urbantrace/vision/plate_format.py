"""Indian license-plate normalisation, validation and format-driven repair.

Design rule (from the spec): NEVER invent characters. This module only performs
corrections that are forced by the plate grammar itself -- e.g. a character in a
position that MUST be a digit, which OCR returned as a visually-identical
letter. Every result keeps the raw OCR string alongside the repaired one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# Valid Indian state / UT registration prefixes.
STATE_CODES = {
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP","HR","JH",
    "JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","OR","PB","PY",
    "RJ","SK","TN","TR","TS","UK","UA","UP","WB","BH",
}

# Standard:  TN 09 AB 1234   |  TN 09 A 1234  |  TN 09 ABC 1234
RE_STANDARD = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$")
# Bharat series: 23 BH 1234 AB
RE_BH = re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$")
# Older / defence style without series letters: TN 09 1234
RE_SHORT = re.compile(r"^([A-Z]{2})(\d{1,2})(\d{4})$")

# OCR confusions, applied ONLY where the grammar forces the character class.
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
                   "Z": "2", "E": "3", "A": "4", "S": "5", "G": "6", "T": "7",
                   "B": "8", "R": "8", "P": "9"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G",
                   "7": "T", "8": "B", "9": "P"}


@dataclass
class PlateValidation:
    raw: str                 # exactly what OCR emitted
    normalized: str          # whitespace/punctuation stripped, uppercased
    corrected: str           # after grammar-forced character repair
    is_valid: bool           # matches a known Indian plate grammar
    state_code: str | None
    state_valid: bool
    pattern: str | None      # which grammar matched
    corrections: list        # [(index, from, to, reason)]
    repaired: bool = False   # True if grammar-forced repair was applied
    format_confidence: float = 1.0  # 1.0 = clean parse; penalised per repair

    @property
    def final(self) -> str:
        """The plate string the system should use as the identity key."""
        return self.corrected if self.is_valid else self.normalized

    def to_dict(self) -> dict:
        d = asdict(self)
        d["final"] = self.final
        return d


def normalize(text: str) -> str:
    """Strip separators and uppercase. Purely lexical - no character guessing."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _coerce(segment: str, to_digits: bool) -> tuple[str, list]:
    """Force a segment into digits or letters using the confusion maps."""
    table = LETTER_TO_DIGIT if to_digits else DIGIT_TO_LETTER
    want_digit = to_digits
    out, fixes = [], []
    for i, ch in enumerate(segment):
        ok = ch.isdigit() if want_digit else ch.isalpha()
        if ok:
            out.append(ch)
        elif ch in table:
            out.append(table[ch])
            fixes.append((i, ch, table[ch], "digit-position" if want_digit else "letter-position"))
        else:
            out.append(ch)
    return "".join(out), fixes


MAX_REPAIRS = 2  # beyond this we report uncertainty instead of asserting a plate


def _try_standard(s: str) -> tuple[str, str, list] | None:
    """Repair against TN09AB1234 grammar using segment positions."""
    if not (9 <= len(s) <= 11):
        return None
    # Anchor on the trailing 4 digits and leading 2 letters.
    head, fx1 = _coerce(s[:2], to_digits=False)
    tail, fx2 = _coerce(s[-4:], to_digits=True)
    middle = s[2:-4]
    if not middle:
        return None
    # middle = RTO digits (1-2) followed by series letters (0-3)
    best = None
    for ndig in (2, 1):
        if len(middle) < ndig:
            continue
        rto, fx3 = _coerce(middle[:ndig], to_digits=True)
        series, fx4 = _coerce(middle[ndig:], to_digits=False)
        cand = f"{head}{rto}{series}{tail}"
        fixes = fx1 + [(i + 2, a, b, r) for i, a, b, r in fx3] + \
                [(i + 2 + ndig, a, b, r) for i, a, b, r in fx4] + \
                [(i + len(s) - 4, a, b, r) for i, a, b, r in fx2]
        if RE_STANDARD.match(cand):
            # Prefer the candidate needing fewer corrections.
            if best is None or len(fixes) < len(best[2]):
                best = (cand, "STANDARD", fixes)
    return best


def validate(text: str) -> PlateValidation:
    raw = text or ""
    norm = normalize(raw)

    # Already valid -> accept untouched.
    for rx, name in ((RE_STANDARD, "STANDARD"), (RE_BH, "BHARAT"), (RE_SHORT, "SHORT")):
        m = rx.match(norm)
        if m:
            state = m.group(1) if name != "BHARAT" else "BH"
            return PlateValidation(raw, norm, norm, state in STATE_CODES, state,
                                   state in STATE_CODES, name, [],
                                   repaired=False, format_confidence=1.0)

    # Attempt grammar-forced repair.
    repaired = _try_standard(norm)
    if repaired:
        cand, pattern, fixes = repaired
        m = RE_STANDARD.match(cand)
        state = m.group(1)
        # Each grammar-forced repair costs confidence. A plate needing many
        # repairs is reported as uncertain rather than asserted as fact.
        conf = max(0.0, 1.0 - 0.18 * len(fixes))
        ok = bool(state in STATE_CODES and len(fixes) <= MAX_REPAIRS)
        return PlateValidation(raw, norm, cand if ok else norm, ok, state,
                               state in STATE_CODES, pattern if ok else None,
                               fixes if ok else [],
                               repaired=bool(fixes) and ok, format_confidence=conf)

    return PlateValidation(raw, norm, norm, False, norm[:2] if len(norm) >= 2 else None,
                           False, None, [])


def format_display(plate: str) -> str:
    """TN09AB1234 -> 'TN 09 AB 1234' for UI display only."""
    m = RE_STANDARD.match(plate or "")
    if m:
        return " ".join(m.groups())
    return plate


if __name__ == "__main__":
    tests = ["TNO9AB1234", "TN09AB1234", "TN 09 AB 1234", "TNO9A81234",
             "1N09AB1234", "KA05MJ2020", "22BH1234AB", "GARBAGE", "TN09AB123"]
    for t in tests:
        v = validate(t)
        flag = "VALID  " if v.is_valid else "INVALID"
        print(f"{t:16} -> {v.final:12} {flag} state={v.state_code} "
              f"fixes={[(f[1],'->',f[2]) for f in v.corrections]}")
