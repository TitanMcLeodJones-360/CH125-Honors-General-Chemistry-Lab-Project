"""Calculate molar mass from a chemical formula using Python 3.

Run with no arguments for the interactive calculator, or pass a quoted formula:
    python molecular_weight.py "FeCl3 * 9 H2O"
"""

import argparse
import re
from collections import defaultdict
from decimal import Decimal

# CIAAW 2024 abridged standard atomic weights, with full precision for Al and P.
# https://www.ciaaw.org/abridged-atomic-weights.htm
# https://www.ciaaw.org/atomic-weights.htm
ATOMIC_WEIGHTS = {
    symbol: Decimal(weight)
    for symbol, weight in (
        entry.split("=") for entry in """
        H=1.0080 He=4.0026 Li=6.94 Be=9.0122 B=10.81 C=12.011 N=14.007
        O=15.999 F=18.998 Ne=20.180 Na=22.990 Mg=24.305 Al=26.9815384
        Si=28.085 P=30.973761998 S=32.06 Cl=35.45 Ar=39.95 K=39.098
        Ca=40.078 Sc=44.956 Ti=47.867 V=50.942 Cr=51.996 Mn=54.938
        Fe=55.845 Co=58.933 Ni=58.693 Cu=63.546 Zn=65.38 Ga=69.723
        Ge=72.630 As=74.922 Se=78.971 Br=79.904 Kr=83.798 Rb=85.468
        Sr=87.62 Y=88.906 Zr=91.222 Nb=92.906 Mo=95.95 Ru=101.07
        Rh=102.91 Pd=106.42 Ag=107.87 Cd=112.41 In=114.82 Sn=118.71
        Sb=121.76 Te=127.60 I=126.90 Xe=131.29 Cs=132.91 Ba=137.33
        La=138.91 Ce=140.12 Pr=140.91 Nd=144.24 Sm=150.36 Eu=151.96
        Gd=157.25 Tb=158.93 Dy=162.50 Ho=164.93 Er=167.26 Tm=168.93
        Yb=173.05 Lu=174.97 Hf=178.49 Ta=180.95 W=183.84 Re=186.21
        Os=190.23 Ir=192.22 Pt=195.08 Au=196.97 Hg=200.59 Tl=204.38
        Pb=207.2 Bi=208.98 Th=232.04 Pa=231.04 U=238.03
        """.split()
    )
}

NO_STANDARD_WEIGHT = set("""
    Tc Pm Po At Rn Fr Ra Ac Np Pu Am Cm Bk Cf Es Fm Md No Lr
    Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())

GROUP_END = {"(": ")", "[": "]"}
HYDRATE_SEPARATORS = "*·"
ELEMENT = re.compile(r"[A-Z][a-z]?")
SUBSCRIPT = re.compile(r"[0-9]+")
COEFFICIENT = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")


class FormulaError(ValueError):
    """An invalid or unsupported chemical formula."""


class FormulaParser:
    def __init__(self, formula):
        self.formula = formula
        self.position = 0

    def peek(self):
        """Return the next non-space character, or an empty string at the end."""
        while (self.position < len(self.formula)
               and self.formula[self.position].isspace()):
            self.position += 1
        return self.formula[self.position:self.position + 1]

    def read(self, pattern):
        self.peek()
        match = pattern.match(self.formula, self.position)
        if match:
            self.position = match.end()
            return match.group()
        return None

    def error(self, message):
        return FormulaError(f"{message} (position {self.position + 1}).")

    def read_number(self, pattern):
        digits = self.read(pattern)
        if digits is None:
            return Decimal(1)
        value = Decimal(digits)
        if value <= 0:
            raise self.error("Subscripts and hydrate coefficients must be positive")
        return value

    def read_group(self, closing=None):
        """Read element counts within a component or a bracketed group."""
        counts = defaultdict(Decimal)
        while True:
            char = self.peek()
            if not char or char == closing or char in HYDRATE_SEPARATORS:
                break

            if char in GROUP_END:
                self.position += 1
                group_end = GROUP_END[char]
                part = self.read_group(group_end)
                self.position += 1  # Consume the closing bracket.
            else:
                if char in ")]":
                    message = f"Expected '{closing}', found '{char}'" if closing else (
                        f"Unexpected closing bracket '{char}'"
                    )
                    raise self.error(message)

                start = self.position
                symbol = self.read(ELEMENT)
                if symbol is None:
                    raise self.error(
                        f"Expected an element or opening bracket, found '{char}'. "
                        "Element symbols start with a capital letter"
                    )
                if symbol not in ATOMIC_WEIGHTS:
                    self.position = start
                    if symbol in NO_STANDARD_WEIGHT:
                        raise self.error(f"'{symbol}' has no standard atomic weight")
                    raise self.error(f"Unknown element '{symbol}'. Check capitalization")
                part = {symbol: Decimal(1)}

            multiplier = self.read_number(SUBSCRIPT)
            for symbol, count in part.items():
                counts[symbol] += multiplier * count

        if closing and char != closing:
            raise self.error(f"Missing closing bracket '{closing}'")
        if not counts:
            raise self.error("Expected an element inside the group" if closing else
                             "Expected an element or group")
        return counts

    def parse(self):
        counts = self.read_group()
        while self.peek():
            self.position += 1  # Consume the hydrate separator.
            multiplier = self.read_number(COEFFICIENT)
            for symbol, count in self.read_group().items():
                counts[symbol] += multiplier * count
        return dict(counts)


def calculate_molar_mass(formula):
    """Return the element counts and total mass in grams per mole."""
    if len(formula) > 256:
        raise FormulaError("Enter a formula of 256 characters or fewer.")
    if not formula.strip():
        raise FormulaError("Enter a chemical formula, such as H2O.")

    counts = FormulaParser(formula).parse()
    mass = sum(count * ATOMIC_WEIGHTS[symbol] for symbol, count in counts.items())
    return counts, mass


def show_result(formula):
    counts, mass = calculate_molar_mass(formula)
    composition = ", ".join(f"{symbol}={count.normalize():f}"
                            for symbol, count in counts.items())
    print(f"  Atom counts: {composition}")
    print(f"  Molar mass:  {mass:.3f} g/mol")


def main():
    parser = argparse.ArgumentParser(description="Calculate molar mass in g/mol.")
    parser.add_argument("formula", nargs="?", help="chemical formula, e.g. CaSO4*0.5H2O")
    args = parser.parse_args()
    if args.formula is not None:
        try:
            show_result(args.formula)
        except FormulaError as error:
            parser.exit(2, f"Error: {error}\n")
        return

    print("Molecular Weight Calculator")
    print("Enter a full formula, such as Ca3(PO4)2 or CaSO4*0.5H2O.")
    print("Use * or · for hydrates. Type quit to exit.")
    while True:
        try:
            formula = input("\nFormula> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if formula.lower() == "quit":
            return
        try:
            show_result(formula)
        except FormulaError as error:
            print(f"  Error: {error}")


if __name__ == "__main__":
    main()
