---
name: roman-numeral
description: Convert a small integer to a Roman numeral.
---
Skill: Roman numerals.

Convert an integer (1-100) to a Roman numeral using these symbols: 1=I, 4=IV, 5=V,
9=IX, 10=X, 40=XL, 50=L, 90=XC, 100=C.

Procedure: repeatedly subtract the largest value whose symbol fits, appending its
symbol each time, until you reach zero. Use uppercase letters and no spaces.

Worked example: 24 = XX + IV = "XXIV"; 49 = XL + IX = "XLIX".
