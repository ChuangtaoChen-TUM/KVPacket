---
name: caesar-shift
description: Decode a word encrypted with a Caesar shift of 3.
---
Skill: Caesar-shift decoder.

A Caesar cipher with shift 3 replaces every letter with the letter three positions
later in the alphabet. To DECODE, shift each letter three positions EARLIER, wrapping
from a back to z when needed (so d->a, e->b, a->x).

Procedure: for each letter, move back by 3 in the alphabet and keep the result
lowercase. Leave any non-letters unchanged.

Worked example: the encoded word "khoor" decodes to "hello" (k->h, h->e, o->l,
o->l, r->o).
