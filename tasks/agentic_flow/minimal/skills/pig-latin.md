---
name: pig-latin
description: Convert a consonant-initial word to Pig Latin.
---
Skill: Pig Latin converter.

To convert a word that begins with one or more consonants to Pig Latin, move the
leading consonant cluster (all letters before the first vowel) to the end of the word
and then add "ay". Keep everything lowercase.

Procedure: find the first vowel (a, e, i, o, u); take all letters before it, append
them after the rest of the word, then append "ay".

Worked example: "pig" -> "igpay"; "smile" -> "ilesmay"; "glove" -> "oveglay".
