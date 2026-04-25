---
description: Analyze a video link, local path, or attached video file and render high-potential social clips with captions and crops.
argument-hint: '<video link|path|attached file> [--ratios 9:16,1:1,16:9] [--clips 5] [--min-score 0.70]'
allowed-tools: [Bash, Read, Write, AskUserQuestion]
---

Invoke the `clip` skill with the user's arguments: $ARGUMENTS

Run the full loop: prepare the source, mine candidates, score every candidate
with the harness model, write `scoring-response.json`, review, approve selected
clips, render final MP4s, and return the rendered output paths.
