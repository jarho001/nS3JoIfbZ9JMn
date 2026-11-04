---
title: Keep staged requests aligned with their episode
author: "/root/task037"
date: 2026-11-04
branch: main
id: l-8556df54
scope: repo
status: active
src: agent
touches:
  - "simulator/request_session.py#project_submission_view"
  - "simulator/request_session.py#_idle_transition"
---

Project the next actor request from result.session.battle, which is the state named by result.session.episode. Keep previous turn events separately projected from result.before and result.transition; otherwise a new token can carry old HP, PP, phase, or terminal data. Use the fixed-shape _empty_tie_transition for waiting and rejected no-advance results with the matching diagnostic status, and reserve transition_with_speed_ties for the second valid commit that resolves the pair.
