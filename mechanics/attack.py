"""Neutral Gen 5 Attack stored-stat and stage-scaling composition.

Inputs follow :func:`mechanics.stats.derive_stored_stats`: int32 base stats in
[1, 255], level in [1, 100], IVs in [0, 31], EVs in [0, 255], and nature
multipliers 9/10/11. ``stage`` is a signed int32 ModifyBoost value. These are
documented caller preconditions, not dynamically checked. This computes only
``calculateStat('atk', stage, 1)`` with identity ModifyBoost and no field stat
selection or stat modifiers; abilities, items, status, and other events are
outside this function.
"""

from .stat_stage import _as_int32, scale_stat_stage
from .stats import derive_stored_stats


def derive_attack(base_stats, level, ivs, evs, nature_multipliers, stage):
	"""Derive Attack from six-stat tables, then apply its Gen 5 stage.

	The inputs broadcast as in ``derive_stored_stats``; the resulting Attack
	array broadcasts with ``stage``. The return dtype is int32.
	"""
	stage_array = _as_int32(stage, "stage")
	stored = derive_stored_stats(base_stats, level, ivs, evs, nature_multipliers, False)
	return scale_stat_stage(stored[..., 1], stage_array)
