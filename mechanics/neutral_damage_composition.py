"""Composition from explicit effective stats through neutral final damage."""

from mechanics.damage_composition import damage_from_preroll
from mechanics.neutral_damage import neutral_damage_before_roll


def damage_from_effective_stats_neutral(
	level,
	base_power,
	effective_attack,
	effective_defense,
	roll,
	stab_eligible,
	adaptability,
	typeless,
	type_exponent,
	burned,
	physical,
	has_guts,
	final_modifier,
):
	"""Compose neutral core/weather with the explicit post-core damage stages.

	Inputs are strict int32 values with level ``[1, 100]``, base power
	``[1, 65535]``, and effective Attack/Defense ``[0, 65535]``,
	roll ``[0, 15]``, signed-int32 type exponent, and final modifier
	``[0, 65535]``. Flags are strict bools. These are caller preconditions;
	values broadcast as in the constituent kernels. The prefix produces
	pre-roll damage in ``[0, 1048575]``, satisfying
	:func:`damage_from_preroll`'s pre-roll precondition.
	"""
	pre_roll_damage = neutral_damage_before_roll(
		level, base_power, effective_attack, effective_defense
	)
	return damage_from_preroll(
		pre_roll_damage,
		roll,
		stab_eligible,
		adaptability,
		typeless,
		type_exponent,
		burned,
		physical,
		has_guts,
		final_modifier,
	)
