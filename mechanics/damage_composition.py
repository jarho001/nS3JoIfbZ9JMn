"""Bounded composition of the post-core Gen 5 damage arithmetic slices."""

from mechanics.burn import apply_burn_reduction
from mechanics.damage_roll import apply_damage_roll
from mechanics.finalize_damage import finalize_damage
from mechanics.stab import apply_stab
from mechanics.type_effectiveness import apply_type_effectiveness


def damage_from_preroll(
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
):
	"""Compose roll, STAB, type, burn, and finalization kernels in Gen 5 order.

	Caller preconditions: ``pre_roll_damage`` is in ``[0, 33554431]``,
	``roll`` is in ``[0, 15]``, ``type_exponent`` is signed int32, and
	``final_modifier`` is in ``[0, 65535]``. These ranges are not checked here.
	All fields retain the strict int32/bool dtype requirements and broadcasting
	behavior of their constituent kernels.
	"""
	damage = apply_damage_roll(pre_roll_damage, roll)
	damage = apply_stab(damage, stab_eligible, adaptability, typeless)
	damage = apply_type_effectiveness(damage, type_exponent)
	damage = apply_burn_reduction(damage, burned, physical, has_guts)
	return finalize_damage(damage, final_modifier)
