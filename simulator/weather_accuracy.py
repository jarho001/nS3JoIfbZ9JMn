"""Gen5 Thunder and Blizzard ModifyMove accuracy, before stage queries."""
import jax.numpy as jnp
from . import gen5_engine_tables as t
THUNDER=t.MOVE_IDS['thunder'];BLIZZARD=t.MOVE_IDS['blizzard']


def accuracy(s,move,value):
    weather=s.weather_state[0]
    sure=((move==THUNDER)&(weather==1))|((move==BLIZZARD)&(weather==4))
    # Zero is the shared representation of native boolean true accuracy.
    return jnp.where(sure,0,jnp.where((move==THUNDER)&(weather==2),50,value)).astype(jnp.int32)
