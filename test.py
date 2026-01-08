import time
from PICO8 import PICO8
from Carts.Celeste import Celeste

# useful Celeste utils
import CelesteUtils as utils

# create a PICO-8 instance with Celeste loaded
p8 = PICO8(Celeste)
CUSTOM_ROOM='''
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
p . . . . . . . . . . . . . . .
w w w w w w w w w w w w w w w w
'''
utils.replace_room(p8, 0, CUSTOM_ROOM)
utils.load_room(p8, 0)

# skip the player spawn
utils.skip_player_spawn(p8)
player=p8.game.get_player()
player.x=120
player.y=116
print(p8.game)