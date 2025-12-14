import time
from PICO8 import PICO8
from Carts.Celeste import Celeste

# useful Celeste utils
import CelesteUtils as utils

# create a PICO-8 instance with Celeste loaded
p8 = PICO8(Celeste)
# 多实例
p82 = PICO8(Celeste)
# swap 100m with this level and reload it
room_data = '''
w w w w w w w w w w . . . . w w
w w w w w w w w w . . . . . < w
w w w v v v v . . . . . . . < w
w w > . . . . . . . . . . . . .
w > . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . b . . . b . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
p . . . . . . . . . . . . . . .
w w w w w w w w w w w w w w w w
'''
utils.replace_room(p8, 0, room_data)
utils.load_room(p8, 0)

# skip the player spawn
utils.skip_player_spawn(p8)
# view the room


# hold right + x
#p8.set_inputs(False, True,  False,  False, True, False)

# run for 10f while outputting player info
print(p8.game.get_player())
player=p8.game.get_player()
player.x=80
for f in range(2):
  p8.step()
  p8.set_inputs()
  print(p8.game)

# 平台跳跃游戏目标是让主角前往y<0的地方，即向上爬
# 上冲 3
# 跳 2