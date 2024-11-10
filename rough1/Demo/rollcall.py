<<<<<<< HEAD
import time
from onvif_hik import Onvif_hik

##  触发方式考虑利用中控台点击（未解决）

if __name__ == '__main__':
    o30 = Onvif_hik(ip="...", username=".", password="...")
    o32 = Onvif_hik(ip="...", username=".", password="...")
    o33 = Onvif_hik(ip="...", username=".", password="...")
    # 4.1 建议修改为统一的预置点便于管理
    o30.goto_preset(10)
    o32.goto_preset(3)
    o33.goto_preset(5)
    print('Cameras movement completed')

=======
import time
from onvif_hik import Onvif_hik

##  触发方式考虑利用中控台点击（未解决）

if __name__ == '__main__':
    o30 = Onvif_hik(ip="...", username=".", password="...")
    o32 = Onvif_hik(ip="...", username=".", password="...")
    o33 = Onvif_hik(ip="...", username=".", password="...")
    # 4.1 建议修改为统一的预置点便于管理
    o30.goto_preset(10)
    o32.goto_preset(3)
    o33.goto_preset(5)
    print('Cameras movement completed')

>>>>>>> 42549d3 (准备pull到github)
