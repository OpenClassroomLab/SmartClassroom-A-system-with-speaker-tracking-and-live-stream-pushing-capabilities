
import time
from loguru import logger 

from onvif_hik import Onvif_hik

if __name__ == '__main__':

    logger.info('start')
    start_time = time.time()
    
    #o30 = Onvif_hik(ip="...", username=".", password="...")
    #o31 = Onvif_hik(ip="...", username=".", password="...")
    o32 = Onvif_hik(ip="...", username=".", password="...")
    #o33 = Onvif_hik(ip="...", username=".", password="...")
    
    end_time = time.time()

    logger.info(end_time-start_time)
    
    c32.goto_preset(21)
    # c31.goto_preset(1)
    # c32.goto_preset(1)
    # c30.goto_preset(15)

    # o30.continuous_move(0.5,0.5)
    # time.sleep(5)

    # o30.stop_movement()

    #o30.absolute_move(0.5,0,0)

    logger.info('end')
    #获取所有预置点，均为x=-1，y=-0.5
    #presets = o.get_presets()
    #for preset in presets:
        #logger.info(preset)
