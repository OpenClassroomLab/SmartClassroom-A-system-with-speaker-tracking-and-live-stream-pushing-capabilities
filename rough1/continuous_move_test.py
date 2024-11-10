import time
from time import sleep

from onvif import ONVIFCamera
import zeep

XMAX = 1
XMIN = -1
YMAX = 1
YMIN = -1

def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue

def perform_move(ptz, request, timeout):
    # Start continuous move
    ptz.ContinuousMove(request)
    # Wait a certain time
    sleep(timeout)
    # Stop continuous move
    ptz.Stop({'ProfileToken': request.ProfileToken})
    
    
def stop_move_image(self):
    # 焦距停止移动
    self.imaging.Stop({'VideoSourceToken': self.profile.VideoSourceConfiguration.token})

def move_up(ptz, request, timeout=1):
    print('move up...') 
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = YMAX
    perform_move(ptz, request, timeout)

def move_down(ptz, request, timeout=1):
    print('move down...') 
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = YMIN
    perform_move(ptz, request, timeout)

def move_right(ptz, request, timeout=1):
    print('move right...') 
    request.Velocity.PanTilt.x = XMAX
    request.Velocity.PanTilt.y = 0
    perform_move(ptz, request, timeout)

def move_left(ptz, request, timeout=1):
    print('move left...') 
    request.Velocity.PanTilt.x = XMIN
    request.Velocity.PanTilt.y = 0
    perform_move(ptz, request, timeout)

def continuous_move():
    mycam = ONVIFCamera("...",80,'...', "...")
    # Create media service object
    media = mycam.create_media_service()
    # Create ptz service object
    ptz = mycam.create_ptz_service()

    # Get target profile
    zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
    media_profile = media.GetProfiles()[0]

    # Get PTZ configuration options for getting continuous move range
    request = ptz.create_type('GetConfigurationOptions')
    request.ConfigurationToken = media_profile.PTZConfiguration.token
    ptz_configuration_options = ptz.GetConfigurationOptions(request)

    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = media_profile.token
    ptz.Stop({'ProfileToken': media_profile.token})

    if request.Velocity is None:
        request.Velocity = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
        request.Velocity = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
        request.Velocity.PanTilt.space = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].URI
        request.Velocity.Zoom.space = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].URI
   
    # Get range of pan and tilt
    # NOTE: X and Y are velocity vector
    global XMAX, XMIN, YMAX, YMIN
    XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
    XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
    YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
    YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min

    # move right
    move_right(ptz, request)
    time.sleep(5)

    # move left
    #move_left(ptz, request)

    # Move up
    #move_up(ptz, request)

    # move down
    #move_down(ptz, request)

def absolute_move():
    pan = 0.1
    pan_speed = 1
    tilt = 0
    tilt_speed = 1
    zoom = 0
    zoom_speed = 1

    mycam = ONVIFCamera("...",80,'...', "...")
    # Create media service object
    media = mycam.create_media_service()
    # Create ptz service object
    ptz = mycam.create_ptz_service()

    # Get target profile
    zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
    media_profile = media.GetProfiles()[0]

    # Get PTZ configuration options for getting absolute move range
    request = ptz.create_type('GetConfigurationOptions')
    request.ConfigurationToken = media_profile.PTZConfiguration.token
    # ptz_configuration_options = ptz.GetConfigurationOptions(request)

    request = ptz.create_type('AbsoluteMove')
    request.ProfileToken = media_profile.token
    ptz.Stop({'ProfileToken': media_profile.token})

    if request.Position is None:
        request.Position = ptz.GetStatus({'ProfileToken': media_profile.token}).Position
    if request.Speed is None:
        request.Speed = ptz.GetStatus({'ProfileToken': media_profile.token}).Position

    request.Position.PanTilt.x = pan
    request.Speed.PanTilt.x = pan_speed

    request.Position.PanTilt.y = tilt
    request.Speed.PanTilt.y = tilt_speed

    request.Position.Zoom = zoom
    request.Speed.Zoom = zoom_speed

    ptz.AbsoluteMove(request)
    print('finish')


    
if __name__ == '__main__':
    continuous_move()    
    #absolute_move()    