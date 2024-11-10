# onvif_hik.py

import time
import requests
import zeep
from onvif import ONVIFCamera
from requests.auth import HTTPDigestAuth


def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue

def perform_move(ptz, request, timeout):
    # Start continuous move
    ptz.ContinuousMove(request)
    # Wait a certain time
    time.sleep(timeout)
    # Stop continuous move
    ptz.Stop({'ProfileToken': request.ProfileToken})


class Onvif_hik(object):

    def __init__(self, ip: str, username: str, password: str):
        self.ip = ip
        self.username = username
        self.password = password
        self.save_path = "./{}T{}.jpg".format(self.ip, str(time.time()))  # 截图保存路径
        self.content_cam()

    def content_cam(self):
        """
        链接相机地址
        :return:
        """
        try:
            print("here")
            self.mycam = ONVIFCamera(self.ip, 80, self.username, self.password)
            self.media = self.mycam.create_media_service()  # 创建媒体服务
            print('ok1')
            # 得到目标概要文件
            zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
            self.media_profile = self.media.GetProfiles()[0]  # 获取配置信息
            self.ptz = self.mycam.create_ptz_service()  # 创建控制台服务
            return True
        except Exception as e:
            return False

    def Snapshot(self):
        """
        截图
        :return:
        """
        res = self.media.GetSnapshotUri({'ProfileToken': self.media_profile.token})

        response = requests.get(res.Uri, auth=HTTPDigestAuth(self.username, self.password))
        with open(self.save_path, 'wb') as f:  # 保存截图
            f.write(response.content)

    def get_presets(self):
        """
        获取预置点列表
        :return:预置点列表--所有的预置点
        """
        presets = self.ptz.GetPresets({'ProfileToken': self.media_profile.token})  # 获取所有预置点,返回值：list
        return presets

    def goto_preset(self, presets_token: int):
        """
        移动到指定预置点
        :param presets_token: 目的位置的token，获取预置点返回值中
        :return:
        """
        try:
            params = self.ptz.create_type('GotoPreset')
            params.ProfileToken = self.media_profile.token
            params.PresetToken = presets_token
            self.ptz.GotoPreset(params)
        except Exception as e:
            print(e)

    def zoom(self, zoom: str, timeout: int = 1):
        """
        变焦
        :param zoom: 1为拉近或-1为远离 
        :param timeout: 生效时间
        :return:
        """
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile.token
        request.Velocity = {"Zoom": zoom}
        self.ptz.ContinuousMove(request)
        time.sleep(timeout)
        self.ptz.Stop({'ProfileToken': request.ProfileToken})

    def get_status(self):
        """
        获取当前预置点的信息
        :return:
        """
        params = self.ptz.create_type('GetStatus')
        params.ProfileToken = self.media_profile.token
        res = self.ptz.GetStatus(params)
        return res
    
    def set_preset(self, preset_token, preset_name=None):
        """
        将当前位置设置为指定的预设点
        :param preset_token: 预设点的标识符
        :param preset_name: 预设点的名称，如果为空，则使用预设点标识符作为名称
        :return: 设置结果
        """
        try:
            request = self.ptz.create_type('SetPreset')
            request.ProfileToken = self.media_profile.token
            request.PresetToken = str(preset_token)  # 预设点的标识符
            request.PresetName = preset_name if preset_name else f"Preset{preset_token}"

            # 发送请求设置预设点
            res = self.ptz.SetPreset(request)
            return res  # 返回结果，通常包含预设点的Token
        except Exception as e:
            print(f"Error setting preset: {e}")
            return None
    
    def get_preset_data(self, preset_token):
        """
        获取特定预置点的数据
        :param preset_token: 预置点的标识符
        :return: 特定预置点的数据
        """
        presets = self.get_presets()
        if presets is not None:
            for preset in presets:
                if preset.token == str(preset_token):
                    return preset
        return None
    
    def move_up(self, velocity=1):
        """
        Move the camera up.
        :param velocity: Speed of movement (0 to 1)
        """
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile.token
        request.Velocity = {
            'PanTilt': {'x': 0, 'y': velocity},
            'Zoom': {'x': 0}
        }
        perform_move(self.ptz, request, 1)

    def move_down(self, velocity=1):
        """
        Move the camera down.
        :param velocity: Speed of movement (0 to 1)
        """
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile.token
        request.Velocity = {
            'PanTilt': {'x': 0, 'y': -velocity},
            'Zoom': {'x': 0}
        }
        perform_move(self.ptz, request, 1)

    def move_left(self, velocity=1):
        """
        Move the camera left.
        :param velocity: Speed of movement (0 to 1)
        """
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile.token
        request.Velocity = {
            'PanTilt': {'x': -velocity, 'y': 0},
            'Zoom': {'x': 0}
        }
        perform_move(self.ptz, request, 1)

    def move_right(self, velocity=1):
        """
        Move the camera right.
        :param velocity: Speed of movement (0 to 1)
        """
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile.token
        request.Velocity = {
            'PanTilt': {'x': velocity, 'y': 0},
            'Zoom': {'x': 0}
        }
        perform_move(self.ptz, request, 1)

    def stop_movement(self):
        """
        Stop the movement of the camera.
        """
        self.ptz.Stop({'ProfileToken': self.media_profile.token})

    def continuous_move(self, velocity_x=0, velocity_y=0, velocity_zoom=0):
        """
        Perform continuous movement of the camera.
        :param velocity_x: Speed of movement along the x-axis (-1 to 1)
        :param velocity_y: Speed of movement along the y-axis (-1 to 1)
        :param velocity_zoom: Speed of zoom movement (-1 to 1)
        """
        request = self.ptz.create_type('ContinuousMove')
        request.ProfileToken = self.media_profile.token
        request.Velocity = {
            'PanTilt': {'x': velocity_x, 'y': velocity_y},
            'Zoom': {'x': velocity_zoom}
        }
        self.ptz.ContinuousMove(request)

    def absolute_move(self, position_x=0, position_y=0, zoom=0):
        """
        Perform absolute movement of the camera.
        :param position_x: Absolute position along the x-axis (-1 to 1)
        :param position_y: Absolute position along the y-axis (-1 to 1)
        :param zoom: Absolute zoom level (0 to 1)
        """
        request = self.ptz.create_type('AbsoluteMove')
        request.ProfileToken = self.media_profile.token
        request.Position = {
            'PanTilt': {'x': position_x, 'y': position_y},
            'Zoom': {'x': zoom}
        }
        self.ptz.AbsoluteMove(request)

