import time

import requests
import zeep
from onvif import ONVIFCamera
from requests.auth import HTTPDigestAuth


def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue


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
            self.mycam = ONVIFCamera(self.ip, 80, self.username, self.password)
            self.media = self.mycam.create_media_service()  # 创建媒体服务
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
            # self.ptz.GotoPreset(
            #     {'ProfileToken': self.media_profile.token, "PresetToken": presets_token})  # 移动到指定预置点位置
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
        # print(res)
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

    

if __name__ == '__main__':
    o = Onvif_hik(ip="10.46.28.33", username="admin", password="1qaz2wsx,")

    #获取所有预置点
    #presets = o.get_presets()
    #for preset in presets:
        #print(preset)
    
    #获取摄像头当前状态
    status = o.get_status()
    #print(status)
    
    result = o.set_preset(3)
    if result is not None:
        print(f"Preset set successfully. Preset Token: {result}")
    else:
        print("Failed to set preset.")

        
    #获取预置点11的数据
    #preset_data = o.get_preset_data(11)
    #if preset_data is not None:
        #print(f"Preset 11 data: {preset_data}")
    #else:
        #print("Preset 11 not found or error occurred.")
        
    # 移动到预置点
    #o.goto_preset(3)