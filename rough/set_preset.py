from onvif_hik import Onvif_hik


o30 = Onvif_hik(ip="...", username=".", password="...")
#o31 = Onvif_hik(ip="...", username=".", password="...")
#o32 = Onvif_hik(ip="...", username=".", password="...")
#o33 = Onvif_hik(ip="...", username=".", password="...")
#c34 = Onvif_hik(ip="...", username="...", password="...")
#获取所有预置点
#presets = o.get_presets()
#for preset in presets:
    #print(preset)
    
#获取摄像头当前状态
status33 = c30.get_status()

#print(status)
    

#设置当前位置为预置点x   
result33 = c30.set_preset(15)
#result31 = c31.set_preset(16)

if result33 is not None:
    print(f"Preset set successfully. Preset Token: {result33}")
else:
    print("Failed to set preset.")

# if result31 is not None:
#     print(f"31 Preset set successfully. Preset Token: {result31}")
# else:
#     print("Failed to set preset.")
        
#获取预置点11的数据
# preset_data = c30.get_preset_data(11)
# if preset_data is not None:
#     print(f"Preset 11 data: {preset_data}")
# else:
#     print("Preset 11 not found or error occurred.")   