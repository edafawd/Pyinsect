[app]

# (str) Title of your application
title = Insect Detector

# (str) Package name
package.name = insectdetector

# (str) Package domain (needed for android packaging)
package.domain = org.ivan

# (str) Source code directory where main.py lives
source.dir = .

# (list) Source files to include (includes ONNX for your model)
source.include_exts = py,png,jpg,kv,atlas,onnx

# (str) Application versioning
version = 0.1

# (list) Application requirements
requirements = hostpython3==3.11.0,python3==3.11.0,kivy==2.3.0,cython==0.29.36,pillow,numpy==v1.26.4,onnxruntime,plyer,pyjnius,android

# (str) Supported orientations
orientation = portrait

# (bool) Indicate if the application should be fullscreen
fullscreen = 0

# (list) Permissions needed for camera and saving images
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# (int) Target Android API
android.api = 33

# (int) Minimum Android API supported
android.minapi = 24

# (str) Android NDK version
android.ndk = 25b

# (list) Architecture of target devices
android.archs = arm64-v8a

# (bool) Accept SDK license automatically
android.accept_sdk_license = True

[buildozer]

# (int) Log level (2 = debug info)
log_level = 2

# (int) Display warning if run as root
warn_on_root = 1
