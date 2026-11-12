import mujoco
import os

# Resolve the '~' to your absolute home directory path (/home/gad/...)
urdf_path = os.path.expanduser("~/rl_ippo/FTP4RM/ur_e_description/urdf/ur5e.urdf")

# Parse and compile the URDF
model = mujoco.MjModel.from_xml_path(urdf_path)

# Save the compiled model as a native MJCF XML file
mujoco.mj_saveLastXML("ur5e.xml", model)
print("Successfully converted URDF to MJCF!")