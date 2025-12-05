import sys
import mujoco
from mujoco import viewer

if len(sys.argv) != 2:
    print("Usage: python view_xml.py path/to/scenario.xml")
    sys.exit(1)

xml_path = sys.argv[1]
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Loaded:", xml_path)
viewer.launch(model, data)