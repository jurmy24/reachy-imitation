#!/usr/bin/env python3

import subprocess
import rerun as rr
import rerun.blueprint as rrb

# Initialize Rerun session
rr.init("urdf_viewer", spawn=True)

# Set up a blueprint for the viewer
# blueprint = rrb.Horizontal(
blueprint = rrb.Spatial3DView(origin="robot", name="Robot")
# rrb.TextDocumentView(origin="description", name="Description"),
# column_shares=[3, 2],
# )
rr.send_blueprint(blueprint)

# Log a description
# rr.log(
#     "description",
#     rr.TextDocument(
#         """
# # URDF Viewer
# This example loads a URDF file and displays it in the Rerun Viewer.
# """,
#         media_type=rr.MediaType.MARKDOWN,
#     ),
#     static=True,
# )

# Run the rerun-loader-urdf executable
urdf_path = "/Users/victoroldensand/Documents/CentraleSupélec/s8-project/reachy_description/reachy.URDF"
subprocess.run(["rerun-loader-urdf", urdf_path], check=True)

# Keep the script running to allow interaction with the viewer
input("Press Enter to exit...")
