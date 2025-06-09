![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.aquitaineonline.com%2Fimages%2Fstories%2FEconomie_Industrie_2021%2FReachy_09.jpg&f=1&nofb=1&ipt=52f307c6ed843ea149dcb2b4546b1dee7d17cdd13580524f9445db3f4e802116)

# Imitation Project

This project is facilitated by Xavier Tao from 1ms.ai and CentraleSupélec. With a longer term vision of teaching engineering students the skills to train imitation policies on robots, this repo introduces a low-cost teleoperation pipeline for the Reachy 1 robot.

- **Scope:** S8 Semester (Feb-Jun 2025) at CentraleSupélec
- **Repository location:** https://github.com/jurmy24/reachy-imitation
- **Previous contributors (S7)**
  - Cheikh, Rouhou Mohamed
  - Maes, Alexis
  - Lafaury, Marin
  - Champaney, Matéo
- **Contributors (S8)**
  - Oldensand, Victor
  - Gandhi, Hugo
  - Lafaury, Marin
  - Ye, Yi
- **Supervisors**
  - Makarov, Maria
  - Valmorbida, Giorgio
  - Tao, Xavier

## Documentation

This codebase was used for the S8 project in the Robotics pole at CentraleSupélec. Therefore, it contains both code used to **demo** our work but also code used to generate graphs and figures for our report and analysis.

The `src/` folder contains the code that allows us to teleoperate the Reachy 1 robot from Pollen Robotics (seen in the image above). You can read their documentation for the robot itself here: https://pollen-robotics.github.io/reachy-2021-docs/sdk

The teleoperation setup consists of a human operator, the Reachy 1 robot, a Realsense D435i camera, and a computer connected via ethernet to Reachy and via usb to the camera.

<img width="934" alt="Screenshot 2025-06-09 at 15 35 41" src="https://github.com/user-attachments/assets/98f998f4-773f-4e95-ac4b-aa06f105a252" />


### How to setup coding environment

To run our teleoperation ("shadowing") pipeline do the following.

Set up the [uv](https://docs.astral.sh/uv/) package manager on your computer (our preferred python package manager). After cloning this repository, run the following commands in the root folder:

```bash
uv sync
source .venv/bin/activate # for mac/linux
.\.venv\Scripts\activate # for windows
```

### How to setup and connect to Reachy over ethernet

1. **Assign Reachy's Static IP** (not necessary if IP is already set)

   - Plug in a monitor and mouse to Reachy
   - Run these commands in the Linux terminal:
     ```bash
     sudo ip addr add 192.168.100.2/24 dev eno1
     sudo ip link set eno1 up
     ```
   - Verify the IP assignment using `ip -a` - check that the inet IP corresponding to eno1 matches what you assigned

2. **Set Windows Laptop Static IP** (not necessary if already set)

   - Go to Network and Internet Settings
   - Set IPv4 address to:
     - IPv4 address: `192.168.100.1`
     - Netmask: `255.255.255.0`
   - Verify settings in Windows PowerShell using `ipconfig`

3. **Reconnect Reachy to your computer via ethernet**

4. **Verify Connection**

   - SSH to Reachy from Windows PowerShell:
     ```bash
     ssh reachy@192.168.100.2
     ```
   - Enter Reachy password: `reachy`
   - Alternatively, you can ping Reachy to verify connection

5. **Connect to Hotspot**
   - Access Reachy's dashboard at `192.168.100.2:3972` in Chrome
     - Note: The port will always be 3972 regardless of IP
   - Navigate to WiFi settings
   - Select your computer's Mobile Hotspot WiFi
     - Enable Mobile Hotspot if not already on
   - Enter the hotspot password
   - If needed, enable Reachy's hotspot on the dashboard
   - Restart Reachy if connection issues persist

> **Note:** We use the `192.168.100.0/24` network. Any private IP range can be used.

### How to run shadowing pipeline

Once Reachy and the robot are connected via ethernet and you have tested that the `ssh` connection works, you can run the pipeline with the following command:

```bash
python -m src.main
```

> **Note:** Make sure that the IP address in `src/main.py` matches the one you set for Reachy above.
> **Note:** To modify which arm(s) you want Reachy to imitate you can set the `arm` parameter in `src/main.py`. You can even choose whether you want to calibrate the arm lengths or have Reachy watch the human with similar parameters.

## Other scripts

There is also a `scripts` folder with various files used to test specific parts of the pipeline, such as gripper code or inverse kinematics tests. You can explore these at your own leisure.
