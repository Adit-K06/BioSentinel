# FILE: actuator/virtual_membrane.py
#
# PURPOSE:
# Represents the physical membrane/valve response as pure software
# state, since no real hardware is being used for this build. This is
# the "2-way virtual membrane" described in the project docs.
#
# WHAT IT WILL CONTAIN:
# - A class holding an OPEN/CLOSED state and a timestamped log of state
#   changes (mirroring exactly what a real microcontroller command log
#   would look like).
# - An update function: if any gas is currently flagged as breached,
#   state becomes OPEN (vent); if all gases are safe, state becomes
#   CLOSED (normal capture loop).
#
# WHAT IT WILL DO:
# Gets called every timestep by the main app loop with the current
# breach status, and its state is what the dashboard displays as the
# "membrane / actuator state indicator."
