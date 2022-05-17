import math


def get_FOV(fx, fy):
    FOVx = 2*math.atan(640/(2*fx)) * 180 / math.pi
    FOVy = 2*math.atan(480/(2*fy)) * 180 / math.pi
    return FOVx, FOVy
