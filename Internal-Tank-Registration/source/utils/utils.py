import math

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    try:
        return math.degrees(
            math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
        )
    except Exception:
        return 0