import math

# position, decimal degrees
def add_distance(lat, lon, radius):
    R = 6378137.0 # earth’s radius, sphere

    # offsets in meters
    dn = radius
    de = radius

    # coordinate offsets in radians
    dLat = dn / R
    dLon = de / (R * math.cos(math.pi * lat / 180))

    # offset position, decimal degrees
    min_lat = lat - ( dLat * 180 / math.pi )
    min_lon = lon - ( dLon * 180 / math.pi )

    # offset position, decimal degrees
    max_lat = lat + ( dLat * 180 / math.pi )
    max_lon = lon + ( dLon * 180 / math.pi )

    return min_lat, min_lon, max_lat, max_lon
