labels = [
    "Coupe",
    "Motorcyclist",
    "Yield Sign",
    "SUV",
    "Wheelchair User",
    "Hatchback",
    "Pedestrian Detour Sign",
    "Construction Worker",
    "Information Sign",
    "Truck Cab",
    "Speed Limit Sign",
    "Bus",
    "Box Truck",
    "Cyclist",
    "Adult Pedestrian",
    "Bus Sign",
    "Vehicular Trailer",
    "School Bus",
    "Construction Barrier",
    "Police Car",
    "Police / Emergency Worker",
    "Street Sign",
    "Warning Sign",
    "Stop Sign",
    "Van",
    "Child Pedestrian",
    "Pickup Truck",
    "Truck",
    "Construction Vehicle",
    "Motorbike",
    "Articulated Bus",
    "Message Board Trailer",
    "Radar Speed Sign",
    "Parking Sign",
    "Sedan",
    "Traffic Light Trailer",
    "Directional Arrowboard",
]

colors = [
    "beige",
    "green",
    "yellow",
    "black",
    "white",
    "silver",
    "brown",
    "gray",
    "purple",
    "blue",
    "orange",
    "gold",
    "red",
]

label_hierarchy = {
    "Pedestrians": [
        "Adult Pedestrian", 
        "Child Pedestrian", 
        "Construction Worker", 
        "Police / Emergency Worker", 
        "Wheelchair User"
    ],
    "Cyclists and Motorbikes": [
        "Cyclist", 
        "Motorbike", 
        "Motorcyclist"
    ],
    "Cars": [
        "Sedan", 
        "SUV", 
        "Pickup Truck", 
        "Hatchback", 
        "Coupe", 
        "Van", 
        "Police Car", 
    ],
    "Truck / Bus Shaped": [
        "Box Truck", 
        "Bus", 
        "Truck", 
        "Articulated Bus", 
        "Vehicular Trailer", 
        "Truck Cab", 
        "School Bus"
    ],
    "Construction and Road Equipment": [
        "Construction Barrier", 
        "Directional Arrowboard", 
        "Construction Vehicle", 
        "Traffic Light Trailer", 
        "Message Board Trailer"
    ],
    "Traffic and Informational Signs": [
        "Information Sign", 
        "Street Sign", 
        "Parking Sign", 
        "Speed Limit Sign", 
        "Stop Sign", 
        "Bus Sign", 
        "Warning Sign", 
        "Radar Speed Sign", 
        "Yield Sign",
        "Pedestrian Detour Sign"
    ]
}

WAYMO_LABEL_TYPES = ['TYPE_UNKNOWN', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']