import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon as Poly
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon as Poly

def Wstrs(x):
    m_AB = (yB - yA) / (xB - xA)
    b_AB = yA - m_AB * xA
    m_BC = (yC - yB) / (xC - xB)
    b_BC = yB - m_BC * xB
    return np.where(x <= xB, m_AB * x + b_AB, m_BC * x + b_BC)

# Function to find the center of a circle
def find_circle_center(point_a, point_b, point_c):
    mid_ab = [(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2]
    mid_ac = [(point_a[0] + point_c[0]) / 2, (point_a[1] + point_c[1]) / 2]
    perp_slope_ab = (point_a[0] - point_b[0]) / (point_b[1] - point_a[1])
    perp_slope_ac = (point_a[0] - point_c[0]) / (point_c[1] - point_a[1])
    y_intercept_ab = mid_ab[1] - perp_slope_ab * mid_ab[0]
    y_intercept_ac = mid_ac[1] - perp_slope_ac * mid_ac[0]
    center_x = (y_intercept_ac - y_intercept_ab) / (perp_slope_ab - perp_slope_ac)
    center_y = perp_slope_ab * center_x + y_intercept_ab
    return [center_x, center_y]

def check_intersection(point1, point2, D):
    x_A, y_A = point1
    x_B, y_B = point2
    # Check if D is within the range of x_A and x_B
    if x_A == x_B: return min(y_A, y_B) <= D <= max(y_A, y_B), (min(x_A, x_B), min(y_A, y_B))
    # Check if point D lies within the x-coordinate range of AB
    return min(x_A, x_B) <= D <= max(x_A, x_B), (D, (y_B - y_A) / (x_B - x_A) * (D - x_A) + y_A)

def Kozeny(z, z0):
    return((z**2-z0**2)/(2*z0))

def Casagrande(b, H, z0):
    zeta = np.arctan2((dam[3][1]-dam[2][1]),(dam[3][0]-dam[2][0]))
    d = z0+b-np.sqrt((z0+b)**2-(H**2)/np.sin(zeta)**2)
    if np.abs(zeta) <= np.pi/6:
        d = np.sqrt(b**2+H**2)-np.sqrt(b**2-(H/np.tan(zeta))**2)
    return(dam[3][0]-np.cos(zeta)*d,dam[3][1]-np.sin(zeta)*d)


def generate_equidistant_points_in_dam(vertices, pairs_list, distance_between_points = 0.6):
    # Create a Polygon object
    poly = Polygon(vertices, closed=True)

    # Get the bounding box of the dam
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)

    # Generate equidistant points within the bounding box
    x_coords = np.arange(min_x, max_x, distance_between_points)
    y_coords = np.arange(min_y, max_y, distance_between_points)

    # Create a meshgrid of coordinates
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))

    coords = [min(pairs_list, key=lambda x: abs(x[0] - i)) for i in x_coords]
    result = np.array([np.round(np.abs(point[1] - coords[i % len(coords)][1])* gamma/1000,1) for i, point in enumerate(points)])
    comparison_result = np.array([point[1] <= coords[i % len(coords)][1] for i, point in enumerate(points)])

    # Filter points inside the dam
    inside_dam = poly.contains_points(points)
    result = result[inside_dam & comparison_result]
    equidistant_points_u = points[inside_dam & comparison_result]
    equidistant_points_no_u = points[inside_dam & ~comparison_result]

    for p in range(0, len(equidistant_points_u)):
        ax.annotate(result[p], equidistant_points_u[p], textcoords="offset points", xytext=(10,-15), ha='center', color='white')
    for p in equidistant_points_no_u:
        ax.annotate('0', p, textcoords="offset points", xytext=(10,-15), ha='center', color='white')

    centers = []

    x, y = zip(*intersection_points)

    for i in range(int(len(x) / 2) - 1):
        centers.append((x[i] + (x[i + 2] - x[i]) / 2, y[i] + (y[i + 2] - y[i]) / 2))

    ph_line = [min(pairs_list, key=lambda x: abs(x[0] - i[0])) for i in centers]

    global u
    u = np.array([np.abs(ph_line[i][1] - center[1]* gamma/1000) if ph_line[i][1] >= center[1] else 0 for i, center in enumerate(centers)])

    norm = Normalize(vmin=result.min(), vmax=result.max())
    normalized_result = norm(result)

    sm = ScalarMappable( norm=norm)
    sm.set_array([])

    plt.scatter(equidistant_points_u[:, 0], equidistant_points_u[:, 1], c=normalized_result)
    plt.colorbar(sm, label='Result Values', ax=ax)
    plt.title('Color Map for Result Array')

# Streamlit app
st.title("Embankment dam with core analysis")
Core_on = st.sidebar.toggle('Core')

# **********************************************************************

H = 7
gamma = 997
dam = np.array([(2, 0), (7, 8), (9, 8.1), (14, 0)])
core = np.array([(5, 0), dam[1], dam[-2], (11, 0)])
soil = np.array([(0, -1), (0, 0), (15, 0), (15, -1)])

# **********************************************************************

# Sidebar with input elements
st.sidebar.header("Input Parameters")

# Water Height
H = st.sidebar.slider("Water Height", min_value=dam[0,1], max_value=dam[2,1], value=7.0)

# Rest of your code...


# Gamma
gamma = st.sidebar.number_input("Gamma", value=gamma)

# Dam Coordinates
with st.sidebar.expander("Dam Coordinates"):
    for i in range(len(dam)):
        col1, col2 = st.columns(2)
        dam[i][0] = col1.number_input(f"Px{i+1}", key=f"dam_px_{i+1}", value=dam[i][0], step=1.0)
        dam[i][1] = col2.number_input(f"Py{i+1}", key=f"dam_py_{i+1}", value=dam[i][1], step=1.0)

# Core Coordinates
if Core_on : 
    with st.sidebar.expander("Core Coordinates"):
        for i in range(len(core)):
            col1, col2 = st.columns(2)
            core[i][0] = col1.number_input(f"Px{i+1}", key=f"core_px_{i+1}", value=core[i][0], step=1.0)
            core[i][1] = col2.number_input(f"Py{i+1}", key=f"core_py_{i+1}", value=core[i][1], step=1.0)

# Soil Coordinates
with st.sidebar.expander("Soil Coordinates"):
    for i in range(len(soil)):
        col1, col2 = st.columns(2)
        soil[i][0] = col1.number_input(f"Px{i+1}", key=f"soil_px_{i+1}", value=soil[i][0], step=1)
        soil[i][1] = col2.number_input(f"Py{i+1}", key=f"soil_py_{i+1}", value=soil[i][1], step=1)

    # Add new point button
    if st.button("Add New Point to Soil"):
        new_point_x = st.number_input("New Point - X", step=1)
        new_point_y = st.number_input("New Point - Y", step=1)
        soil = np.vstack([soil, [new_point_x, new_point_y]])


# **********************************************************************

# Coordinates of A, B, C and O
#xA, yA = np.float64(dam[3][0]), np.float64(-1.3)
xA, yA = np.float64(dam[3][0]), np.float64(0)
#xB, yB = np.float64(13.1), np.float64(-1.3)
xB, yB = np.float64(13.1), np.float64(0)
#xC, yC = np.float64(dam[2][0]), np.float64(-1.3) #np.float64(5), np.float64(1.3)
xC, yC = np.float64(dam[2][0]), np.float64(0)
xO, yO = np.float64(12), np.float64(8)

E = ((dam[1][0]-dam[0][0])*( H - dam[0][1] + dam[0][0]*(dam[1][1]-dam[0][1])/(dam[1][0]-dam[0][0]))/(dam[1][1]-dam[0][1]), H)

# Points coordinates
water = np.array([(0, 0), dam[0], E, (0, E[1])])

z = np.linspace(0, H, 1000)
H = water[2][1]
y5 = water[2][0]
y1 = dam[0][0]
y4 = dam[-1][0]
b_prime = np.abs(y5 - y1)
b = np.abs(y5 - b_prime*0.3 - y4)
z0 = np.abs(np.sqrt(b**2+H**2)-b)+np.sqrt((y5**2)+H**2)-y5
if not Core_on: z0 = np.sqrt(np.abs(y5**2 + H**2)) - y5
y6 = y4-z0/2-Kozeny(z, z0)[0]
z = np.linspace(0, H, 1000)

st.write(z0, y5, H, Kozeny(y5, z0))

bande_1=np.array([  (core[0,0]-(core[1][0]-core[0][0])*( H - core[0][1] + core[0][0]*(core[1][1]-core[0][1])/(core[1][0]-core[0][0]))/(core[1][1]-core[0][1])+y5, 0),\
    (core[0,0],0),core[1],E])

bande_2=np.array([(core[-1,0],0),(core[-1,0]+(core[1][0]-core[0][0])*( H - core[0][1] + core[0][0]*(core[1][1]-core[0][1])/(core[1][0]-core[0][0]))/(core[1][1]-core[0][1])-y5, 0),\
    ((core[-1][0]-core[-2][0])*( H - core[-2][1] + core[-2][0]*(core[-1][1]-core[-2][1])/(core[-1][0]-core[-2][0]))/(core[-1][1]-core[-2][1])+(core[1][0]-core[0][0])*( H - core[0][1] +\
    core[0][0]*(core[1][1]-core[0][1])/(core[1][0]-core[0][0]))/(core[1][1]-core[0][1])-y5,H),\
    core[2]])

bande_3 = np.array([core[-1], dam[-1], (dam[-1,0], dam[-1,1] + 0.25), (core[-1,0], core[-1,1] + 0.25)])

# Specify the figure size
fig, ax = plt.subplots(figsize=(18, 8))  # Width: 18 inches, Height: 12 inches

ax.add_patch(Polygon(soil, closed=True, facecolor='brown', alpha=0.9, label='Soil'))
ax.add_patch(Polygon(water, closed=True, facecolor='lightblue', alpha=0.5, label='Water'))
ax.add_patch(Polygon(dam, closed=True, facecolor='gray', alpha=0.7, label='Dam'))
if Core_on : ax.add_patch(Polygon(core, closed=True, facecolor='brown', alpha=.5, label='Core'))
if Core_on : ax.add_patch(Polygon(bande_1, closed=True, facecolor='orange', alpha=1, label='Core'))
if Core_on : ax.add_patch(Polygon(bande_2, closed=True, facecolor='orange', alpha=1, label='Core'))
if Core_on : ax.add_patch(Polygon(bande_3, closed=True, facecolor='orange', alpha=1, label='Core'))

#------------------------------

R1, R2, R3=0.2787220909890784, 0.13652712021194733, 0.29135305166088526
#R1, R2, R3 = np.random.uniform(0, 1, 3)

# Determining S coordinates
xS = dam[1][0] + R1 * (dam[2][0] - dam[1][0])
yS = ((dam[2][1] - dam[1][1]) / (dam[2][0] - dam[1][0])) * (xS - dam[1][0]) + dam[1][1]

# Determining E coordinates
xE = xA #xA + R2 * (xC - xA)
yE = yA #Wstrs(xE)

# Determining M coordinates
xM, yM  = ((xC + xS) / 2, (yC + yS) / 2)

# Determining T coordinates
Kj = (yO - yM) / (xO - xM)
slope = 2.1592419085127095e-06
intercept = 6.296780919814355
xTmin, xTmax = xM, max(intercept, xM) + (R1+R2)/2 * slope
xT = xTmin + R3 * (xTmax - xTmin)
yT = Kj * (xT - xM) + yM

# Find the center of the circle
center = find_circle_center((xT, yT), (xS, yS), (xE, yE))
radius = np.linalg.norm(np.array(center) - np.array(list((xT, yT))))

# Define the angles for center
angle_start_E = np.degrees(np.pi * 2 + np.arctan2(yE - center[1], xE - center[0]))
angle_end_E = np.degrees(np.arctan2(yS - center[1], xS - center[0]))

angles = np.linspace(np.deg2rad(angle_end_E), np.deg2rad(angle_start_E), 100)
x_coordinates = center[0] + radius * np.cos(angles)
y_coordinates = center[1] + radius * np.sin(angles)
b_unit = np.abs(xS-xE)/10
values=[dam[3][0]] + [x_coordinates.max() - i * b_unit for i in range(0,10)] + [x_coordinates.min()]


intersection_points = []
slices = []
# Plot the vertical lines and mark the intersections
for D in values:
    for i in range(len(x_coordinates) -1):
          point1 = (x_coordinates[i], y_coordinates[i])
          point2 = (x_coordinates[i + 1], y_coordinates[i + 1])
          intersection, intersection_point = check_intersection(point1, point2, D)
          if intersection:
            intersection_points.append(intersection_point)
    for j in range(len(dam[1:])-1):
        point1 = dam[1:][j]
        point2 = dam[1:][j + 1]
        intersection, intersection_point = check_intersection(point1, point2, D)
        if intersection:
            intersection_points.append(intersection_point)
if intersection_points:
    del intersection_points[-2]
    for i in range(1,len(intersection_points) - 3, 2):
        polygon = np.array([intersection_points[i], intersection_points[i + 2], intersection_points[i + 3], intersection_points[i + 1]])
        ax.scatter(polygon[:, 0], polygon[:, 1], color='red', marker='o')
        polygon = Polygon(polygon, closed=True, edgecolor='black', linestyle='--', alpha=0.3)
        ax.add_patch(polygon)
        slices.append(polygon)

#------------------------------------------------------
L = Casagrande(b, H, z0)

if Core_on :
    pairs_list = np.array(list(zip(core[-1][0]-z0/2 - Kozeny(z, z0), z)))
    filtered_pairs = np.array(list(filter(lambda x: y5 <= x[0] <= L[0], pairs_list)))[::-1]

    x_cor, y_cor = zip(*filtered_pairs)
    ax.plot(x_cor, y_cor, color="lightblue")


elif not Core_on:
    pairs_list = np.array(list(zip(dam[-1][0]-z0/2 - Kozeny(z, z0), z)))
    filtered_pairs = np.array(list(filter(lambda x: y5 <= x[0] <= L[0], pairs_list)))[::-1]


    x_cor, y_cor = zip(*filtered_pairs)
    intersection_ph_line = [[x,y] for x,y in pairs_list if np.abs(((dam[3][1]-dam[2][1])/(dam[3][0]-dam[2][0]))*x+dam[3][1]-((dam[3][1]-dam[2][1])/(dam[3][0]-dam[2][0]))*dam[3][0]-y)<0.005][1]

    x_ph, y_ph = zip(*pairs_list)
    ax.plot(x_ph, y_ph, color="lightblue")
    ax.plot(x_cor[-200:], np.array(y_cor)[-200:] + np.linspace(0, 0.4, 200), color="lightblue", linestyle="--")
    ax.plot(x_cor[0:50], np.array(y_cor)[0:50] + np.linspace(0.6, 0, 50), color="lightblue", linestyle="--")

pairs_list[279:479][:, 1] += np.linspace(0.3, 0, 200)

generate_equidistant_points_in_dam(dam, pairs_list)

ax.add_patch(Arc(center, 2 * radius, 2 * radius, angle=0, theta1=angle_end_E, theta2=angle_start_E, color='purple'))

# Draw points
ax.scatter([xA], [yA], color='blue')
ax.scatter([xB], [yB], color='blue')
ax.scatter([xC], [yC], color='blue')
ax.scatter([xS], [yS], color='red')
ax.scatter([xE], [yE], color='green')
ax.scatter([xM], [yM], color='orange')
ax.scatter([xO], [yO], color='black')
ax.scatter([xT], [yT], color='purple')

# Add annotations
ax.annotate('A', (xA, yA), textcoords="offset points", xytext=(0,-15), ha='center', color='blue', zorder=50)
ax.annotate('B', (xB, yB), textcoords="offset points", xytext=(-4,5), ha='center', color='blue', zorder=50)
ax.annotate('C', (xC, yC), textcoords="offset points", xytext=(0,10), ha='center', color='blue', zorder=50)
ax.annotate('S', (xS, yS), textcoords="offset points", xytext=(10,-15), ha='center', color='red', zorder=50)
ax.annotate('E', (xE, yE), textcoords="offset points", xytext=(0,-15), ha='center', color='green', zorder=50)
ax.annotate('M', (xM, yM), textcoords="offset points", xytext=(0,10), ha='center', color='orange', zorder=50)
ax.annotate('O', (xO, yO), textcoords="offset points", xytext=(-10,10), ha='center', color='black', zorder=50)
ax.annotate('T', (xT, yT), textcoords="offset points", xytext=(0,-15), ha='center', color='purple', zorder=50)

ax.plot((dam[-1,0], core[-1,0]),(dam[-1,1], core[-1,1]), color="lightblue", zorder=50)


ax.scatter(L[0], L[1],marker='P', zorder=50, color="yellow")

ax.annotate('1',dam[0], textcoords="offset points", xytext=(10,-15), ha='center')
ax.annotate('2', dam[1], textcoords="offset points", xytext=(0,5), ha='center')
ax.annotate('3', dam[2], textcoords="offset points", xytext=(0,5), ha='center')
ax.annotate('4', dam[3], textcoords="offset points", xytext=(0,-15), ha='center')
ax.annotate('5', E, textcoords="offset points", xytext=(-10,5), ha='center')
ax.annotate('6', (y6, 0), textcoords="offset points", xytext=(-10,5), ha='center')

ax.axvline(y5, color='pink', linestyle='--')
ax.axvline(y1, color='pink', linestyle='--')
ax.axvline(y5 - b_prime*0.3, color='pink', linestyle='--')
ax.axvline(y6, color='pink', linestyle='--')
ax.axvline(y5 - b_prime*0.3 - z0, color='pink', linestyle='--')

ax.plot([xA, xB], [yA, yB], color='blue')
ax.plot([xB, xC], [yB, yC], color='blue')

# Flatten the meshgrid
x_O = np.meshgrid(np.linspace(12, 25, 20), np.linspace(8, 21, 20))[0].ravel()
y_O = np.meshgrid(np.linspace(12, 25, 20), np.linspace(8, 21, 20))[1].ravel()

# Plot the flattened meshgrid
#ax.scatter(x_O, y_O, color='blue', marker='.')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
plt.legend()
plt.subplots_adjust(wspace=0.3)
fig.tight_layout()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

st.pyplot(fig)
st.divider()

# Default values
rho = 2.87
g = 9.80665
c_prime = np.array([20, 20, 30, 30, 30, 20, 20, 20, 20, 20])
phi_prime = np.array([52, 52, 30, 30, 30, 52, 52, 52, 52, 52])
A = np.array([Poly(polygon.get_xy()).area for polygon in slices])
pts = intersection_points[1:][::2]
alpha = np.abs([np.arctan2(pts[i+1][1]-pts[i][1], pts[i+1][0]-pts[i][0]) for i in range(len(pts)-1)])
b_ = np.array([b_unit for i in range(10)])
gamma_ = rho * g
W = A * gamma_

# Input fields
rho_ = st.sidebar.number_input("Rho", value=rho, step=0.01)
c_prime = st.sidebar.text_input("c_prime", value=','.join(map(str, c_prime)))
phi_prime = st.sidebar.text_input("phi_prime", value=','.join(map(str, phi_prime)))

# Convert input values to appropriate types
rho = rho_
c_prime = np.array([float(val) for val in c_prime.split(',')])
phi_prime = np.array([float(val) for val in phi_prime.split(',')])

# Display the values in a table
table_data = {
    'Parameter': ['ρ', 'g', 'c\'', 'ϕ\''],
    'Value': [rho, g, c_prime, phi_prime]
}
st.table(table_data)

# Calculate the trigonometric functions
tan_phi = np.tan(np.deg2rad(phi_prime.astype(float)))
sin_alpha = np.sin(alpha)
cos_alpha = np.cos(alpha)
tan_alpha = np.tan(alpha)


def bishop_factor_of_safety_iteration(F = 2):
    # Precompute values that do not change during the loop
    Mobilizing_Moment = np.sum(W * sin_alpha)

    # Calculate the moment of resisting forces due to shear strength
    m_alpha = (cos_alpha * (1 + (tan_alpha) * tan_phi / F))
    Resisting_Moment = np.sum((b_ *(c_prime + (W / b_ - u) * tan_phi)) / m_alpha )

    # Calculation of new F
    F = Resisting_Moment / Mobilizing_Moment
    return(F)

F = 2
Factors = [F]
while True:
    FoS = bishop_factor_of_safety_iteration(F)
    if abs(FoS - F) < 1e-2 or FoS < 0:
        break
    F = FoS
    Factors.append(F)

st.write(f"Bishop Factor of Safety F:")
st.subheader(f"{F:.4f}")

fig, ax = plt.subplots(figsize=(18, 8))  # Width: 18 inches, Height: 12 inches
x = np.linspace(1,2.1,100)
y= np.array([bishop_factor_of_safety_iteration(i) for i in x])
plt.plot(x,y)
plt.ylim(1, 2)

#st.pyplot(fig)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
rerun_button = st.sidebar.button("Update")
if rerun_button:
    raise st.experimental_rerun()