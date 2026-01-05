import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import magiccube
from matplotlib.colors import to_rgba

# --- Cube visualization ---
def plot_cube(state, ax=None):
    face_order = ['U', 'L', 'F', 'R', 'B', 'D']
    color_map = {'W':'white','Y':'yellow','G':'green','B':'blue','O':'orange','R':'red','?':'grey'}
    face_pos = {'U':(0,3),'L':(3,0),'F':(3,3),'R':(3,6),'B':(3,9),'D':(6,3)}
    grid = [['grey']*12 for _ in range(9)]
    si = 0
    for f in face_order:
        r0,c0 = face_pos[f]
        for i in range(3):
            for j in range(3):
                grid[r0+i][c0+j] = color_map.get(state[si],'grey')
                si += 1
    #if ax is None:
    img_data = [[to_rgba(c) for c in row] for row in grid]
    top_face = [x[3:6] for x in img_data[:3]]
    bottom_face = [x[3:6] for x in img_data[6:9]]
    left_face = [x[:3] for x in img_data[3:6]]
    right_face = [x[6:9] for x in img_data[3:6]]
    front_face = [x[3:6] for x in img_data[3:6]]
    back_face = [x[9:12] for x in img_data[3:6]]
    ax[0, 0].imshow(top_face)
    ax[0, 1].imshow(bottom_face)
    ax[0, 2].imshow(left_face)
    ax[1, 0].imshow(right_face)
    ax[1, 1].imshow(front_face)
    ax[1, 2].imshow(back_face)
    labels = ["Top (Yellow)", "Bottom (White)", "Left (Red)", "Right (Orange)", "Front (Green)", "Back (Blue)"]
    for idx, sub_ax in enumerate(ax.flat):
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        sub_ax.set_title(labels[idx])
    return axes

# --- Color detection ---
COLOR_RANGES = {
    'red1': ([0,75,75],[4,255,255]),
    'red2': ([130,75,75],[180,255,255]),
    'orange':([5,100,100],[23,255,255]),
    'yellow':([24,100,100],[34,255,255]),
    'green':([35,100,100],[85,255,255]),
    'blue':([86,75,75],[129,255,255]),
    'white':([0,0,150],[180,75,255])
}
COLOR_TO_FACE = {'white':'W','red':'R','green':'G','yellow':'Y','orange':'O','blue':'B'}

def get_color(hsv_pixel):
    for color,(lower,upper) in COLOR_RANGES.items():
        lower=np.array(lower,dtype=np.uint8)
        upper=np.array(upper,dtype=np.uint8)
        if np.all(hsv_pixel>=lower) and np.all(hsv_pixel<=upper):
            return 'red' if color in ['red1','red2'] else color
    return 'unknown'

def get_dominant_color(region):
    pixels = region.reshape(-1,3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    _,_,centers = cv2.kmeans(pixels.astype(np.float32),1,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    return centers[0].astype(int)

def order_points(pts):
    rect=np.zeros((4,2),dtype="float32")
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    diff = np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]
    return rect

def preprocess_and_flatten(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray,240,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        raise Exception("No contours found")
    cube_contour = max(contours,key=cv2.contourArea)
    epsilon = 0.02*cv2.arcLength(cube_contour,True)
    approx = cv2.approxPolyDP(cube_contour,epsilon,True)
    if len(approx)!=4:
        raise Exception("Cube face not detected as quadrilateral")
    pts = approx.reshape(4,2)
    rect = order_points(pts)
    side = int(max([
        np.linalg.norm(rect[0]-rect[1]),
        np.linalg.norm(rect[1]-rect[2]),
        np.linalg.norm(rect[2]-rect[3]),
        np.linalg.norm(rect[3]-rect[0]),
    ]))
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(img,M,(side,side))
    return warped

def process_face_from_image(img):
    img = preprocess_and_flatten(img)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,w,_ = img.shape
    grid_h,grid_w = h//3,w//3
    sample_frac = 0.4
    face_colors=[]
    for row in range(3):
        for col in range(3):
            y1=int((row+0.5-sample_frac/2)*grid_h)
            y2=int((row+0.5+sample_frac/2)*grid_h)
            x1=int((col+0.5-sample_frac/2)*grid_w)
            x2=int((col+0.5+sample_frac/2)*grid_w)
            sticker_region = hsv[y1:y2,x1:x2]
            dominant_hsv = get_dominant_color(sticker_region)
            color_name = get_color(dominant_hsv)
            face_colors.append(color_name)
    return face_colors

def build_cube_state_from_images(uploaded_files):
    face_names = ['Top (yellow)','Left (red)','Front (green)','Right (orange)','Back (blue)','Bottom (white)']
    cube_state=[]
    for idx,uf in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(uf.read()),dtype=np.uint8)
        img = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode {uf.name}")
        face = process_face_from_image(img)
        cube_state.append(face)
        st.write(f"**{face_names[idx]} face colors:**")
        for i in range(3):
            st.write(' '.join(face[i*3:(i+1)*3]))
    state_str = ''
    for face in cube_state:
        for color in face:
            state_str += COLOR_TO_FACE.get(color,'?')
    if len(state_str)>=50:
        state_str = state_str[:49] + 'W' + state_str[50:]
    return state_str

def solve_rubiks_cube(state_str):
    try:
        cube = magiccube.Cube(3,state_str)
        solver = magiccube.BasicSolver(cube)
        return solver.solve()
    except Exception as e:
        return f"Cube solving failed: {e}"

# === Streamlit UI ===
st.title("Rubik's Cube Face Color Detection & Interactive Solution (MagicCube)")

st.markdown("""
Upload 6 images of the cube faces in this order: **Top (Yellow), Left (Red), Front (Green), Right (Orange), Back (Blue), Bottom (White)**.

Images should clearly show the face with minimal background for best detection.

For the yellow face, upload the image with **blue top**.

For the red, green, orange, and blue faces, upload the image with **yellow top**.

For the white face, upload the image with **green top**.

(as of right now, please try to make the faces as bright as possible!)
""")

uploaded_files = st.file_uploader(
    "Upload 6 images (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="rubik_faces"
)

# --- Initialize session state ---
if 'step_states' not in st.session_state: st.session_state['step_states'] = []
if 'step_moves' not in st.session_state: st.session_state['step_moves'] = []
if 'step_idx' not in st.session_state: st.session_state['step_idx'] = 0

if uploaded_files and len(uploaded_files)==6:
    try:
        state_str = build_cube_state_from_images(uploaded_files)

        if '?' in state_str:
            st.error("Some colors could not be detected. Please upload clearer images.")
        else:
            solution_moves = solve_rubiks_cube(state_str)

            # Convert CubeMove objects to strings
            if isinstance(solution_moves,str):
                moves_list = solution_moves.strip().split()
            elif isinstance(solution_moves,list):
                moves_list = [str(mv) for mv in solution_moves]
            else:
                moves_list = []

            st.session_state['step_moves'] = moves_list

            # Only compute step_states if not already done
            if not st.session_state['step_states']:
                step_states = [state_str]
                for i in range(len(moves_list)):
                    cube = magiccube.Cube(3,state_str)
                    cube.rotate(' '.join(moves_list[:i+1]))
                    step_states.append(cube.get())
                st.session_state['step_states'] = step_states
                st.session_state['step_idx'] = 0

            st.write("**Solution moves:**")
            st.markdown(
                f"<div style='font-family: monospace; background-color:#f0f2f6; padding:1em; border-radius:5px;'>{' '.join(moves_list)}</div>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Processing failed: {e}")
elif uploaded_files:
    st.warning("Please upload exactly 6 images.")

# --- Step-by-step viewer ---
if st.session_state['step_states']:
    states = st.session_state['step_states']
    moves = st.session_state['step_moves']
    idx = st.session_state['step_idx']

    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        if st.button("Previous", key="prev"):
            st.session_state['step_idx'] = max(0, idx-1)
    with col3:
        if st.button("Next", key="next"):
            st.session_state['step_idx'] = min(len(states)-1, idx+1)

    idx = st.session_state['step_idx']

    st.write(f"Step {idx} / {len(states)-1}")
    if idx>0:
        st.write(f"Move applied: **{moves[idx-1]}**")
    else:
        st.write("Initial cube state")

    fig,axes = plt.subplots(2, 3)
    fig.suptitle("Current cube state")
    plot_cube(states[idx], ax=axes)
    st.pyplot(fig)