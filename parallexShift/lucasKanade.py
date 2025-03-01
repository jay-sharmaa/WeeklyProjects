import numpy as np
import cv2
import matplotlib.pyplot as plt

global points_img1, points_img2
points_img1, points_img2 = [], []

def inRanges(coords, limits):
    x, y = coords
    X_limit, Y_limit = limits
    return 0 <= x and x <= X_limit and 0 <= y and y <= Y_limit

def reset_points():
    global points_img1, points_img2
    points_img1, points_img2 = [], []

def resize_with_aspect_ratio(image, width):
    (h, w) = image.shape[:2]
    aspect_ratio = width / w
    new_dim = (width, int(h * aspect_ratio))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

# Mouse click event handler
def select_points(event, x, y, flags, param):
    global points_img1, points_img2, selecting_img1
    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting_img1:
            points_img1.append((x, y))
            print(f"Point selected in Image 1: {x}, {y}")
        else:
            points_img2.append((x, y))
            print(f"Point selected in Image 2: {x}, {y}")

# Function to manually align images
def align_images(img1, img2, width=600):
    global selecting_img1
    reset_points()
    img1_resized = resize_with_aspect_ratio(img1, width)
    img2_resized = resize_with_aspect_ratio(img2, width)
    
    selecting_img1 = True
    cv2.imshow("Select a Point in Image 1", img1_resized)
    cv2.setMouseCallback("Select a Point in Image 1", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    selecting_img1 = False
    cv2.imshow("Select a Point in Image 2", img2_resized)
    cv2.setMouseCallback("Select a Point in Image 2", select_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points_img1) < 1 or len(points_img2) < 1:
        print("A single point is needed for alignment.")
        return None
    
    p1 = np.array(points_img1[0]) * (img1.shape[1] / img1_resized.shape[1], img1.shape[0] / img1_resized.shape[0])
    p2 = np.array(points_img2[0]) * (img2.shape[1] / img2_resized.shape[1], img2.shape[0] / img2_resized.shape[0])
    
    translation_vector = p1 - p2
    translation_matrix = np.float32([[1, 0, translation_vector[0]], [0, 1, translation_vector[1]]])
    aligned_img2 = cv2.warpAffine(img2, translation_matrix, (img1.shape[1], img1.shape[0]))
    aligned_img2 = cv2.resize(aligned_img2, (400, 400))
    return aligned_img2


def optical_flow(old_frame, new_frame, window_size, min_quality = 0.01):
    old_frame = cv2.resize(old_frame, (400, 400))
    max_corners = 10000
    min_distance = 1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

    w = int(window_size/2)

    old_frame_norm = old_frame/255.0
    new_frame_norm = new_frame/255.0

    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])

    fx = cv2.filter2D(old_frame_norm, -1, kernel_x)
    fy = cv2.filter2D(old_frame_norm, -1, kernel_y)
    
    ft = new_frame_norm - old_frame_norm

    u = np.zeros(old_frame.shape)
    v = np.zeros(new_frame.shape)

    for feature in feature_list:
        j, i = feature.ravel()
        i, j = int(i), int(j)
        
        if i-w < 0 or i+w+1 > old_frame.shape[0] or j-w < 0 or j+w+1 > old_frame.shape[1]:
            continue

        I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
        I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
        I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

        b = np.reshape(-I_t, (I_t.shape[0], 1))
        A = np.vstack((I_x, I_y)).T
        
        if np.linalg.matrix_rank(A) < 2:
            continue
            
    
        U = np.matmul(np.linalg.pinv(A), b)
        u[i, j] = U[0][0]
        v[i, j] = U[1][0]
        

    return (u, v)

def drawOnFrame(frame, U, V, output_file):
    line_color = (0, 0, 255)  
    
    height, width = U.shape
    
    result_frame = frame.copy()
    
    for i in range(height):
        for j in range(width):
            u, v = V[i][j], U[i][j]

            if abs(u) > 0.1 or abs(v) > 0.1:
                end_j = int(round(j+u))
                end_i = int(round(i+v))
                
                if 0 <= end_i < frame.shape[0] and 0 <= end_j < frame.shape[1]:
                    result_frame = cv2.arrowedLine(result_frame, (j, i), (end_j, end_i), 
                                            line_color, thickness=2)
    
    cv2.imwrite(output_file, result_frame)
    return result_frame

def drawSeperately(old_frame, new_frame, U, V, output_file):
    displacement = np.ones_like(new_frame)
    if len(displacement.shape) == 3:
        displacement = np.ones((new_frame.shape[0], new_frame.shape[1], 3), dtype=np.uint8) * 255
    else:
        displacement = np.ones((new_frame.shape[0], new_frame.shape[1]), dtype=np.uint8) * 255
        
    line_color = (0, 0, 0)
    
    height, width = U.shape
    for i in range(height):
        for j in range(width):
            start_point = (j, i)
            
            u, v = V[i][j], U[i][j]
            
            if abs(u) > 0.1 or abs(v) > 0.1:
                end_point = (int(j+u), int(i+v))
                
                if inRanges((end_point[1], end_point[0]), (height-1, width-1)):
                    displacement = cv2.arrowedLine(displacement, start_point, end_point, line_color, thickness=1)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(old_frame, cmap="gray")
    axes[0].set_title("First Image")
    axes[1].imshow(new_frame, cmap="gray")
    axes[1].set_title("Second Image")
    
    if len(displacement.shape) == 3:
        axes[2].imshow(displacement)
    else:
        axes[2].imshow(displacement, cmap="gray")
    axes[2].set_title("Displacement Vectors")
    
    figure.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=200)

if __name__ == "__main__":
    img1 = cv2.imread("parallexShift/img1.jpg")
    if img1 is None:
        print("Error: Could not read img1.jpg. Check the file path.")
        exit()
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread("parallexShift/img2.jpg")
    if img2 is None:
        print("Error: Could not read img2.jpg. Check the file path.")
        exit()
    
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    aligned_img2 = align_images(img1_gray, img2_gray)
    
    if len(aligned_img2.shape) == 3:
        aligned_img2_gray = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY)
    else:
        aligned_img2_gray = aligned_img2

    U, V = optical_flow(img1_gray, aligned_img2_gray, 5, 0.05)

    aligned_img2_color = cv2.cvtColor(aligned_img2_gray, cv2.COLOR_GRAY2BGR)
    

    drawSeperately(img1_gray, aligned_img2_gray, U, V, "parallexShift/flow_comparison.png")
    result = drawOnFrame(aligned_img2_color, U, V, 'parallexShift/flow_overlay.png')
    
    print("Processing complete. Output saved to flow_comparison.png and flow_overlay.png")