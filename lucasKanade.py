import numpy as np
import cv2
import matplotlib.pyplot as plt

def inRanges(coords, limits):
    x, y = coords
    X_limit, Y_limit = limits
    return 0 <= x and x <= X_limit and 0 <= y and y <= Y_limit

def align_images(img1, img2):
    # Ensure images are 3-channel (BGR) before conversion
    if len(img1.shape) == 2:  # If grayscale, convert to 3-channel
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match features using FLANN Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the best matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 10.0)

    # Warp second image to align with the first
    aligned_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    return aligned_img2

def optical_flow(old_frame, new_frame, window_size, min_quality = 0.01):
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
    
    # Fix: Calculate temporal gradient properly as the difference between frames
    ft = new_frame_norm - old_frame_norm

    u = np.zeros(old_frame.shape)
    v = np.zeros(new_frame.shape)

    for feature in feature_list:
        j, i = feature.ravel()  # j is x (column), i is y (row)
        i, j = int(i), int(j)
        
        # Check if the window is completely within image boundaries
        if i-w < 0 or i+w+1 > old_frame.shape[0] or j-w < 0 or j+w+1 > old_frame.shape[1]:
            continue

        I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
        I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
        I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

        b = np.reshape(-I_t, (I_t.shape[0], 1))  # Negative temporal gradient for correct flow direction
        A = np.vstack((I_x, I_y)).T
        
        # Check if A is well-conditioned before solving
        if np.linalg.matrix_rank(A) < 2:
            continue
            
        try:
            U = np.matmul(np.linalg.pinv(A), b)
            # Remove debugging print statement
            u[i, j] = U[0][0]
            v[i, j] = U[1][0]
        except np.linalg.LinAlgError:
            # Skip if matrix is singular or other numerical issues
            continue

    return (u, v)

def drawOnFrame(frame, U, V, output_file):
    line_color = (0, 0, 255)  
    
    height, width = U.shape
    
    # Create a copy to avoid modifying the original
    result_frame = frame.copy()
    
    for i in range(height):
        for j in range(width):
            u, v = V[i][j], U[i][j]  # Swap u and v to match OpenCV coordinate system

            if abs(u) > 0.1 or abs(v) > 0.1:  # Only draw significant movements
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
            # Fix: correct coordinate order for OpenCV (x,y) = (j,i)
            start_point = (j, i)  # OpenCV uses (x,y) which is (column, row)
            
            u, v = V[i][j], U[i][j]  # Swap u and v to match OpenCV coordinate system
            
            if abs(u) > 0.1 or abs(v) > 0.1:  # Only draw significant movements
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

# Main execution code
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

    # Align the second image to the first
    aligned_img2 = align_images(img1_gray, img2_gray)
    
    # If aligned_img2 is BGR, convert to grayscale
    if len(aligned_img2.shape) == 3:
        aligned_img2_gray = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY)
    else:
        aligned_img2_gray = aligned_img2

    # Calculate optical flow
    U, V = optical_flow(img1_gray, aligned_img2_gray, 5, 0.05)  # Increased window size for better results

    # Visualize results
    aligned_img2_color = cv2.cvtColor(aligned_img2_gray, cv2.COLOR_GRAY2BGR)
    
    # Create two visualizations
    drawSeperately(img1_gray, aligned_img2_gray, U, V, "flow_comparison.png")
    result = drawOnFrame(aligned_img2_color, U, V, 'flow_overlay.png')
    
    print("Processing complete. Output saved to flow_comparison.png and flow_overlay.png")