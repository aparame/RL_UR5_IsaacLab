import cv2
import cv2.aruco as aruco
import numpy as np

def create_us_letter_charuco():
    # --- Configuration for US Letter ---
    # US Letter Size: 8.5 x 11 inches
    # 300 DPI (Dots Per Inch) is standard for high-quality printing
    DPI = 300
    page_width_in = 8.5
    page_height_in = 11
    
    # Convert to pixels for the image
    img_width = int(page_width_in * DPI)   # 2550 px
    img_height = int(page_height_in * DPI) # 3300 px
    
    # --- Board Layout ---
    # 5x7 grid fits comfortably with margins
    SQUARES_X = 5
    SQUARES_Y = 7
    
    # Physical sizes in meters (used for calibration later)
    # 35mm squares = 0.035m
    SQUARE_LENGTH = 0.035 
    # Marker is usually 70-80% of square size
    MARKER_LENGTH = 0.026 
    
    # Define Dictionary (ZED usually supports 4x4 or 6x6)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Create Board
    # Note: Modern OpenCV (4.7+) uses CharucoBoard class directly
    # Legacy OpenCV uses aruco.CharucoBoard_create(...)
    try:
        board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    except AttributeError:
        # Fallback for older OpenCV versions
        board = aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

    # Calculate margins to center the board on the US letter page
    # Board pixel width = squares_x * (pixels per meter?) 
    # Easier approach: generate image and let OpenCV handle scaling to our calculated resolution
    
    # Margin calculation:
    # 35mm * 5 cols = 175mm board width
    # US Letter width = 215.9mm
    # Margin total = 40.9mm -> ~20mm (0.8 inches) margin on each side
    margin_px = int(0.8 * DPI) 

    img = board.generateImage((img_width, img_height), marginSize=margin_px, borderBits=1)

    # Save
    filename = "charuco_us_letter_5x7_35mm.png"
    cv2.imwrite(filename, img)
    print(f"Generated {filename}")
    print(f"Dimensions: {img_width}x{img_height} px")
    print("PRINTING INSTRUCTION: Print at '100% Scale' or 'Actual Size' (Do not scale to fit)")

if __name__ == "__main__":
    create_us_letter_charuco()