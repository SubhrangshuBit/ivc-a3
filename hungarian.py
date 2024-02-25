# Heavily inspired by the following blog/article and code snippets
# Blog/Article
# 1. https://brc2.com/the-algorithm-workshop/
# 2. https://www.netlib.org/utk/lsi/pcwLSI/text/node222.html
# Code:
# 1. https://github.com/tdedecko/hungarian-algorithm/blob/master/hungarian.py
# 2. https://www.feiyilin.com/munkres.html#example3
# 3. https://github.com/Smorodov/Multitarget-tracker/tree/master?tab=readme-ov-file


# Import libraries
import numpy as np

NORMAL = 0
STAR = 1
PRIME = 2

# Start of uncovered primed zeros
uncovered_prime_zero = (0,0)

# Function to compute distance between two points
def distance(point1: float, point2: float):
    """Calculates euclidean distance between two points

    Args:
        point1 (float): First point
        point2 (float): Second point

    Returns:
        _type_: _description_
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x_min1, y_min1, w1, h1 = box1["x_min"], box1["y_min"], box1["width"], box1["height"]
    x_min2, y_min2, w2, h2 = box2["x_min"], box2["y_min"], box2["width"], box2["height"]
    
    x_max1, y_max1 = x_min1 + w1, y_min1 + h1
    x_max2, y_max2 = x_min2 + w2, y_min2 + h2

    xA = max(x_min1, x_min2)
    yA = max(y_min1, y_min2)
    xB = min(x_max1, x_max2)
    yB = min(y_max1, y_max2)

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def build_cost_matrix(frame1: dict, frame2: dict) -> np.ndarray:
    """Builds a cost matrix that computes the Euclidean distances between all objects
    in one frame to the objects in another frame.

    Args:
        frame1 (dict): Frame - 1
        frame2 (dict): Frame - 2

    Returns:
        np.ndarray: Cost matrix of shape (n, m) with pairwise Euclidean distances.
    """
    n = len(frame1)
    m = len(frame2)

    k = max(n, m)
    cost_matrix = np.zeros((k, k))
    
    for i, obj in enumerate(frame1):
        x1, y1 = obj["x_min"], obj["y_min"]
        x2, y2 = x1 + obj["width"], y1 + obj["height"]
        
        for j, next_obj in enumerate(frame2):
            x11, y11 = next_obj["x_min"], next_obj["y_min"]
            x22, y22 = x11 + next_obj["width"], y11 + next_obj["height"]

            cost_matrix[i, j] = min(distance([x1, y1], [x11, y11]), distance([x2, y2], [x22, y22])) \
                                + calculate_iou(obj, next_obj) 
    
    return cost_matrix

# Utilities
def find_star_zero_in_col(col_index: int, zero_mask: np.ndarray):
    r = np.where(zero_mask[:, col_index] == STAR)[0]
    if len(r):
        return r[0]
    return None

def find_prime_zero_in_row(row_index: int, zero_mask: np.ndarray):
    c = np.where(zero_mask[row_index, :] == PRIME)[0]
    if len(c):
        return c[0]
    return None

def find_smallest(cost_matrix: np.ndarray, row_cover: np.ndarray, col_cover: np.ndarray):
    # Add maximum value and then find the minimum
    max_val = cost_matrix.max()
    updated_cost_matrix = cost_matrix + row_cover[:, np.newaxis]*max_val
    updated_cost_matrix += col_cover*max_val
    return updated_cost_matrix.min()

# Hungarian Algorithm
def hungarian_assignment(cost_matrix: np.ndarray):
    """Implementation of Hungarian algorithm

    Args:
        cost_matrix (np.ndarray): Cost matrix
    """
    # Preliminaries
    # Shape
    n_rows, n_cols = cost_matrix.shape

    # Mask for zeros: Starred, Primed, Normal
    zero_mask = np.zeros_like(cost_matrix)

    # Covers: Row (Horizontal line) and Column (Vertical line)
    row_cover = np.zeros(cost_matrix.shape[0]).astype(int)
    col_cover = np.zeros(cost_matrix.shape[1]).astype(int)

    # Step 0:
    def step0(cost_matrix: np.ndarray, zero_mask: np.ndarray, row_cover: np.ndarray, col_cover: np.ndarray):
        rows, cols = cost_matrix.shape

        # Step 2: Find the lowest cost entry in each row and subtract it from all other entries in that row
        for r_index, row in enumerate(cost_matrix):
            cost_matrix[r_index] -= row.min()

        # Step 3: Mark starred zeros, each row or col will have at most 1
        for row in range(rows):
            for col in range(cols):
                if cost_matrix[row, col] == 0 and row_cover[row] == 0 and col_cover[col] == 0:
                    zero_mask[row, col] = STAR
                    row_cover[row] = 1
                    col_cover[col] = 1
        return cover_starred_zeros(cost_matrix, zero_mask, row_cover, col_cover)

    def cover_starred_zeros(mat: np.ndarray, zero_mask: np.ndarray, row_cover: np.ndarray, col_cover: np.ndarray):
        rows, cols = mat.shape
        
        # Erase all marks
        row_cover.fill(0)
        col_cover.fill(0)

        # Erase primed zeros
        zero_mask[zero_mask == PRIME] = 0

        # Cover all columns corresponding to starred zeros
        starred_zeros = (zero_mask == STAR)
        col_cover = (starred_zeros.sum(axis=0) > 0).astype(int)

        if col_cover.sum() >= rows or col_cover.sum() >= cols:
            return zero_mask
        
        return step1(cost_matrix, zero_mask, row_cover, col_cover)

    # Step 1:
    def step1(cost_matrix, zero_mask, row_cover, col_cover):
        
        all_zeros = (cost_matrix == 0).astype(int)
        # all row covered zeros are masked as 1
        uncovered_zeros = all_zeros * (1 - row_cover[:, np.newaxis])
        # all col covered zeros are masked as 1
        uncovered_zeros *= (1 - col_cover)

        while True:
            # find an uncovered zero
            
            row, col = np.unravel_index(np.argmax(uncovered_zeros), uncovered_zeros.shape)
            
            if uncovered_zeros[row, col] == 0: # no uncovered zero
                return step3(cost_matrix, row_cover, col_cover)
            
            # else prime the uncovered zero
            zero_mask[row, col] = PRIME
            # check if there is any starred zero in the row
            if np.any(zero_mask[row, :] == STAR):
                col = np.where(zero_mask[row, :] == STAR)[0]
                
                # cover row
                row_cover[row] = 1
                # uncover the column
                col_cover[col] = 0
                # Update covered zeros mask
                uncovered_zeros[:, col] = all_zeros[:, col] * (1 - row_cover.reshape(-1, 1))
                uncovered_zeros[row] = 0
            else:
                uncovered_prime_zero = (row, col)
                return step2(uncovered_prime_zero, zero_mask)

    # Step 2:
    def step2(uncovered_prime_zero: tuple, zero_mask: np.ndarray):
        
        
        # Sequence of alternating prime and starred zeros
        # z0 = Uncovered prime zero
        line_count = 1
        lines = np.zeros((n_rows+n_cols, 2)).astype(int)
        lines[0, :] = uncovered_prime_zero

        # Alternating sequence of finding a starred and a prime zero
        while True:
            # (1) Find starred zero in same column as z0
            zero_r = find_star_zero_in_col(lines[line_count-1, 1], zero_mask)
            if zero_r is None:
                break    
            # Append sequence
            line_count += 1
            lines[line_count-1, :] = [zero_r, lines[line_count-2, 1]]
            
            # (2) Find primed zero in same row as z1
            zero_c = find_prime_zero_in_row(lines[line_count-1, 0], zero_mask)
            # Append sequence
            line_count += 1
            lines[line_count-1, :] = [lines[line_count-2, 0], zero_c]

        # Unstar the starred zeros and star the primed zeros
        for l in range(line_count):
            if zero_mask[lines[l, 0], lines[l, 1]] == STAR:
                zero_mask[lines[l, 0], lines[l, 1]] = NORMAL
            else:
                zero_mask[lines[l, 0], lines[l, 1]] = STAR

        return cover_starred_zeros(cost_matrix, zero_mask, row_cover, col_cover)      

    # Step 3:
    def step3(cost_matrix: np.ndarray, row_cover: np.ndarray, col_cover: np.ndarray):
        
        
        
        # Generate more distinct zeros in the cost matrix by adding/subtracting constants to row and column, 
        # At same time make sure all elements are not negative
        min_val = find_smallest(cost_matrix, row_cover, col_cover)
        # Add to covered rows
        cost_matrix += row_cover[:, np.newaxis] * min_val
        # Subtract from uncovered cols
        cost_matrix -= (1 - col_cover) * min_val
        return step1(cost_matrix, zero_mask, row_cover, col_cover)
    
    zero_mask = step0(cost_matrix, zero_mask, row_cover, col_cover)
    return zero_mask