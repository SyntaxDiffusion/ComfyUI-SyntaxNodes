import cv2
import numpy as np

def create_puzzle_mask(height, width, piece_size, thickness):
    puzzle_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(puzzle_mask, (0, 0), (puzzle_mask.shape[1] - 1, puzzle_mask.shape[0] - 1), 255, thickness=thickness)

    for i in range(0, puzzle_mask.shape[0], piece_size):
        for j in range(0, puzzle_mask.shape[1], piece_size):
            if i != 0:
                mid_j = (piece_size // 2)
                l_j = (piece_size // 3)
                r_j = (2 * piece_size) // 3
                offset = 0
                half_joint_size = (piece_size // 16)
                if (j // piece_size) % 2:
                    cv2.ellipse(puzzle_mask, (j + l_j, i), (l_j, 2), 0, 180, 280, 255, thickness=thickness)
                    cv2.ellipse(puzzle_mask, (j + r_j, i), (piece_size - r_j, 2), 0, 260, 360, 255, thickness=thickness)
                    offset = -2
                else:
                    cv2.ellipse(puzzle_mask, (j + l_j, i), (l_j, 2), 0, 80, 180, 255, thickness=thickness)
                    cv2.ellipse(puzzle_mask, (j + r_j, i), (piece_size - r_j, 2), 0, 0, 100, 255, thickness=thickness)
                    offset = 2
                if np.random.randint(1, 3) == 1:
                    cv2.ellipse(puzzle_mask, (j + mid_j, i + offset - 2 * half_joint_size), ((r_j - l_j) // 2, (5 * half_joint_size) // 2), 90, 40, 320, 255, thickness=thickness)
                else:
                    cv2.ellipse(puzzle_mask, (j + mid_j, i + offset + 2 * half_joint_size), ((r_j - l_j) // 2, (5 * half_joint_size) // 2), 270, 40, 320, 255, thickness=thickness)
            if j != 0:
                mid_i = (piece_size // 2)
                l_i = (piece_size // 3)
                r_i = (2 * piece_size) // 3
                offset = 0
                half_joint_size = (piece_size // 16)
                if (i // piece_size) % 2:
                    cv2.ellipse(puzzle_mask, (j, i + l_i), (2, l_i), 0, 270, 370, 255, thickness=thickness)
                    cv2.ellipse(puzzle_mask, (j, i + r_i), (2, l_i), 0, -10, 90, 255, thickness=thickness)
                    offset = 2
                else:
                    cv2.ellipse(puzzle_mask, (j, i + l_i), (2, l_i), 0, 170, 270, 255, thickness=thickness)
                    cv2.ellipse(puzzle_mask, (j, i + r_i), (2, l_i), 0, 90, 190, 255, thickness=thickness)
                    offset = -2
                if np.random.randint(1, 3) == 1:
                    cv2.ellipse(puzzle_mask, (j + offset - 2 * half_joint_size, i + mid_i), ((5 * half_joint_size) // 2, (r_i - l_i) // 2), 0, 40, 320, 255, thickness=thickness)
                else:
                    cv2.ellipse(puzzle_mask, (j + offset + 2 * half_joint_size, i + mid_i), ((5 * half_joint_size) // 2, (r_i - l_i) // 2), 180, 40, 320, 255, thickness=thickness)
    return puzzle_mask

def create(image, piece_size):
    row_leftover = image.shape[0] % piece_size
    col_leftover = image.shape[1] % piece_size
    image = cv2.resize(image, (image.shape[1] - col_leftover, image.shape[0] - row_leftover))
    puzzle_mask = create_puzzle_mask(image.shape[0], image.shape[1], piece_size=piece_size, thickness=2)
    puzzle_image = image.copy()
    puzzle_image = cv2.bitwise_or(puzzle_image, np.dstack([puzzle_mask] * 3))
    puzzle_mask = np.expand_dims(puzzle_mask, axis=2)
    return puzzle_image, puzzle_mask
