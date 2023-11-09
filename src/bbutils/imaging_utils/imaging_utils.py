import numpy as np

def surrounding_coords(coords):
    return [(coords[0] - 1, coords[1] - 1),(coords[0] - 1, coords[1] + 0),(coords[0] - 1, coords[1] + 1),
            (coords[0] + 0, coords[1] - 1),                               (coords[0] + 0, coords[1] + 1),
            (coords[0] + 1, coords[1] - 1),(coords[0] + 1, coords[1] + 0),(coords[0] + 1, coords[1] + 1)]

def get_image_groups(mask):
    queue = []
    traversed = np.zeros(mask.shape, dtype = bool)
    groups = []
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if not traversed[i][j]:
                # Queue of region pixels
                group = [(i, j)]
                queue = [(i, j)]
                while len(queue) > 0:
                    coords = queue.pop()
                    # Iterate over surrounding pixels
                    for x, y in surrounding_coords(coords):
                        if 0 <= x < len(mask) and  0 <= y < len(mask[0]):
                            if mask[x][y] and not traversed[x][y]:
                                queue.append((x, y))
                                group.append((x,y))
                                traversed[x, y] = True
                groups.append(mask[np.array(group).transpose()])
    return groups