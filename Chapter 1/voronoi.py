import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon



def modified_voronoi(vor: Voronoi) -> tuple[np.ndarray, np.ndarray]:
    """
    Take in a Voronoi data structure, and return the voronoi vertices
    and regions that are modified to include the points at infinity.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            if far_point.tolist() in new_vertices:
                new_region.append(new_vertices.index(far_point.tolist()))

            else:

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return np.asarray(new_vertices), np.asarray(new_regions)



def bounded_voronoi(points: np.ndarray, boundary: np.ndarray) -> list[list[np.ndarray]]:
    """
    Given a bounding path, calculate the Voronoi regions for each point specified

    Params:
        points          (N, 2) x-y coordinates of N points
        boundary        (M, 2) a path specifying the diagram boundary. 

    Returns
        A list of lists of regions, one set of regions for each point passed
    """

    vor = Voronoi(points)
    boundary_poly = Polygon(boundary)

    vertices, modified_regions = modified_voronoi(vor)

    new_regions = []

    for region in modified_regions:

        poly = Polygon(vertices[region]).intersection(boundary_poly)

        if isinstance(poly, MultiPolygon):
            new_regions.append([np.asarray(sub_poly.exterior.coords) for sub_poly in poly.geoms])

        else:
            new_regions.append([np.asarray(poly.exterior.coords)])


    return new_regions





