import numpy as np
from typing import List, Tuple
import heapq


class KDTreeIndex:
    """
    A K-D Tree for fast nearest-neighbor search.
    - Time Complexity: O(log N) average search time, O(N log N) build time.
    - Space Complexity: O(N*D).
    - Why? Much faster than brute force for low to medium dimensional data.
    - This implementation rebuilds the entire tree on each `add` call.
      A more complex implementation could support incremental insertions.
    """
    class _Node:
        def __init__(self, point: np.ndarray, chunk_id: str, axis: int, left=None, right=None):
            self.point = point
            self.chunk_id = chunk_id
            self.axis = axis
            self.left = left
            self.right = right

    def __init__(self, dim: int):
        self.dim = dim
        self.root = None
        self._all_points = []
        self._all_chunk_ids = []

    def add(self, vectors: np.ndarray, chunk_ids: List[str]):
        """
        Adds vectors and rebuilds the tree.
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch. Index expects {self.dim}, got {vectors.shape[1]}")
        
        # rebuilding the tree from scratch each time.
        self._all_points.extend(list(vectors))
        self._all_chunk_ids.extend(chunk_ids)
        
        # Combine points and IDs for sorting
        combined = list(zip(self._all_points, self._all_chunk_ids))
        self.root = self._build_tree(combined, depth=0)

    def _build_tree(self, points_with_ids: list, depth: int):
        if not points_with_ids:
            return None

        # Determine axis to split on
        axis = depth % self.dim
        
        # Sort points by the current axis and find the median
        points_with_ids.sort(key=lambda x: x[0][axis])
        median_idx = len(points_with_ids) // 2
        median_point, median_id = points_with_ids[median_idx]

        # Recursively build left and right subtrees
        node = self._Node(
            point=median_point,
            chunk_id=median_id,
            axis=axis,
            left=self._build_tree(points_with_ids[:median_idx], depth + 1),
            right=self._build_tree(points_with_ids[median_idx + 1:], depth + 1)
        )
        return node

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Finds the k-nearest neighbors to the query vector.
        """
        if self.root is None:
            return []
        
        # Use a min-heap to store the k-best candidates found so far.
        # We store (-distance, chunk_id) to simulate a max-heap, making it
        # easy to check against the farthest neighbor.
        best = []
        self._search_recursive(self.root, query_vector, k, best)
        
        # sort by actual distance, and return
        return sorted([ (chunk_id, -neg_dist) for neg_dist, chunk_id in best ], key=lambda x: x[1])

    def _search_recursive(self, node: _Node, query_vector: np.ndarray, k: int, best: list):
        if node is None:
            return
        
        # Determine which path to take down the tree
        axis = node.axis
        if query_vector[axis] < node.point[axis]:
            primary_child, secondary_child = node.left, node.right
        else:
            primary_child, secondary_child = node.right, node.left

        # Recurse down the primary path
        self._search_recursive(primary_child, query_vector, k, best)

        # Check the current node against the best candidates
        dist_sq = np.sum((node.point - query_vector)**2)
        
        if len(best) < k:
            heapq.heappush(best, (-dist_sq, node.chunk_id))
        elif dist_sq < -best[0][0]: # -best[0][0] is the largest distance (farthest neighbor)
            heapq.heapreplace(best, (-dist_sq, node.chunk_id))

        # Check if the secondary path could contain a closer point
        dist_to_plane = (query_vector[axis] - node.point[axis]) ** 2
        
        if dist_to_plane < -best[0][0] or len(best) < k:
            self._search_recursive(secondary_child, query_vector, k, best)
