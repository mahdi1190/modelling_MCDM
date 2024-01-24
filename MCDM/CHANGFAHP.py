# Chang's Fuzzy AHP Method: Consolidated Python Script
import numpy as np
class ChangFuzzyAHP:
    
    def __init__(self):
        self.systems = ["Hydrogen", "Biomass", "Natural Gas"]
    
    @staticmethod
    def normalize_tfn(matrix):
        """Normalize a TFN matrix."""
        column_sums = matrix.sum(axis=0)
        return matrix / column_sums
    
    @staticmethod
    def geometric_mean_tfn(matrix):
        """Compute the geometric mean for a TFN matrix."""
        n = matrix.shape[0]
        gm_list = []
        for i in range(n):
            tfn_product = matrix[i].prod()
            gm_list.append((tfn_product**(1/3), tfn_product**(1/3), tfn_product**(1/3)))
        return np.array(gm_list)
    
    @staticmethod
    def defuzzify_center_of_gravity(tfn):
        """Defuzzify a TFN using the center of gravity method."""
        return sum(tfn) / 3
    
    @staticmethod
    def chang_extent_analysis(matrix, criterion_weight):
        """Compute weights using Chang's extent analysis method."""
        extents = matrix.sum(axis=1)
        n = matrix.shape[0]
        S = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if extents[i, 0] / extents[j, 1] <= extents[i, 1] and extents[i, 1] <= extents[j, 2] / extents[i, 2]:
                    S[i] += 1
        S[S == 0] = 1e-10
        M = n / S
        final_weights = M / sum(M) * criterion_weight
        return final_weights
    
    def rank_systems(self, criteria_weights, matrices):
        """Rank systems based on Chang's method."""
        total_weights = np.zeros(3)
        for i in range(len(matrices)):
            weights = self.chang_extent_analysis(matrices[i], criteria_weights[i])
            total_weights += weights
        ranked_systems = [x for _, x in sorted(zip(total_weights, self.systems), reverse=True)]
        return ranked_systems

# Example Usage for Chang's method
chang_fahp = ChangFuzzyAHP()

# Criteria matrix
criteria_matrix_example = np.array([
    [(1,1,1), (1,2,3), (3,4,5)],
    [(1/3, 0.5, 1), (1,1,1), (2,3,4)],
    [(1/5, 0.25, 1/3), (1/4, 0.33, 0.5), (1,1,1)]
])

# Pairwise comparison matrices
safety_matrix_example = np.array([
    [(1,1,1), (1,1,1), (2,3,4)],
    [(1,1,1), (1,1,1), (2,3,4)],
    [(1/4, 0.33, 0.5), (1/4, 0.33, 0.5), (1,1,1)]
])
economics_matrix_example = np.array([
    [(4,5,6), (1,2,3), (2,3,4)],
    [(1/3, 0.5, 1), (1,1,1), (1,1,1)],
    [(1/4, 0.33, 0.5), (1,1,1), (1,1,1)]
])
emissions_matrix_example = np.array([
    [(1,1,1), (4,5,6), (4,5,6)],
    [(1/6, 0.2, 1/4), (1,1,1), (1,1,1)],
    [(1/6, 0.2, 1/4), (1,1,1), (1,1,1)]
])

matrices_example = [safety_matrix_example, economics_matrix_example, emissions_matrix_example]

# Computing criteria weights for Chang's method
criteria_weights_chang_example = [chang_fahp.defuzzify_center_of_gravity(w) for w in chang_fahp.normalize_tfn(chang_fahp.geometric_mean_tfn(criteria_matrix_example))]

# Computing system rankings using Chang's method
ranked_systems_chang_example = chang_fahp.rank_systems(criteria_weights_chang_example, matrices_example)

print(ranked_systems_chang_example)
